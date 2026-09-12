# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Connectivity check for the configured planner backend.

This is the single implementation behind both the ``rpent-check-llm`` console
script and the Dashboard's ``POST /api/llm/check`` route. It sends the smallest
real request each backend supports — no tools, no images, no ``Toolkit``, no
robot runtime, no output directory — and classifies the outcome.

The layer is deliberately UI-agnostic: it reports a machine-readable ``status``
plus structured credential metadata and the provider's verbatim ``detail``.
Human-readable remediation text belongs to the CLI and the Dashboard, not here.
"""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rpent.utils.logging import get_logger

logger = get_logger("check")

# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

#: The backend replied. Everything else is a failure.
STATUS_OK = "ok"
#: No model id, or an ``api`` model id without a ``provider:`` prefix.
STATUS_MISSING_CONFIG = "missing_config"
#: The provider prefix is not one pydantic-ai can resolve.
STATUS_UNSUPPORTED_PROVIDER = "unsupported_provider"
#: The backend's credential env var is unset or empty.
STATUS_MISSING_API_KEY = "missing_api_key"
#: The provider rejected the credential (HTTP 401 / 403).
STATUS_AUTH_FAILED = "auth_failed"
#: The provider was reached and rejected the model id (HTTP 400).
STATUS_INVALID_MODEL = "invalid_model"
#: DNS, connection, TLS failure, or a timeout.
STATUS_NETWORK_ERROR = "network_error"
#: The provider was reached and refused (404, 429, 5xx, ...).
STATUS_PROVIDER_ERROR = "provider_error"
#: SDK missing, child process failure, or anything unclassified.
STATUS_SDK_ERROR = "sdk_error"

#: Every status this module can return, in rough severity order.
CHECK_STATUSES = (
    STATUS_OK,
    STATUS_MISSING_CONFIG,
    STATUS_UNSUPPORTED_PROVIDER,
    STATUS_MISSING_API_KEY,
    STATUS_AUTH_FAILED,
    STATUS_INVALID_MODEL,
    STATUS_NETWORK_ERROR,
    STATUS_PROVIDER_ERROR,
    STATUS_SDK_ERROR,
)

#: Planner backends this module can probe.
CHECK_PLANNERS = ("api", "claude_code", "codex")

#: Backends whose endpoint comes from an env var, not from ``--base-url``.
#: ``build_planner`` forwards ``base_url`` to the api model alone, so both
#: CLIs reject the flag for these rather than dropping it silently — a check
#: that honoured an endpoint the run ignores would be testing the wrong
#: thing.
BASE_URL_ENV_BY_PLANNER = {
    "claude_code": "ANTHROPIC_BASE_URL",
    "codex": "CODEX_BASE_URL",
}

#: Diagnostic timeouts, deliberately independent of the 1200s run default.
DEFAULT_TIMEOUT_S = {"api": 30, "claude_code": 90, "codex": 90}

#: Budget for an SDK backend running on a CLI login rather than an env var.
#: Shorter than the default because a diagnostic should not hold the terminal
#: for a minute and a half, but not much shorter: a cold Codex probe against a
#: gateway was measured at 15.3s, so a budget near that turns a slow success
#: into a false failure.
CLI_LOGIN_TIMEOUT_S = 25

#: Best-effort CLI login markers for the two child-process SDK backends. Both
#: accept an interactive login instead of an env var, so absence is only a
#: hint that no credential exists — never a verdict. It shortens the probe's
#: budget; it never skips the probe.
_CLI_LOGIN_FILES = {
    "claude_code": ("~/.claude/.credentials.json",),
    "codex": ("~/.codex/auth.json",),
}

#: macOS keeps the Claude Code credential in the login Keychain instead of in
#: a file, so the file check alone reports a logged-in Mac user as having no
#: credential. The lookup asks only whether the item exists — it never reads
#: the secret, so it does not prompt for the Keychain password.
_KEYCHAIN_SERVICES = {"claude_code": "Claude Code-credentials"}

#: The smallest prompt that still proves the model answered.
PROBE_PROMPT = "Reply with the single word: ok"

#: Output cap for the probe. Large enough for a word, small enough to be free.
PROBE_MAX_TOKENS = 16

#: Advisory only: names the env var in an error before pydantic-ai is asked to
#: resolve it. An unmapped prefix skips the pre-flight and falls through to
#: ``UserError`` classification, so this table can never reject a provider it
#: does not know about. Mirrors docs/source-en/.../usage/configure_planner.rst.
_API_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openai-chat": "OPENAI_API_KEY",
}
_API_BASE_URL_ENV = {
    "anthropic": "ANTHROPIC_BASE_URL",
    "openai": "OPENAI_BASE_URL",
    "openai-chat": "OPENAI_BASE_URL",
}

#: Credential env vars for the two child-process SDK backends. Both accept an
#: existing CLI login instead, so a missing var is reported but not fatal.
_CLAUDE_CODE_KEY_ENV = "ANTHROPIC_API_KEY"
_CLAUDE_CODE_BASE_URL_ENV = "ANTHROPIC_BASE_URL"
_CODEX_KEY_ENV = "CODEX_API_KEY"
_CODEX_BASE_URL_ENV = "CODEX_BASE_URL"

#: Cap on how much provider error text is carried in ``detail``.
_DETAIL_LIMIT = 2000

#: Credential env vars whose *values* are redacted out of any reported text.
#: Provider error bodies and SDK stderr can echo the key that was sent, and a
#: check report is exactly the kind of output that gets pasted into an issue.
_SECRET_ENV_NAMES = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "CODEX_API_KEY",
    "RPENT_CODEX_PROVIDER_KEY",
)

#: Values shorter than this are too generic to redact without mangling text.
_MIN_SECRET_LEN = 8


@dataclass(frozen=True, slots=True)
class LlmCheckRequest:
    """One connectivity check to perform.

    Attributes:
        planner: Backend to probe; one of :data:`CHECK_PLANNERS`.
        model: Model id. Required (and provider-prefixed) for ``api``;
            optional for ``claude_code`` and ``codex``.
        base_url: Base URL overriding the backend's own env var.
        timeout_s: Wall-clock cap. Defaults per backend when ``None``.
    """

    planner: str = "api"
    model: str | None = None
    base_url: str | None = None
    timeout_s: int | None = None

    def resolved_timeout_s(self) -> int:
        """Return the effective timeout, applying the per-backend default."""
        if self.timeout_s is not None:
            return int(self.timeout_s)
        return DEFAULT_TIMEOUT_S.get(self.planner, 30)


@dataclass(frozen=True, slots=True)
class LlmCheckResult:
    """The structured outcome of one check.

    Attributes:
        ok: True only when the backend replied with non-empty text.
        status: One of :data:`CHECK_STATUSES`.
        planner: The backend that was probed.
        model: The model id that was used, when known.
        credential_env: Name of the credential env var consulted, when known.
            The value is never captured.
        credential_present: Whether that env var was set and non-empty.
        base_url: The base URL override in effect, when given.
        base_url_env: Name of the env var that supplies the base URL otherwise.
        detail: Verbatim error text from the provider or SDK. Empty on success.
        reply: The model's reply text on success.
        latency_s: Seconds spent on the request itself, when one was made.
    """

    ok: bool
    status: str
    planner: str
    model: str | None = None
    credential_env: str | None = None
    credential_present: bool | None = None
    base_url: str | None = None
    base_url_env: str | None = None
    detail: str = ""
    reply: str = ""
    latency_s: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view for ``--json`` and the HTTP route."""
        payload: dict[str, Any] = {
            "ok": self.ok,
            "status": self.status,
            "planner": self.planner,
            "model": self.model,
            "credential_env": self.credential_env,
            "credential_present": self.credential_present,
            "base_url": self.base_url,
            "base_url_env": self.base_url_env,
            "detail": self.detail,
            "reply": self.reply,
            "latency_s": self.latency_s,
        }
        if self.extra:
            payload.update(self.extra)
        return payload


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def check_llm(request: LlmCheckRequest) -> LlmCheckResult:
    """Probe the configured planner backend and classify the outcome.

    Never raises: every failure is returned as a typed
    :class:`LlmCheckResult`, so callers do not each need their own error
    taxonomy.

    This function is synchronous and starts its own event loop, so it must be
    called from a thread with no running loop. FastAPI's threadpool for ``def``
    routes satisfies this; calling it from inside a coroutine does not.

    Args:
        request: The backend, model, base URL, and timeout to check.

    Returns:
        The structured outcome. ``result.ok`` is the single success signal.
    """
    planner = request.planner
    if planner not in CHECK_PLANNERS:
        return LlmCheckResult(
            ok=False,
            status=STATUS_MISSING_CONFIG,
            planner=planner,
            model=request.model,
            detail=(
                f"unknown planner {planner!r}; expected one of "
                f"{', '.join(CHECK_PLANNERS)}"
            ),
        )

    if _running_loop():
        return LlmCheckResult(
            ok=False,
            status=STATUS_SDK_ERROR,
            planner=planner,
            model=request.model,
            detail=(
                "check_llm() is synchronous and cannot run inside an active "
                "event loop; call it from a worker thread instead."
            ),
        )

    checker = {
        "api": _check_api,
        "claude_code": _check_claude_code,
        "codex": _check_codex,
    }[planner]

    # The budget is logged where it is resolved: the SDK backends shorten it
    # when no credential is in sight, so it cannot be stated up front here.
    logger.info(
        "checking planner %s (model=%s)",
        planner,
        request.model or "<backend default>",
    )
    try:
        return checker(request)
    except Exception as exc:  # noqa: BLE001 - the contract is to never raise
        return LlmCheckResult(
            ok=False,
            status=STATUS_SDK_ERROR,
            planner=planner,
            model=request.model,
            detail=_describe(exc),
        )


def _running_loop() -> bool:
    """Return True when the calling thread already drives an event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def redact_secrets(text: str) -> str:
    """Replace any configured credential value with its variable name.

    Args:
        text: Text that may embed a credential, such as a provider error body.

    Returns:
        The text with every known key value replaced by ``<VAR_NAME>``.
    """
    for env_name in _SECRET_ENV_NAMES:
        value = os.environ.get(env_name)
        if value and len(value) >= _MIN_SECRET_LEN and value in text:
            text = text.replace(value, f"<{env_name}>")
    return text


def _has_cli_login(planner: str) -> bool:
    """Whether a CLI login file exists for ``planner`` (best effort).

    Args:
        planner: The backend being probed.

    Returns:
        True when a known login file is present. False means "no evidence of
        a login", not "not logged in": a backend may keep its credential
        somewhere this module does not look.
    """
    for candidate in _CLI_LOGIN_FILES.get(planner, ()):
        try:
            if Path(candidate).expanduser().is_file():
                return True
        except OSError:
            continue
    return _has_keychain_login(planner)


def _has_keychain_login(planner: str) -> bool:
    """Whether the macOS Keychain holds a login item for ``planner``.

    Existence only: the secret is never read, so no Keychain prompt appears.

    Args:
        planner: The backend being probed.

    Returns:
        True when the Keychain item exists. Always False off macOS.
    """
    service = _KEYCHAIN_SERVICES.get(planner)
    if service is None or sys.platform != "darwin":
        return False
    try:
        completed = subprocess.run(
            ["security", "find-generic-password", "-s", service],
            capture_output=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def _sdk_probe_budget(request: LlmCheckRequest, *, cli_login_only: bool) -> int:
    """Return the probe timeout, shortened when only a CLI login backs it.

    An explicit ``--timeout-s`` always wins: the caller asked for a specific
    budget and must not have it silently overridden.

    Args:
        request: The check being performed.
        cli_login_only: Whether the backend is running on a CLI login because
            its credential env var is unset.

    Returns:
        The timeout in seconds.
    """
    if request.timeout_s is not None:
        budget = int(request.timeout_s)
    elif cli_login_only:
        budget = CLI_LOGIN_TIMEOUT_S
    else:
        budget = request.resolved_timeout_s()
    logger.info(
        "probe budget: %ds%s",
        budget,
        " (running on a CLI login, no credential env var)" if cli_login_only else "",
    )
    return budget


def _describe(exc: BaseException) -> str:
    """Render an exception as ``TypeName: message``, redacted and capped."""
    text = redact_secrets(f"{type(exc).__name__}: {exc}")
    if len(text) > _DETAIL_LIMIT:
        return text[:_DETAIL_LIMIT] + " …[truncated]"
    return text


# ---------------------------------------------------------------------------
# api planner
# ---------------------------------------------------------------------------


def _check_api(request: LlmCheckRequest) -> LlmCheckResult:
    """Probe the pydantic-ai ``api`` planner with one minimal request."""
    model = (request.model or "").strip()
    provider_name = model.split(":", 1)[0] if ":" in model else ""
    key_env = _API_KEY_ENV.get(provider_name)
    base_url_env = _API_BASE_URL_ENV.get(provider_name)
    key_present = bool(os.environ.get(key_env)) if key_env else None

    def _result(status: str, **kwargs: Any) -> LlmCheckResult:
        return LlmCheckResult(
            ok=status == STATUS_OK,
            status=status,
            planner="api",
            model=model or None,
            credential_env=key_env,
            credential_present=key_present,
            base_url=request.base_url,
            base_url_env=base_url_env,
            **kwargs,
        )

    if not model:
        return _result(
            STATUS_MISSING_CONFIG,
            detail=(
                "the 'api' planner requires a model id; pass --model with a "
                "provider prefix (e.g. 'anthropic:claude-opus-4-8', "
                "'openai:gpt-5.5', 'openai-chat:glm-5.2')."
            ),
        )
    if not provider_name:
        return _result(
            STATUS_MISSING_CONFIG,
            detail=(
                f"model {model!r} has no provider prefix; expected "
                f"'<provider>:<model>', e.g. 'anthropic:{model}'."
            ),
        )
    if key_env is not None and not key_present:
        return _result(
            STATUS_MISSING_API_KEY,
            detail=f"{key_env} is not set for provider {provider_name!r}.",
        )

    try:
        from pydantic_ai import Agent, ModelSettings
        from pydantic_ai.exceptions import ModelHTTPError, UserError

        from rpent.planner.base import build_api_model
    except ImportError as exc:
        return _result(STATUS_SDK_ERROR, detail=_describe(exc))

    try:
        api_model = build_api_model(model, request.base_url)
    except UserError as exc:
        return _result(_classify_user_error(exc), detail=_describe(exc))
    except ValueError as exc:
        # ``model`` is non-empty by the checks above, so build_api_model's own
        # ValueError cannot fire here: this comes from provider resolution,
        # where infer_provider raises ValueError for an unknown prefix.
        return _result(STATUS_UNSUPPORTED_PROVIDER, detail=_describe(exc))

    timeout_s = request.resolved_timeout_s()
    logger.info("probe budget: %ds", timeout_s)
    started = time.monotonic()
    try:
        reply = asyncio.run(
            asyncio.wait_for(
                _run_api_probe(Agent, ModelSettings, api_model),
                timeout=timeout_s,
            )
        )
    except (asyncio.TimeoutError, TimeoutError):
        return _result(
            STATUS_NETWORK_ERROR,
            detail=f"the request did not complete within {timeout_s}s.",
            latency_s=round(time.monotonic() - started, 3),
        )
    except ModelHTTPError as exc:
        return _result(
            _classify_http_status(exc.status_code),
            detail=_describe(exc),
            latency_s=round(time.monotonic() - started, 3),
        )
    except UserError as exc:
        return _result(_classify_user_error(exc), detail=_describe(exc))
    except Exception as exc:  # noqa: BLE001 - classified below
        return _result(
            _classify_transport_error(exc),
            detail=_describe(exc),
            latency_s=round(time.monotonic() - started, 3),
        )

    latency_s = round(time.monotonic() - started, 3)
    if not reply.strip():
        return _result(
            STATUS_PROVIDER_ERROR,
            detail="the provider returned an empty response.",
            latency_s=latency_s,
        )
    return _result(STATUS_OK, reply=reply.strip(), latency_s=latency_s)


async def _run_api_probe(agent_cls: Any, settings_cls: Any, api_model: Any) -> str:
    """Run the minimal tool-free pydantic-ai request and return its text."""
    agent = agent_cls(api_model)
    result = await agent.run(
        PROBE_PROMPT,
        model_settings=settings_cls(max_tokens=PROBE_MAX_TOKENS),
    )
    return str(result.output or "")


def _classify_http_status(status_code: int | None) -> str:
    """Map a provider HTTP status onto a check status.

    Shared by the ``api`` planner's typed ``ModelHTTPError`` and the Claude
    SDK's ``ResultMessage.api_error_status`` so both report one thing the
    same way.

    Args:
        status_code: The HTTP status the provider returned, when known.

    Returns:
        The matching status constant.
    """
    if status_code in (401, 403):
        return STATUS_AUTH_FAILED
    if status_code == 400:
        return STATUS_INVALID_MODEL
    return STATUS_PROVIDER_ERROR


def _classify_user_error(exc: Exception) -> str:
    """Map a pydantic-ai ``UserError`` onto a check status.

    pydantic-ai reports an unresolvable model id as ``Unknown model: <id>``
    and a missing credential as ``Set the `<PROVIDER>_API_KEY` environment
    variable ...``.

    Args:
        exc: The ``UserError`` raised while resolving the model.

    Returns:
        The matching status constant.
    """
    text = str(exc)
    if text.startswith("Unknown model"):
        return STATUS_UNSUPPORTED_PROVIDER
    if "_API_KEY" in text or "environment variable" in text:
        return STATUS_MISSING_API_KEY
    return STATUS_SDK_ERROR


def _classify_transport_error(exc: Exception) -> str:
    """Map a transport-layer exception onto a network or SDK status.

    Args:
        exc: The exception raised while performing the request.

    Returns:
        :data:`STATUS_NETWORK_ERROR` for transport failures, else
        :data:`STATUS_SDK_ERROR`.
    """
    try:
        import httpx
    except ImportError:
        return STATUS_SDK_ERROR
    if isinstance(exc, (httpx.TransportError, ConnectionError, TimeoutError)):
        return STATUS_NETWORK_ERROR
    return STATUS_SDK_ERROR


# ---------------------------------------------------------------------------
# claude_code planner
# ---------------------------------------------------------------------------


def _check_claude_code(request: LlmCheckRequest) -> LlmCheckResult:
    """Probe the Claude Agent SDK with one tool-free, single-turn query."""
    model = (request.model or "").strip() or "sonnet"
    key_present = bool(os.environ.get(_CLAUDE_CODE_KEY_ENV))

    def _result(status: str, **kwargs: Any) -> LlmCheckResult:
        return LlmCheckResult(
            ok=status == STATUS_OK,
            status=status,
            planner="claude_code",
            model=model,
            credential_env=_CLAUDE_CODE_KEY_ENV,
            credential_present=key_present,
            base_url=os.environ.get(_CLAUDE_CODE_BASE_URL_ENV) or None,
            base_url_env=_CLAUDE_CODE_BASE_URL_ENV,
            **kwargs,
        )

    try:
        import claude_agent_sdk
    except ImportError as exc:
        return _result(STATUS_SDK_ERROR, detail=_describe(exc))

    if not key_present and not _has_cli_login("claude_code"):
        # Nothing to authenticate with, so the probe can only time out. The
        # env var alone is not enough to conclude this: both backends accept
        # an interactive CLI login, which is why the login check is here too.
        return _result(
            STATUS_MISSING_API_KEY,
            detail=f"{_CLAUDE_CODE_KEY_ENV} is not set and no CLI login was found.",
        )
    timeout_s = _sdk_probe_budget(request, cli_login_only=not key_present)
    started = time.monotonic()
    try:
        outcome = asyncio.run(
            asyncio.wait_for(
                _run_claude_probe(claude_agent_sdk, model),
                timeout=timeout_s,
            )
        )
    except (asyncio.TimeoutError, TimeoutError):
        return _result(
            STATUS_NETWORK_ERROR,
            detail=(
                f"the Claude Agent SDK did not respond within {timeout_s}s. "
                "The provider, the network, or the local CLI the SDK "
                "runs could each cause this."
            ),
            latency_s=round(time.monotonic() - started, 3),
        )
    except Exception as exc:  # noqa: BLE001 - classified below
        return _result(
            _classify_sdk_error(exc, key_present=key_present, model=model),
            detail=_describe(exc),
            latency_s=round(time.monotonic() - started, 3),
        )

    latency_s = round(time.monotonic() - started, 3)
    if (status := _classify_claude_outcome(outcome)) is not None:
        return _result(
            status,
            detail=_describe_claude_failure(outcome),
            latency_s=latency_s,
        )
    return _result(STATUS_OK, reply=outcome.reply.strip(), latency_s=latency_s)


def _describe_claude_failure(outcome: _ClaudeProbeOutcome) -> str:
    """Render a Claude probe failure, preferring the SDK's own error fields."""
    parts: list[str] = []
    if outcome.sdk_error:
        parts.append(f"the Claude Agent SDK reported {outcome.sdk_error}")
    if outcome.api_error_status is not None:
        parts.append(f"HTTP {outcome.api_error_status}")
    if not parts:
        return "the Claude Agent SDK returned no assistant text."
    return redact_secrets("; ".join(parts) + ".")


@dataclass(frozen=True, slots=True)
class _ClaudeProbeOutcome:
    """What one Claude Agent SDK probe turn reported.

    Attributes:
        reply: Concatenated assistant text. Empty when the model said nothing.
        sdk_error: ``AssistantMessage.error``, e.g. ``authentication_failed``.
        api_error_status: ``ResultMessage.api_error_status``, e.g. 401.
    """

    reply: str = ""
    sdk_error: str | None = None
    api_error_status: int | None = None


async def _run_claude_probe(sdk: Any, model: str) -> _ClaudeProbeOutcome:
    """Consume one tool-free Claude Agent SDK turn and report what it said."""
    options = sdk.ClaudeAgentOptions(
        model=model,
        max_turns=1,
        tools=None,
        allowed_tools=[],
        mcp_servers={},
    )
    chunks: list[str] = []
    sdk_error: str | None = None
    api_error_status: int | None = None
    async for message in sdk.query(prompt=PROBE_PROMPT, options=options):
        chunks.extend(_assistant_text(message))
        if isinstance(message, sdk.AssistantMessage):
            sdk_error = sdk_error or getattr(message, "error", None)
        elif isinstance(message, sdk.ResultMessage):
            api_error_status = getattr(message, "api_error_status", None)
    return _ClaudeProbeOutcome(
        reply="".join(chunks),
        sdk_error=sdk_error,
        api_error_status=api_error_status,
    )


def _assistant_text(message: Any) -> list[str]:
    """Extract assistant text from one Claude Agent SDK message.

    Only ``AssistantMessage`` carries the model's reply. The stream also
    contains a ``UserMessage`` echoing the prompt we sent, so reading
    ``content`` off every message would mistake our own probe prompt for an
    answer and report a dead backend as healthy. ``ThinkingBlock`` keeps its
    text under ``thinking`` and is correctly skipped by the ``TextBlock``
    check.

    Args:
        message: One message from the SDK's stream.

    Returns:
        The message's assistant text blocks; empty for every other kind.
    """
    import claude_agent_sdk as sdk

    if not isinstance(message, sdk.AssistantMessage):
        return []
    content = getattr(message, "content", None)
    if not isinstance(content, list):
        return []
    return [
        block.text
        for block in content
        if isinstance(block, sdk.TextBlock) and block.text
    ]


def _classify_claude_outcome(outcome: _ClaudeProbeOutcome) -> str | None:
    """Map a probe outcome onto a failure status, or ``None`` when it passed.

    The SDK reports typed failures of its own, so they are classified from
    that structure rather than from message-text matching.

    Args:
        outcome: What the probe turn reported.

    Returns:
        A status constant, or ``None`` if the probe succeeded.
    """
    if outcome.sdk_error:
        return (
            STATUS_AUTH_FAILED
            if outcome.sdk_error == "authentication_failed"
            else STATUS_PROVIDER_ERROR
        )
    if outcome.api_error_status is not None:
        return _classify_http_status(outcome.api_error_status)
    if not outcome.reply.strip():
        return STATUS_PROVIDER_ERROR
    return None


# ---------------------------------------------------------------------------
# codex planner
# ---------------------------------------------------------------------------


def _check_codex(request: LlmCheckRequest) -> LlmCheckResult:
    """Probe the Codex SDK with one turn and no MCP server attached."""
    model = (request.model or "").strip() or os.environ.get("CODEX_MODEL") or None
    key_present = bool(os.environ.get(_CODEX_KEY_ENV))
    base_url = os.environ.get(_CODEX_BASE_URL_ENV) or None

    def _result(status: str, **kwargs: Any) -> LlmCheckResult:
        return LlmCheckResult(
            ok=status == STATUS_OK,
            status=status,
            planner="codex",
            model=model,
            credential_env=_CODEX_KEY_ENV,
            credential_present=key_present,
            base_url=base_url,
            base_url_env=_CODEX_BASE_URL_ENV,
            **kwargs,
        )

    try:
        from rpent.planner.codex import build_probe_config, run_probe_turn
    except ImportError as exc:
        return _result(STATUS_SDK_ERROR, detail=_describe(exc))

    if not key_present and not _has_cli_login("codex"):
        # Nothing to authenticate with, so the probe can only time out. The
        # env var alone is not enough to conclude this: both backends accept
        # an interactive CLI login, which is why the login check is here too.
        return _result(
            STATUS_MISSING_API_KEY,
            detail=f"{_CODEX_KEY_ENV} is not set and no CLI login was found.",
        )
    timeout_s = _sdk_probe_budget(request, cli_login_only=not key_present)
    started = time.monotonic()
    try:
        # Turn consumption and timeout/cleanup live in codex.py, next to the
        # planner's own SDK usage, so both share one set of API assumptions.
        reply = run_probe_turn(
            build_probe_config(base_url),
            prompt=PROBE_PROMPT,
            model=model,
            timeout_s=timeout_s,
        )
    except (asyncio.TimeoutError, TimeoutError):
        return _result(
            STATUS_NETWORK_ERROR,
            detail=(
                f"the Codex SDK did not respond within {timeout_s}s. "
                "The provider, the network, or the local CLI the SDK "
                "runs could each cause this."
            ),
            latency_s=round(time.monotonic() - started, 3),
        )
    except Exception as exc:  # noqa: BLE001 - classified below
        return _result(
            _classify_sdk_error(exc, key_present=key_present, model=model),
            detail=_describe(exc),
            latency_s=round(time.monotonic() - started, 3),
        )

    latency_s = round(time.monotonic() - started, 3)
    if not reply.strip():
        return _result(
            STATUS_PROVIDER_ERROR,
            detail="the Codex SDK returned no assistant text.",
            latency_s=latency_s,
        )
    return _result(STATUS_OK, reply=reply.strip(), latency_s=latency_s)


def _names_the_requested_model(text: str, model: str | None) -> bool:
    """Whether a 4xx message echoes back the model id that was requested.

    Gateways reword their model rejections freely: the same endpoint
    answered one probe with ``is not available for this group`` and the
    next, hours later, with ``is not supported by any configured account
    in this group``. Chasing that wording is a losing game. What every one
    of them does do is name the model it refused, so the model id plus a
    4xx status is the stable signal.

    Args:
        text: The lower-cased error message.
        model: The model id that was requested, when one was.

    Returns:
        True when a 4xx message contains the requested model id.
    """
    if not model or len(model) < 3:
        # Too short to be distinctive; a bare "o3" would match by accident.
        return False
    return model.lower() in text


def _classify_sdk_error(
    exc: Exception,
    *,
    key_present: bool,
    model: str | None = None,
) -> str:
    """Classify a child-process SDK failure from its message.

    The Claude and Codex SDKs surface provider problems as SDK exceptions
    rather than typed HTTP errors, so the message is the only signal.

    Args:
        exc: The exception raised by the SDK.
        key_present: Whether the backend's credential env var was set.
        model: The model id that was requested, used to recognise a model
            rejection without depending on the gateway's wording.

    Returns:
        The matching status constant.
    """
    text = f"{type(exc).__name__}: {exc}".lower()
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return STATUS_NETWORK_ERROR
    if any(
        token in text
        for token in ("401", "403", "unauthor", "authentication", "invalid api key")
    ):
        return STATUS_AUTH_FAILED
    if any(
        token in text
        for token in ("not logged in", "no credentials", "login", "api key")
    ):
        return STATUS_MISSING_API_KEY if not key_present else STATUS_AUTH_FAILED
    if any(
        token in text
        for token in ("connection", "dns", "resolve", "network", "unreachable", "tls")
    ):
        return STATUS_NETWORK_ERROR
    if any(token in text for token in ("429", "rate limit", "500", "502", "503")):
        return STATUS_PROVIDER_ERROR
    # Everything the message could otherwise be about has now been excluded,
    # so an error that names the very model that was requested is about that
    # model. Backends word this three different ways and one of them carries
    # no status code at all, which is why neither wording nor a 4xx can gate
    # it.
    if _names_the_requested_model(text, model):
        return STATUS_INVALID_MODEL
    if any(
        token in text
        for token in (
            # Fallback for backends that refuse without echoing the id. Every
            # phrase below was observed from a real backend, not guessed.
            "model does not exist",
            "unknown model",
            "invalid model",
            "model not found",
            "is not available",
            "unrecognized_model",
            "unrecognized model",
        )
    ) or re.search(r"\b400\b", text):
        # A bare 404 is deliberately NOT matched: a wrong base_url returns 404
        # too, and calling that an invalid model would send people to change
        # the one setting that was already correct.
        return STATUS_INVALID_MODEL
    return STATUS_SDK_ERROR
