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

"""Validate Harness configuration locally before allocating robot resources."""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

SDK_VERSION = "0.2.1rc2"


def preflight_configuration(
    *,
    dsh_home: str | Path | None,
    model: str,
    reasoning_effort: str,
    provider: str = "openai-codex",
    require_images: bool = True,
    timeout_s: float = 15,
) -> dict[str, Any]:
    """Check local model capabilities and credential presence without inference.

    The runtime lists credential record metadata; neither token values nor
    grants cross this interface. Presence does not establish server acceptance,
    subscription entitlement, remaining quota, or whether refresh will succeed.
    Use ``rpent-check-llm`` for a real connectivity check.

    Args:
        dsh_home: Explicit runtime/credential home, falling back to DSH_HOME.
        model: Exact model identifier in the pinned runtime catalog.
        reasoning_effort: Requested reasoning level, checked against the model.
        provider: Supported runtime provider; currently openai-codex.
        require_images: Require image input support for robot observations.
        timeout_s: Bound on runtime startup and metadata inspection.

    Returns:
        Public model capabilities and credential presence, without secrets.

    Raises:
        ValueError: If configuration or model capabilities are unsupported.
        RuntimeError: If dependencies, credentials, or runtime inspection fail.
        TimeoutError: If the runtime does not report before the deadline.
    """
    if provider != "openai-codex":
        raise ValueError("deepseek_harness currently requires openai-codex")
    if not model or not model.strip() or not reasoning_effort:
        raise ValueError("deepseek_harness requires model and reasoning effort")
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("preflight timeout must be finite and positive")
    selected = dsh_home or os.environ.get("DSH_HOME")
    if not selected or not str(selected).strip():
        raise ValueError("deepseek_harness requires --dsh-home or DSH_HOME")
    home = Path(selected).expanduser().resolve()
    try:
        installed = {
            version(package)
            for package in ("deepseek-harness-sdk", "deepseek-harness-runtime-bin")
        }
    except PackageNotFoundError as exc:
        raise RuntimeError("Install rpent[deepseek-harness] before preflight") from exc
    if installed != {SDK_VERSION}:
        raise RuntimeError(f"Preflight requires SDK/runtime {SDK_VERSION}")

    from deepseek_harness import HarnessClient, HarnessConfig

    with tempfile.TemporaryDirectory(prefix="rpent-dsh-preflight-") as temporary:
        work = Path(temporary)
        result_path = work / "result.json"
        patch = work / "preflight.patch.json"
        patch.write_text(
            json.dumps(
                [
                    {"id": "agent-loop", "config": {"agents": []}},
                    {"id": "session-telemetry-otel", "disabled": True},
                    {"id": "llm-pi-ai", "config": {"providers": {provider: {}}}},
                    {
                        "insert": [
                            {
                                "id": "rpent-preflight",
                                "name": str(
                                    Path(__file__).with_name("deepseek_preflight.mjs")
                                ),
                                "config": {
                                    "provider": provider,
                                    "model": model,
                                    "reasoningEffort": reasoning_effort,
                                    "requireImages": require_images,
                                    "resultPath": str(result_path),
                                },
                            }
                        ]
                    },
                ]
            )
        )
        client = HarnessClient(
            HarnessConfig(
                profile="sdk",
                dsh_home=str(home),
                cwd=str(work),
                patches=(str(patch),),
                env={"DSH_TELEMETRY_DISABLED": "1", "DSH_PRIMARY_RUNTIME": ""},
                shutdown_timeout_seconds=1,
            )
        )
        deadline = time.monotonic() + timeout_s
        try:
            client.start()
            # No initialize, session/prompt, or prepareCall: this only starts
            # the metadata plugin, so it cannot infer, refresh, or execute tools.
            while not result_path.is_file():
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "DeepSeek Harness configuration preflight timed out"
                    )
                time.sleep(min(0.05, max(0, deadline - time.monotonic())))
            try:
                result = json.loads(result_path.read_text())
            except (OSError, ValueError) as exc:
                raise RuntimeError("Invalid Harness preflight response") from exc
        finally:
            client.close()

    if not isinstance(result, dict):
        raise RuntimeError("Invalid Harness preflight response")
    if result.get("status") == "missing_credential":
        raise RuntimeError(
            "No ChatGPT grant is configured; run rpent-dsh-login with the same --dsh-home"
        )
    if result.get("status") == "unsupported_capability":
        raise ValueError(
            "Selected model does not support the requested reasoning or image input"
        )
    if result.get("status") != "ready":
        raise RuntimeError(
            "Harness could not inspect the selected model or credential metadata"
        )
    if result.get("provider") != provider or result.get("model") != model:
        raise RuntimeError("Harness preflight returned a different model selection")
    return {
        **{
            key: result[key]
            for key in (
                "status",
                "provider",
                "model",
                "reasoning_effort",
                "reasoning_efforts",
                "input_modalities",
                "credential_configured",
                "credential_kind",
            )
            if key in result
        },
        "dsh_home": str(home),
    }
