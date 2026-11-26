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

"""External streamable-HTTP MCP server configuration.

External MCP servers are registered on the robot's :class:`~rpent.tools.toolkit.Toolkit`
so every online planner (``api``, ``claude_code``, ``codex``, and ``gemini_cli``)
exposes the same tools, timeouts, capture, and filtering rules.

Configuration comes from two places:

* A shared config (``--mcp-config``) is merged into every run, so several
  robots can use one endpoint without copying it.
* A robot's own config may declare ``mcp_servers`` for services that belong to
  that robot. Its ``parse_config`` runs :func:`parse_mcp_servers` and carries
  the result on :class:`~rpent.robots.robot_spec.RunConfig`; the CLI merges
  both sources at run time.

Only streamable-HTTP servers are supported: a spec carries a ``url``, optional
``headers``, optional ``tools`` selection (defaulting to ``"*"``), optional
``sync_with_robot`` (defaulting to ``False``), and optional ``timeout_s``
(defaulting to ``None``, which falls back to ``MCP_CALL_TIMEOUT_S`` or 60 s).
:func:`parse_mcp_servers` resolves YAML interpolation (``${oc.env:...}``) for
either source, so this module is a pure validator with no network access.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

#: Server name RPent uses for its own in-process toolkit server.
RPENT_SERVER_NAME = "rpent"

#: The ``tools`` value that exposes every tool the server advertises.
ALL_TOOLS = "*"

#: Allowed server names: the prefix in ``<server>__<tool>``.
_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")

#: Tool selection for one MCP server. ``"*"`` exposes every tool on the server;
#: a tuple exposes only those tool names.
ToolSelection = tuple[str, ...] | Literal["*"]

#: Placeholder for a credential in serialized run configuration.
REDACTED_VALUE = "<redacted>"


@dataclass(frozen=True)
class McpServerSpec:
    """One external streamable-HTTP MCP server.

    Attributes:
        name: Server name, used as the ``<name>__<tool>`` prefix on ``Toolkit``.
        url: Streamable-HTTP endpoint (``http://`` or ``https://``). A URL with
            a user name or password is rejected, and task records redact its
            query and fragment.
        headers: Optional HTTP request headers. Values stay in-process (never
            passed on child-process command lines) and are redacted in task
            records.
        tools: ``"*"`` (default) exposes every tool the server advertises; a
            tuple exposes only those named tools.
        sync_with_robot: When ``False`` (default), tools run concurrently with
            other read-only tools and return the remote payload directly without
            triggering ``get_env_state``. When ``True``, tools run exclusively
            alongside robot actions, trigger post-action observation capture,
            and preserve their remote payload under ``tool_result``.
        timeout_s: Optional per-request timeout in seconds for MCP discovery and
            tool calls. When ``None`` (default), RPent falls back to the
            ``MCP_CALL_TIMEOUT_S`` environment variable or 60 seconds.
    """

    name: str
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    tools: ToolSelection = ALL_TOOLS
    sync_with_robot: bool = False
    timeout_s: float | None = None

    def __post_init__(self) -> None:
        """Reject a spec no consumer could use safely."""
        where = f"mcp_servers.{self.name}"
        if (
            not isinstance(self.name, str)
            or _NAME_PATTERN.fullmatch(self.name) is None
            or "__" in self.name
            or self.name.endswith("_")
        ):
            raise ValueError(
                f"{where}: server name must match {_NAME_PATTERN.pattern} "
                "without '__' or a trailing '_' (it becomes the "
                "<server>__<tool> tool prefix)"
            )
        if self.name == RPENT_SERVER_NAME:
            raise ValueError(
                f"{where}: server name {RPENT_SERVER_NAME!r} is reserved for "
                "RPent's own toolkit server"
            )
        if not isinstance(self.url, str) or not self.url.startswith(
            ("http://", "https://")
        ):
            raise ValueError(
                f"{where}.url must be an http(s) URL; only streamable-HTTP MCP "
                f"servers are supported (got {self.url!r})"
            )
        try:
            parts = urlsplit(self.url)
        except ValueError as exc:
            raise ValueError(f"{where}.url is not a valid URL: {exc}") from exc
        if not parts.hostname:
            raise ValueError(f"{where}.url must include a hostname (got {self.url!r})")
        if parts.username is not None or parts.password is not None:
            raise ValueError(
                f"{where}.url must not embed credentials; pass them in 'headers'"
            )
        if not isinstance(self.sync_with_robot, bool):
            raise ValueError(
                f"{where}.sync_with_robot must be a boolean; got "
                f"{type(self.sync_with_robot).__name__}"
            )
        object.__setattr__(self, "tools", _parse_tools(self.tools, where))
        object.__setattr__(
            self, "timeout_s", _parse_timeout_s(self.timeout_s, f"{where}.timeout_s")
        )


def parse_mcp_servers(
    raw: Mapping[str, Any] | None,
) -> tuple[McpServerSpec, ...]:
    """Validate an ``mcp_servers`` section into specs.

    YAML interpolation (``${oc.env:...}``) is resolved here, so a shared config
    and a robot's own config behave the same.

    Args:
        raw: The ``mcp_servers`` mapping from a shared or robot config, or
            ``None`` when the config declares none.

    Returns:
        The validated specs, in config order. Empty when ``raw`` is ``None``.

    Raises:
        ValueError: On a malformed section, naming the offending server and
            field. Validation is shape-only and performs no network requests.
    """
    if raw is None:
        return ()
    if not isinstance(raw, Mapping):
        raise ValueError("mcp_servers must be a mapping of server name to config")

    from omegaconf import OmegaConf

    try:
        resolved = OmegaConf.to_container(OmegaConf.create(raw), resolve=True)
    except Exception as exc:
        raise ValueError(f"invalid mcp_servers section: {exc}") from exc

    return tuple(_parse_one(str(name), value) for name, value in resolved.items())


def load_mcp_config(
    path: str | Path | None,
) -> tuple[McpServerSpec, ...]:
    """Load the ``mcp_servers`` section of a shared MCP config YAML file.

    Only the ``mcp_servers`` key is consumed, so the file may carry unrelated
    settings. :func:`parse_mcp_servers` resolves ``${oc.env:...}`` inside that
    section only.

    Args:
        path: Path to the shared config, or ``None`` when ``--mcp-config`` was
            not passed.

    Returns:
        The validated specs. Empty when no path is given or the file has no
        ``mcp_servers`` section.

    Raises:
        ValueError: If the file is missing, unreadable, not a mapping, or the
            section is malformed.
    """
    if path is None:
        return ()
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise ValueError(f"MCP config not found: {config_path}")

    from omegaconf import DictConfig, OmegaConf

    try:
        config = OmegaConf.load(config_path)
    except Exception as exc:
        raise ValueError(f"invalid MCP config {config_path}: {exc}") from exc
    if not isinstance(config, DictConfig):
        raise ValueError(f"MCP config must be a mapping: {config_path}")
    try:
        return parse_mcp_servers(config.get("mcp_servers"))
    except ValueError as exc:
        raise ValueError(f"invalid MCP config {config_path}: {exc}") from exc


def merge_mcp_servers(
    *groups: tuple[McpServerSpec, ...],
) -> tuple[McpServerSpec, ...]:
    """Concatenate spec groups, rejecting duplicate server names.

    Raises:
        ValueError: If two groups declare the same server name.
    """
    merged: list[McpServerSpec] = []
    seen: set[str] = set()
    for group in groups:
        for spec in group:
            if spec.name in seen:
                raise ValueError(
                    f"duplicate MCP server name {spec.name!r}: it is declared "
                    "both in the shared MCP config and by the robot, or twice "
                    "in one config"
                )
            seen.add(spec.name)
            merged.append(spec)
    return tuple(merged)


def public_spec(spec: McpServerSpec) -> dict[str, Any]:
    """Return a serializable view of a spec with credentials redacted.

    Task records persist run configuration in ``task.json`` and Dashboard
    records, so header values and URL queries/fragments are replaced with
    :data:`REDACTED_VALUE`.
    """
    return {
        "name": spec.name,
        "url": _recorded_url(spec.url),
        "headers": dict.fromkeys(spec.headers, REDACTED_VALUE),
        "tools": spec.tools,
        "sync_with_robot": spec.sync_with_robot,
        "timeout_s": spec.timeout_s,
    }


def _recorded_url(url: str) -> str:
    """Return a URL with its query and fragment replaced by a placeholder."""
    parts = urlsplit(url)
    recorded = urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
    if parts.query:
        recorded += "?" + REDACTED_VALUE
    if parts.fragment:
        recorded += "#" + REDACTED_VALUE
    return recorded


def _parse_one(name: str, value: Any) -> McpServerSpec:
    """Validate one ``mcp_servers.<name>`` entry."""
    where = f"mcp_servers.{name}"
    if not isinstance(value, Mapping):
        raise ValueError(f"{where} must be a mapping with a 'url'")

    valid_keys = {"url", "headers", "tools", "sync_with_robot", "timeout_s"}
    unknown = sorted(set(value) - valid_keys)
    if unknown:
        raise ValueError(
            f"{where}: unknown key(s) {unknown}. Valid keys: {sorted(valid_keys)}."
        )

    raw_sync = value.get("sync_with_robot", False)
    if not isinstance(raw_sync, bool):
        raise ValueError(f"{where}.sync_with_robot must be a boolean; got {raw_sync!r}")

    return McpServerSpec(
        name=name,
        url=value.get("url"),
        headers=_parse_headers(value.get("headers"), f"{where}.headers"),
        tools=_parse_tools(value.get("tools"), where),
        sync_with_robot=raw_sync,
        timeout_s=_parse_timeout_s(value.get("timeout_s"), f"{where}.timeout_s"),
    )


def _parse_timeout_s(raw: Any, where: str) -> float | None:
    """Validate an optional positive per-request timeout in seconds."""
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"{where} must be a positive number of seconds; got {raw!r}")
    timeout_s = float(raw)
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError(f"{where} must be a positive number of seconds; got {raw!r}")
    return timeout_s


def _parse_headers(raw: Any, where: str) -> dict[str, str]:
    """Validate a ``headers`` mapping."""
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"{where} must be a mapping of string to string")
    invalid = sorted(key for key, val in raw.items() if not isinstance(val, str))
    if invalid:
        raise ValueError(f"{where} values must be strings; got {invalid}")
    return {str(key): str(val) for key, val in raw.items()}


def _parse_tools(raw: Any, where: str) -> ToolSelection:
    """Validate ``tools`` (``"*"``, a non-empty list of names, or omitted)."""
    if raw is None or raw == ALL_TOOLS:
        return ALL_TOOLS
    if isinstance(raw, str):
        raise ValueError(
            f"{where}.tools must be {ALL_TOOLS!r} or a list of tool names; got {raw!r}"
        )
    if not isinstance(raw, Sequence):
        raise ValueError(f"{where}.tools must be {ALL_TOOLS!r} or a list of tool names")
    invalid = [item for item in raw if not isinstance(item, str) or not item]
    if invalid:
        raise ValueError(f"{where}.tools entries must be non-empty strings")
    if ALL_TOOLS in raw:
        raise ValueError(
            f"{where}.tools: write {ALL_TOOLS!r} rather than a list "
            "containing it; a list names individual tools"
        )
    if not raw:
        raise ValueError(
            f"{where}.tools must not be empty; omit it or use {ALL_TOOLS!r} "
            "to expose every tool on the server"
        )
    return tuple(dict.fromkeys(str(item) for item in raw))
