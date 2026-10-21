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

"""External MCP servers a robot declares for its planners to attach.

A robot opts in by adding an ``mcp_servers`` mapping to its own config YAML;
the robot's ``parse_config`` runs :func:`parse_mcp_servers` on that section and
carries the result on :class:`~rpent.robots.robot_spec.RunConfig`. Planners turn
each :class:`McpServerSpec` into their native MCP-server entry — this module
never talks to the server itself.

Only streamable-HTTP servers are supported: a spec carries a ``url`` (and
optional ``headers``), no stdio ``command``. The mapping has already been
resolved by the robot's own YAML loader (``${oc.env:...}`` interpolation
included), so this module is a pure validator with no YAML or network access.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

#: Server name RPent reserves for its own in-process toolkit server. An
#: external server must not reuse it, or the two would collide in the
#: ``mcp__<server>__<tool>`` namespace.
RESERVED_SERVER_NAME = "rpent"

#: The ``tools`` value that admits every tool the server exposes.
ALL_TOOLS = "*"

#: Allowed server names: the token between ``mcp__`` and ``__<tool>``.
_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


@dataclass(frozen=True)
class McpServerSpec:
    """One external streamable-HTTP MCP server.

    Attributes:
        name: Server name, also its namespace in ``mcp__<name>__<tool>``.
        url: Streamable-HTTP endpoint (``http://`` or ``https://``).
        headers: Optional request headers (e.g. an ``Authorization`` token).
        tools: ``None`` admits every tool the server exposes; otherwise the
            explicit tool names to admit. Only the Claude Code planner needs
            this — its ``allowed_tools`` is an allowlist, so it cannot admit a
            server's tools without their names. Codex and the API planner
            expose all of an attached server's tools regardless.
    """

    name: str
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    tools: tuple[str, ...] | None = None


def parse_mcp_servers(
    raw: Mapping[str, Any] | None,
) -> tuple[McpServerSpec, ...]:
    """Validate a robot config's ``mcp_servers`` section into specs.

    Args:
        raw: The ``mcp_servers`` mapping from the robot config, or ``None``
            when the robot declares none.

    Returns:
        The validated specs, in config order. Empty when ``raw`` is ``None``.

    Raises:
        ValueError: On a malformed section, naming the offending server and
            field. Validation is shape-only: a non-HTTP URL, a reserved or
            invalid name, non-string headers, or an unreadable ``tools`` value.
    """
    if raw is None:
        return ()
    if not isinstance(raw, Mapping):
        raise ValueError("mcp_servers must be a mapping of server name to config")

    specs: list[McpServerSpec] = []
    for name, value in raw.items():
        specs.append(_parse_one(str(name), value))
    return tuple(specs)


def _parse_one(name: str, value: Any) -> McpServerSpec:
    """Validate one ``mcp_servers.<name>`` entry."""
    where = f"mcp_servers.{name}"
    if not _NAME_PATTERN.match(name) or "__" in name:
        raise ValueError(
            f"{where}: server name must match {_NAME_PATTERN.pattern} without "
            "'__' (it becomes the mcp__<server>__<tool> namespace)"
        )
    if name == RESERVED_SERVER_NAME:
        raise ValueError(
            f"{where}: server name {RESERVED_SERVER_NAME!r} is reserved for "
            "RPent's own toolkit server"
        )
    if not isinstance(value, Mapping):
        raise ValueError(f"{where} must be a mapping with a 'url'")

    unknown = sorted(set(value) - {"url", "headers", "tools"})
    if unknown:
        raise ValueError(
            f"{where}: unknown key(s) {unknown}. Valid keys: ['headers', 'tools', 'url']."
        )

    url = value.get("url")
    if not isinstance(url, str) or not url:
        raise ValueError(f"{where}.url must be a non-empty string")
    if not url.startswith(("http://", "https://")):
        raise ValueError(
            f"{where}.url must be an http(s) URL; only streamable-HTTP MCP "
            f"servers are supported (got {url!r})"
        )

    return McpServerSpec(
        name=name,
        url=url,
        headers=_parse_headers(value.get("headers"), where),
        tools=_parse_tools(value.get("tools"), where),
    )


def _parse_headers(raw: Any, where: str) -> dict[str, str]:
    """Validate the optional ``headers`` mapping."""
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"{where}.headers must be a mapping of string to string")
    invalid = sorted(key for key, val in raw.items() if not isinstance(val, str))
    if invalid:
        raise ValueError(f"{where}.headers values must be strings; got {invalid}")
    return {str(key): str(val) for key, val in raw.items()}


def _parse_tools(raw: Any, where: str) -> tuple[str, ...] | None:
    """Validate the optional ``tools`` value (``"*"`` or a list of names)."""
    if raw is None or raw == ALL_TOOLS:
        return None
    if isinstance(raw, str):
        raise ValueError(
            f"{where}.tools must be {ALL_TOOLS!r} or a list of tool names; got {raw!r}"
        )
    if not isinstance(raw, Sequence):
        raise ValueError(f"{where}.tools must be {ALL_TOOLS!r} or a list of tool names")
    invalid = [item for item in raw if not isinstance(item, str) or not item]
    if invalid:
        raise ValueError(f"{where}.tools entries must be non-empty strings")
    return tuple(str(item) for item in raw)
