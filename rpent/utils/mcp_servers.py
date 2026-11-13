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

"""External MCP servers RPent may consume.

MCP is a transport, not a layer. Which RPent layer consumes a server's tools
is decided by whether the control loop must coordinate with the device:

* Tools that are independent of the robot's execution loop are exposed to the
  planners directly: they are listed under ``planner_tools`` and attached by
  ``api``, ``claude_code`` and ``codex``.
* Tools that need coordination with the execution loop, a safety interlock,
  or the observation stream are first-class robot/Env capabilities. They are
  exposed through the robot's Toolkit with
  :meth:`~rpent.tools.toolkit.Toolkit.add_mcp_tools`; planner adapters then
  expose them like any Toolkit tool (``claude_code`` and ``codex`` add the
  ``mcp__rpent__`` prefix, ``api`` uses the tool's own name). A robot must opt
  in; a server is never promoted to the toolkit layer by configuration alone.

A single server can mix both kinds of tools: list the independent ones under
``planner_tools`` and adopt the rest in robot code.

Configuration has two axes:

* Site config (``--site-config``) is deployment-level and shared by every
  robot at that site. Facility devices (lighting, charging docks, power
  outlets) belong here so robots do not each copy their endpoint.
* A robot's own config may declare ``mcp_servers`` for services that belong to
  that robot. Its ``parse_config`` runs :func:`parse_mcp_servers` and carries
  the result on :class:`~rpent.robots.robot_spec.RunConfig`; the CLI merges
  those specs with the site config at run time.

Only streamable-HTTP servers are supported: a spec carries a ``url`` and
optional headers, no stdio ``command``. YAML interpolation (``${oc.env:...}``)
is resolved by the loader, so this module is a pure validator with no network
access.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

#: Server name RPent reserves for its own in-process toolkit server. An
#: external server must not reuse it, or the two would collide in the
#: ``mcp__<server>__<tool>`` namespace.
RESERVED_SERVER_NAME = "rpent"

#: The ``planner_tools`` value that admits every tool the server exposes.
ALL_TOOLS = "*"

#: Allowed server names: the token between ``mcp__`` and ``__<tool>``.
_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")

#: Tool selection for the planner layer. ``None`` keeps the server out of the
#: planner layer (the robot's Env/Toolkit layer may still adopt its tools).
PlannerTools = tuple[str, ...] | Literal["*"] | None


@dataclass(frozen=True)
class McpServerSpec:
    """One external streamable-HTTP MCP server.

    Attributes:
        name: Server name, also its namespace in ``mcp__<name>__<tool>``.
        url: Streamable-HTTP endpoint (``http://`` or ``https://``).
        headers: Optional literal request headers. The ``claude_code`` and
            ``codex`` backends pass these through child-process arguments,
            where other local users can read them; keep secrets in
            ``env_http_headers`` instead.
        env_http_headers: Header name to environment-variable name. Values are
            read when the server is used, keeping secrets out of generated
            config and out of the child process arguments.
        planner_tools: ``"*"`` attaches every tool to the planners; a tuple
            attaches only those names; ``None`` does not attach the server to
            the planners at all. Only the planner layer consumes this field.
            ``api`` and ``codex`` expose exactly the named tools;
            ``claude_code`` mounts the whole server and uses the tuple as its
            auto-approved set, so the server's other tools stay visible to the
            model.
    """

    name: str
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    env_http_headers: dict[str, str] = field(default_factory=dict)
    planner_tools: PlannerTools = None


def parse_mcp_servers(
    raw: Mapping[str, Any] | None,
) -> tuple[McpServerSpec, ...]:
    """Validate an ``mcp_servers`` section into specs.

    Args:
        raw: The ``mcp_servers`` mapping from a site or robot config, or
            ``None`` when the config declares none.

    Returns:
        The validated specs, in config order. Empty when ``raw`` is ``None``.

    Raises:
        ValueError: On a malformed section, naming the offending server and
            field. Validation is shape-only: a non-HTTP URL, a reserved or
            invalid name, non-string headers, or an unreadable tool list.
    """
    if raw is None:
        return ()
    if not isinstance(raw, Mapping):
        raise ValueError("mcp_servers must be a mapping of server name to config")

    specs: list[McpServerSpec] = []
    for name, value in raw.items():
        specs.append(_parse_one(str(name), value))
    return tuple(specs)


def load_site_mcp_servers(
    path: str | Path | None,
) -> tuple[McpServerSpec, ...]:
    """Load the ``mcp_servers`` section of a deployment-level site config.

    The file is a YAML mapping; only its ``mcp_servers`` key is consumed, so
    it can carry other site settings. OmegaConf resolves ``${oc.env:...}``
    interpolation in that section before validation; other sections may keep
    unresolved references because RPent does not read them.

    Args:
        path: Path to the site config, or ``None`` when ``--site-config`` was
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
        raise ValueError(f"site config not found: {config_path}")

    from omegaconf import DictConfig, OmegaConf

    try:
        config = OmegaConf.load(config_path)
    except Exception as exc:
        raise ValueError(f"invalid site config {config_path}: {exc}") from exc
    if not isinstance(config, DictConfig):
        raise ValueError(f"site config must be a mapping: {config_path}")
    section = config.get("mcp_servers")
    try:
        raw = None if section is None else OmegaConf.to_container(section, resolve=True)
    except Exception as exc:
        raise ValueError(f"invalid site config {config_path}: {exc}") from exc
    return parse_mcp_servers(raw)


def merge_mcp_servers(
    *groups: tuple[McpServerSpec, ...],
) -> tuple[McpServerSpec, ...]:
    """Concatenate spec groups, rejecting duplicate server names.

    Raises:
        ValueError: If two groups declare the same server name. Site-level and
            robot-level config must not silently shadow each other, and a
            duplicate would collide in the MCP tool namespace.
    """
    merged: list[McpServerSpec] = []
    seen: set[str] = set()
    for group in groups:
        for spec in group:
            if spec.name in seen:
                raise ValueError(
                    f"duplicate MCP server name {spec.name!r}: it is declared "
                    "both at site level and by the robot, or twice in one config"
                )
            seen.add(spec.name)
            merged.append(spec)
    return tuple(merged)


def request_headers(spec: McpServerSpec) -> dict[str, str]:
    """Return the request headers for one server, resolving env references.

    Raises:
        ValueError: If a referenced environment variable is unset.
    """
    headers = dict(spec.headers)
    for name, env_name in spec.env_http_headers.items():
        value = os.environ.get(env_name)
        if value is None:
            raise ValueError(
                f"mcp_servers.{spec.name}.env_http_headers.{name}: environment "
                f"variable {env_name!r} is not set"
            )
        headers[name] = value
    return headers


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

    unknown = sorted(
        set(value) - {"url", "headers", "env_http_headers", "planner_tools"}
    )
    if unknown:
        raise ValueError(
            f"{where}: unknown key(s) {unknown}. Valid keys: "
            "['env_http_headers', 'headers', 'planner_tools', 'url']."
        )

    url = value.get("url")
    if not isinstance(url, str) or not url:
        raise ValueError(f"{where}.url must be a non-empty string")
    if not url.startswith(("http://", "https://")):
        raise ValueError(
            f"{where}.url must be an http(s) URL; only streamable-HTTP MCP "
            f"servers are supported (got {url!r})"
        )

    headers = _parse_headers(value.get("headers"), f"{where}.headers")
    env_http_headers = _parse_env_http_headers(
        value.get("env_http_headers"), f"{where}.env_http_headers"
    )
    overlap = sorted(set(headers) & set(env_http_headers))
    if overlap:
        raise ValueError(
            f"{where}: header(s) {overlap} are set in both 'headers' and "
            "'env_http_headers'"
        )

    return McpServerSpec(
        name=name,
        url=url,
        headers=headers,
        env_http_headers=env_http_headers,
        planner_tools=_parse_planner_tools(value.get("planner_tools"), where),
    )


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


def _parse_env_http_headers(raw: Any, where: str) -> dict[str, str]:
    """Validate an ``env_http_headers`` mapping (header to env var name)."""
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"{where} must be a mapping of header to env var name")
    invalid = sorted(
        key for key, val in raw.items() if not isinstance(val, str) or not val.strip()
    )
    if invalid:
        raise ValueError(
            f"{where} values must be non-empty environment variable names; "
            f"got {invalid}"
        )
    return {str(key): str(val) for key, val in raw.items()}


def _parse_planner_tools(raw: Any, where: str) -> PlannerTools:
    """Validate ``planner_tools`` (``"*"``, a list of names, or omitted)."""
    if raw is None:
        return None
    if raw == ALL_TOOLS:
        return ALL_TOOLS
    if isinstance(raw, str):
        raise ValueError(
            f"{where}.planner_tools must be {ALL_TOOLS!r} or a list of tool "
            f"names; got {raw!r}"
        )
    if not isinstance(raw, Sequence):
        raise ValueError(
            f"{where}.planner_tools must be {ALL_TOOLS!r} or a list of tool names"
        )
    invalid = [item for item in raw if not isinstance(item, str) or not item]
    if invalid:
        raise ValueError(f"{where}.planner_tools entries must be non-empty strings")
    if ALL_TOOLS in raw:
        raise ValueError(
            f"{where}.planner_tools: write {ALL_TOOLS!r} rather than a list "
            "containing it; a list names individual tools"
        )
    if not raw:
        raise ValueError(
            f"{where}.planner_tools must not be empty; omit it to keep the "
            "server out of the planner layer"
        )
    return tuple(dict.fromkeys(str(item) for item in raw))
