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

MCP is only the transport. Where a server's tools appear depends on whether
the control loop must coordinate with the device:

* Tools the model can call at any time are attached to the planners. They are
  listed under ``planner_tools``; ``api``, ``claude_code`` and ``codex``
  consume that field.
* Tools that must be ordered against robot actions, a safety interlock, or the
  observation stream are registered on the robot's Toolkit with
  :meth:`~rpent.tools.toolkit.Toolkit.add_mcp_tools`. Planner adapters then
  expose them like any Toolkit tool: ``claude_code`` and ``codex`` add the
  ``mcp__rpent__`` prefix, while ``api`` uses the tool's own name. A robot must
  opt in; configuration alone never registers a tool on the Toolkit.

A server can supply both kinds: list the independent tools under
``planner_tools`` and register the rest in robot code.

Configuration comes from two places:

* A shared config (``--mcp-config``) is merged into every run, so several
  robots can use one endpoint without copying it.
* A robot's own config may declare ``mcp_servers`` for services that belong to
  that robot. Its ``parse_config`` runs :func:`parse_mcp_servers` and carries
  the result on :class:`~rpent.robots.robot_spec.RunConfig`; the CLI merges
  both sources at run time.

Only streamable-HTTP servers are supported: a spec carries a ``url`` and
optional headers, no stdio ``command``. :func:`parse_mcp_servers` resolves YAML
interpolation (``${oc.env:...}``) for either source, so this module is a pure
validator with no network access.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

#: Server name RPent uses for its own in-process toolkit server. An external
#: server must not reuse it: the ``mcp__<server>__<tool>`` namespace and the
#: per-backend server maps would otherwise collide.
RPENT_SERVER_NAME = "rpent"

#: The ``planner_tools`` value that admits every tool the server exposes.
ALL_TOOLS = "*"

#: Allowed server names: the token between ``mcp__`` and ``__<tool>``.
_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")

#: Tool selection for the planner. ``None`` keeps the server out of the
#: planner; the robot's Toolkit may still register its tools.
PlannerTools = tuple[str, ...] | Literal["*"] | None


@dataclass(frozen=True)
class McpServerSpec:
    """One external streamable-HTTP MCP server.

    Attributes:
        name: Server name, also its namespace in ``mcp__<name>__<tool>``.
        url: Streamable-HTTP endpoint (``http://`` or ``https://``). A URL with
            a user name or password is rejected, and task records drop its
            query and fragment.
        headers: Optional literal request headers. The ``claude_code`` and
            ``codex`` backends pass these through child-process arguments,
            where other local users can read them.
        env_http_headers: Header name to environment-variable name. Values are
            read when the server is used, keeping secrets out of generated
            config and out of the child-process arguments.
        planner_tools: ``"*"`` attaches every tool to the planners; a tuple
            attaches only those names; ``None`` does not attach the server to
            the planners at all. Only the planners consume this field.
            ``api`` and ``codex`` expose exactly the named tools;
            ``claude_code`` mounts the whole server and uses the tuple as its
            auto-approved set, so the server's other tools stay visible to the
            model.

    Construction validates the name and URL, so every consumer receives a spec
    whose ``mcp__<name>__<tool>`` namespace is usable and whose endpoint RPent
    can reach.
    """

    name: str
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    env_http_headers: dict[str, str] = field(default_factory=dict)
    planner_tools: PlannerTools = None

    def __post_init__(self) -> None:
        """Reject a spec no backend could use."""
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
                "mcp__<server>__<tool> namespace, and a trailing '_' makes "
                "that name ambiguous)"
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
            userinfo = urlsplit(self.url).username
        except ValueError as exc:
            raise ValueError(f"{where}.url is not a valid URL: {exc}") from exc
        if userinfo is not None:
            raise ValueError(
                f"{where}.url must not embed credentials; use headers or "
                "env_http_headers"
            )
        # HTTP header names are case-insensitive, so ``Authorization`` in one
        # map and ``authorization`` in the other describe the same header.
        literal = {name.casefold(): name for name in self.headers}
        overlap = sorted(
            literal[name.casefold()]
            for name in self.env_http_headers
            if name.casefold() in literal
        )
        if overlap:
            raise ValueError(
                f"{where}: header(s) {overlap} are set in both 'headers' and "
                "'env_http_headers'"
            )


def parse_mcp_servers(
    raw: Mapping[str, Any] | None,
) -> tuple[McpServerSpec, ...]:
    """Validate an ``mcp_servers`` section into specs.

    YAML interpolation (``${oc.env:...}``) is resolved here, so a shared config
    and a robot's own config behave the same: a header value written that way
    holds the environment variable's value when the section is parsed.

    Args:
        raw: The ``mcp_servers`` mapping from a shared or robot config, or
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

    from omegaconf import OmegaConf

    try:
        resolved = OmegaConf.to_container(OmegaConf.create(raw), resolve=True)
    except Exception as exc:
        raise ValueError(f"invalid mcp_servers section: {exc}") from exc

    specs: list[McpServerSpec] = []
    for name, value in resolved.items():
        specs.append(_parse_one(str(name), value))
    return tuple(specs)


def load_mcp_config(
    path: str | Path | None,
) -> tuple[McpServerSpec, ...]:
    """Load the ``mcp_servers`` section of a shared MCP config.

    The file is a YAML mapping; only its ``mcp_servers`` key is consumed, so
    it can carry unrelated settings. :func:`parse_mcp_servers` resolves
    ``${oc.env:...}`` in that section; other sections may keep unresolved
    references because RPent does not read them.

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
        ValueError: If two groups declare the same server name. The shared
            config and a robot's config must not silently shadow each other,
            and a duplicate would collide in the MCP tool namespace.
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


#: Placeholder for a credential in serialized run configuration.
REDACTED_VALUE = "<redacted>"


def public_spec(spec: McpServerSpec) -> dict[str, Any]:
    """Return a serializable view of a spec that omits credentials.

    Parsing resolves ``${oc.env:...}`` in ``headers``, so the values may be
    credentials, and a URL query is another place servers accept a token. Task
    records keep the run configuration, and that payload is written to
    ``task.json`` and the Dashboard task record, so only header names, the
    endpoint without its query and fragment, and the environment-variable
    names are kept.
    """
    return {
        "name": spec.name,
        "url": _recorded_url(spec.url),
        "headers": dict.fromkeys(spec.headers, REDACTED_VALUE),
        "env_http_headers": dict(spec.env_http_headers),
        "planner_tools": spec.planner_tools,
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
    """Validate one ``mcp_servers.<name>`` entry.

    The spec validates its own name, URL, and header overlap.
    """
    where = f"mcp_servers.{name}"
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

    return McpServerSpec(
        name=name,
        url=value.get("url"),
        headers=_parse_headers(value.get("headers"), f"{where}.headers"),
        env_http_headers=_parse_env_http_headers(
            value.get("env_http_headers"), f"{where}.env_http_headers"
        ),
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
            "server out of the planner"
        )
    return tuple(dict.fromkeys(str(item) for item in raw))
