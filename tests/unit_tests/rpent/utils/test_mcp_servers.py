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

from __future__ import annotations

import pytest

from rpent.utils.mcp_servers import McpServerSpec, parse_mcp_servers


def test_absent_section_yields_no_specs() -> None:
    assert parse_mcp_servers(None) == ()


def test_parses_url_headers_and_tool_modes() -> None:
    specs = parse_mcp_servers(
        {
            "web": {"url": "https://mcp.example/mcp"},
            "kb": {
                "url": "http://127.0.0.1:8765/mcp",
                "headers": {"Authorization": "Bearer token"},
                "tools": ["search", "fetch"],
            },
            "all": {"url": "https://all.example/mcp", "tools": "*"},
        }
    )

    assert specs == (
        McpServerSpec(name="web", url="https://mcp.example/mcp"),
        McpServerSpec(
            name="kb",
            url="http://127.0.0.1:8765/mcp",
            headers={"Authorization": "Bearer token"},
            tools=("search", "fetch"),
        ),
        McpServerSpec(name="all", url="https://all.example/mcp"),
    )


def test_rejects_non_mapping_section() -> None:
    with pytest.raises(ValueError, match="must be a mapping"):
        parse_mcp_servers(["web"])


def test_rejects_non_http_url() -> None:
    with pytest.raises(ValueError, match=r"mcp_servers\.web\.url must be an http"):
        parse_mcp_servers({"web": {"url": "ftp://example/mcp"}})


def test_rejects_reserved_server_name() -> None:
    with pytest.raises(ValueError, match="reserved"):
        parse_mcp_servers({"rpent": {"url": "https://example/mcp"}})


def test_rejects_invalid_server_name() -> None:
    with pytest.raises(ValueError, match=r"mcp_servers\.bad name"):
        parse_mcp_servers({"bad name": {"url": "https://example/mcp"}})


def test_rejects_double_underscore_in_name() -> None:
    with pytest.raises(ValueError, match="without"):
        parse_mcp_servers({"a__b": {"url": "https://example/mcp"}})


def test_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="unknown key"):
        parse_mcp_servers({"web": {"url": "https://example/mcp", "command": "x"}})


def test_rejects_non_string_headers() -> None:
    with pytest.raises(ValueError, match=r"mcp_servers\.web\.headers"):
        parse_mcp_servers({"web": {"url": "https://example/mcp", "headers": {"A": 1}}})


def test_rejects_string_tools_other_than_wildcard() -> None:
    with pytest.raises(ValueError, match=r"mcp_servers\.web\.tools"):
        parse_mcp_servers({"web": {"url": "https://example/mcp", "tools": "search"}})
