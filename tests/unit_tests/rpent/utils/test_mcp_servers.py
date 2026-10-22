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

from pathlib import Path

import pytest

from rpent.utils.mcp_servers import (
    McpServerSpec,
    load_site_mcp_servers,
    merge_mcp_servers,
    parse_mcp_servers,
    request_headers,
)


def test_absent_section_yields_no_specs() -> None:
    assert parse_mcp_servers(None) == ()


def test_parses_url_headers_and_planner_tool_modes() -> None:
    specs = parse_mcp_servers(
        {
            "facility": {"url": "https://mcp.example/mcp"},
            "kb": {
                "url": "http://127.0.0.1:8765/mcp",
                "headers": {"Authorization": "Bearer token"},
                "planner_tools": ["search", "fetch"],
            },
            "all": {"url": "https://all.example/mcp", "planner_tools": "*"},
        }
    )

    assert specs == (
        McpServerSpec(name="facility", url="https://mcp.example/mcp"),
        McpServerSpec(
            name="kb",
            url="http://127.0.0.1:8765/mcp",
            headers={"Authorization": "Bearer token"},
            planner_tools=("search", "fetch"),
        ),
        McpServerSpec(
            name="all",
            url="https://all.example/mcp",
            planner_tools="*",
        ),
    )


def test_omitted_planner_tools_keeps_the_server_out_of_the_planner_layer() -> None:
    (spec,) = parse_mcp_servers({"facility": {"url": "https://mcp.example/mcp"}})

    assert spec.planner_tools is None


def test_parses_env_http_headers() -> None:
    (spec,) = parse_mcp_servers(
        {
            "facility": {
                "url": "https://mcp.example/mcp",
                "env_http_headers": {"Authorization": "FACILITY_TOKEN"},
            }
        }
    )

    assert spec.env_http_headers == {"Authorization": "FACILITY_TOKEN"}


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


def test_rejects_string_planner_tools_other_than_wildcard() -> None:
    with pytest.raises(ValueError, match=r"mcp_servers\.web\.planner_tools"):
        parse_mcp_servers(
            {"web": {"url": "https://example/mcp", "planner_tools": "search"}}
        )


def test_rejects_empty_planner_tools_list() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        parse_mcp_servers({"web": {"url": "https://example/mcp", "planner_tools": []}})


def test_rejects_header_in_both_header_maps() -> None:
    with pytest.raises(ValueError, match="both 'headers' and 'env_http_headers'"):
        parse_mcp_servers(
            {
                "web": {
                    "url": "https://example/mcp",
                    "headers": {"Authorization": "Bearer literal"},
                    "env_http_headers": {"Authorization": "WEB_TOKEN"},
                }
            }
        )


def test_request_headers_resolves_environment_references(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WEB_TOKEN", "Bearer resolved")
    spec = McpServerSpec(
        name="web",
        url="https://example/mcp",
        headers={"X-Site": "lab"},
        env_http_headers={"Authorization": "WEB_TOKEN"},
    )

    assert request_headers(spec) == {
        "X-Site": "lab",
        "Authorization": "Bearer resolved",
    }


def test_request_headers_rejects_unset_environment_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WEB_TOKEN", raising=False)
    spec = McpServerSpec(
        name="web",
        url="https://example/mcp",
        env_http_headers={"Authorization": "WEB_TOKEN"},
    )

    with pytest.raises(ValueError, match=r"mcp_servers\.web.*WEB_TOKEN"):
        request_headers(spec)


def test_merge_rejects_duplicate_site_and_robot_names() -> None:
    robot = (McpServerSpec(name="web", url="https://robot.example/mcp"),)
    site = (McpServerSpec(name="web", url="https://site.example/mcp"),)

    with pytest.raises(ValueError, match="duplicate MCP server name 'web'"):
        merge_mcp_servers(robot, site)


def test_merge_preserves_order() -> None:
    robot = (McpServerSpec(name="mine", url="https://robot.example/mcp"),)
    site = (
        McpServerSpec(name="lights", url="https://lights.example/mcp"),
        McpServerSpec(name="docs", url="https://docs.example/mcp"),
    )

    assert [spec.name for spec in merge_mcp_servers(robot, site)] == [
        "mine",
        "lights",
        "docs",
    ]


def test_load_site_config_reads_only_the_mcp_section(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FACILITY_TOKEN", "secret")
    path = tmp_path / "site.yaml"
    path.write_text(
        "site_name: lab-a\n"
        "mcp_servers:\n"
        "  lights:\n"
        "    url: https://lights.example/mcp\n"
        "    headers:\n"
        "      Authorization: Bearer ${oc.env:FACILITY_TOKEN}\n"
        "    planner_tools: [set_scene]\n"
        "  dock:\n"
        "    url: https://dock.example/mcp\n"
    )

    assert load_site_mcp_servers(path) == (
        McpServerSpec(
            name="lights",
            url="https://lights.example/mcp",
            headers={"Authorization": "Bearer secret"},
            planner_tools=("set_scene",),
        ),
        McpServerSpec(name="dock", url="https://dock.example/mcp"),
    )


def test_load_site_config_without_section_yields_no_specs(tmp_path: Path) -> None:
    path = tmp_path / "site.yaml"
    path.write_text("site_name: lab-a\n")

    assert load_site_mcp_servers(path) == ()


def test_load_site_config_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="site config not found"):
        load_site_mcp_servers(tmp_path / "missing.yaml")
