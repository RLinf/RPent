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
    load_mcp_config,
    merge_mcp_servers,
    parse_mcp_servers,
    public_spec,
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


def test_parse_resolves_a_robot_config_like_the_shared_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FACILITY_TOKEN", "secret")

    (spec,) = parse_mcp_servers(
        {
            "facility": {
                "url": "https://facility.example/mcp",
                "headers": {"Authorization": "Bearer ${oc.env:FACILITY_TOKEN}"},
            }
        }
    )

    assert spec.headers == {"Authorization": "Bearer secret"}


def test_public_spec_keeps_no_credential_from_the_url_or_headers() -> None:
    spec = McpServerSpec(
        name="web",
        url="https://web.example/mcp?token=secret#secret-fragment",
        headers={"Authorization": "Bearer secret"},
    )

    public = public_spec(spec)

    assert public["url"] == "https://web.example/mcp?<redacted>#<redacted>"
    assert public["headers"] == {"Authorization": "<redacted>"}
    assert "secret" not in repr(public)


def test_construction_rejects_a_spec_no_backend_could_use() -> None:
    for values in (
        {"name": "rpent", "url": "https://web.example/mcp"},
        {"name": "a__b", "url": "https://web.example/mcp"},
        {"name": "docs\n", "url": "https://web.example/mcp"},
        {"name": "bad name", "url": "https://web.example/mcp"},
        {"name": "power_", "url": "https://web.example/mcp"},
        {"name": "web", "url": "ftp://web.example/mcp"},
        {"name": "web", "url": "https://user:pass@web.example/mcp"},
    ):
        with pytest.raises(ValueError, match=r"mcp_servers\."):
            McpServerSpec(**values)


def test_rejects_non_mapping_section() -> None:
    with pytest.raises(ValueError, match="must be a mapping"):
        parse_mcp_servers(["web"])


def test_parse_surfaces_construction_errors_with_the_section_path() -> None:
    with pytest.raises(ValueError, match=r"mcp_servers\.web\.url must be an http"):
        parse_mcp_servers({"web": {"url": "ftp://example/mcp"}})


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


def test_rejects_wildcard_inside_planner_tools_list() -> None:
    with pytest.raises(ValueError, match="rather than a list"):
        parse_mcp_servers(
            {"web": {"url": "https://example/mcp", "planner_tools": ["*"]}}
        )


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


def test_rejects_case_variant_header_in_both_header_maps() -> None:
    with pytest.raises(ValueError, match="both 'headers' and 'env_http_headers'"):
        parse_mcp_servers(
            {
                "web": {
                    "url": "https://example/mcp",
                    "headers": {"Authorization": "Bearer literal"},
                    "env_http_headers": {"authorization": "WEB_TOKEN"},
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
        headers={"X-Client": "robot"},
        env_http_headers={"Authorization": "WEB_TOKEN"},
    )

    assert request_headers(spec) == {
        "X-Client": "robot",
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


def test_merge_rejects_duplicate_shared_and_robot_names() -> None:
    robot = (McpServerSpec(name="web", url="https://robot.example/mcp"),)
    shared = (McpServerSpec(name="web", url="https://shared.example/mcp"),)

    with pytest.raises(ValueError, match="duplicate MCP server name 'web'"):
        merge_mcp_servers(robot, shared)


def test_merge_preserves_order() -> None:
    robot = (McpServerSpec(name="mine", url="https://robot.example/mcp"),)
    shared = (
        McpServerSpec(name="lights", url="https://lights.example/mcp"),
        McpServerSpec(name="docs", url="https://docs.example/mcp"),
    )

    assert [spec.name for spec in merge_mcp_servers(robot, shared)] == [
        "mine",
        "lights",
        "docs",
    ]


def test_load_mcp_config_reads_only_the_mcp_section(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FACILITY_TOKEN", "secret")
    path = tmp_path / "mcp.yaml"
    path.write_text(
        "label: lab-a\n"
        "mcp_servers:\n"
        "  lights:\n"
        "    url: https://lights.example/mcp\n"
        "    headers:\n"
        "      Authorization: Bearer ${oc.env:FACILITY_TOKEN}\n"
        "    planner_tools: [set_scene]\n"
        "  dock:\n"
        "    url: https://dock.example/mcp\n"
    )

    assert load_mcp_config(path) == (
        McpServerSpec(
            name="lights",
            url="https://lights.example/mcp",
            headers={"Authorization": "Bearer secret"},
            planner_tools=("set_scene",),
        ),
        McpServerSpec(name="dock", url="https://dock.example/mcp"),
    )


def test_load_mcp_config_ignores_unresolved_values_outside_the_section(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("VLA_CKPT", raising=False)
    path = tmp_path / "mcp.yaml"
    path.write_text(
        "vla:\n"
        "  checkpoint: ${oc.env:VLA_CKPT}\n"
        "mcp_servers:\n"
        "  dock:\n"
        "    url: https://dock.example/mcp\n"
    )

    assert load_mcp_config(path) == (
        McpServerSpec(name="dock", url="https://dock.example/mcp"),
    )


def test_load_mcp_config_rejects_unresolved_value_in_the_section(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("WEB_TOKEN", raising=False)
    path = tmp_path / "mcp.yaml"
    path.write_text(
        "mcp_servers:\n"
        "  web:\n"
        "    url: https://web.example/mcp\n"
        "    headers:\n"
        "      Authorization: Bearer ${oc.env:WEB_TOKEN}\n"
    )

    with pytest.raises(ValueError, match="WEB_TOKEN"):
        load_mcp_config(path)


def test_load_mcp_config_without_section_yields_no_specs(tmp_path: Path) -> None:
    path = tmp_path / "mcp.yaml"
    path.write_text("label: lab-a\n")

    assert load_mcp_config(path) == ()


def test_load_mcp_config_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="MCP config not found"):
        load_mcp_config(tmp_path / "missing.yaml")
