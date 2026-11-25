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

"""Offline contract tests for external MCP server configuration parsing."""

from __future__ import annotations

from pathlib import Path

import pytest

from rpent.utils.mcp_servers import (
    ALL_TOOLS,
    REDACTED_VALUE,
    McpServerSpec,
    load_mcp_config,
    merge_mcp_servers,
    parse_mcp_servers,
    public_spec,
)


def test_parse_defaults_to_all_tools_and_unsynced() -> None:
    specs = parse_mcp_servers({"power": {"url": "http://127.0.0.1:8765/mcp"}})
    assert specs == (
        McpServerSpec(
            name="power",
            url="http://127.0.0.1:8765/mcp",
            headers={},
            tools=ALL_TOOLS,
            sync_with_robot=False,
        ),
    )


def test_parse_resolves_env_headers_tools_and_sync_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POWER_TOKEN", "s3cr3t")
    specs = parse_mcp_servers(
        {
            "power-1": {
                "url": "https://power.example/mcp",
                "headers": {"Authorization": "Bearer ${oc.env:POWER_TOKEN}"},
                "tools": ["read_power_meter", "read_power_meter", "read_temp"],
                "sync_with_robot": True,
                "timeout_s": 15,
            }
        }
    )
    assert specs == (
        McpServerSpec(
            name="power-1",
            url="https://power.example/mcp",
            headers={"Authorization": "Bearer s3cr3t"},
            tools=("read_power_meter", "read_temp"),
            sync_with_robot=True,
            timeout_s=15.0,
        ),
    )


def test_parse_unset_env_header_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MISSING_MCP_TOKEN", raising=False)
    with pytest.raises(ValueError, match="MISSING_MCP_TOKEN"):
        parse_mcp_servers(
            {
                "power": {
                    "url": "https://power.example/mcp",
                    "headers": {"Authorization": "Bearer ${oc.env:MISSING_MCP_TOKEN}"},
                }
            }
        )


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("", "server name"),
        ("_power", "server name"),
        ("power_", "trailing '_'"),
        ("lab__power", "without '__'"),
        ("lab.power", "server name"),
        ("rpent", "reserved"),
    ],
)
def test_spec_rejects_invalid_server_names(name: str, expected: str) -> None:
    with pytest.raises(ValueError, match=expected):
        McpServerSpec(name=name, url="http://127.0.0.1:8765/mcp")


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("stdio://power", "http\\(s\\) URL"),
        ("http://", "hostname"),
        ("https:///mcp", "hostname"),
        ("https://user:pass@power.example/mcp", "must not embed credentials"),
        ("https://:pass@power.example/mcp", "must not embed credentials"),
        ("https://user@power.example/mcp", "must not embed credentials"),
    ],
)
def test_spec_rejects_invalid_urls(url: str, expected: str) -> None:
    with pytest.raises(ValueError, match=expected):
        McpServerSpec(name="power", url=url)


@pytest.mark.parametrize(
    ("raw_tools", "expected"),
    [
        ([], "must not be empty"),
        ("read_power", "must be '\\*' or a list"),
        (["*"], "write '\\*' rather than a list"),
        ([""], "non-empty strings"),
        ([123], "non-empty strings"),
        (42, "must be '\\*' or a list"),
    ],
)
def test_parse_rejects_invalid_tools(raw_tools: object, expected: str) -> None:
    with pytest.raises(ValueError, match=expected):
        parse_mcp_servers(
            {"power": {"url": "http://127.0.0.1:8765/mcp", "tools": raw_tools}}
        )


def test_parse_rejects_unknown_keys_and_non_bool_sync() -> None:
    with pytest.raises(ValueError, match="unknown key"):
        parse_mcp_servers(
            {
                "power": {
                    "url": "http://127.0.0.1:8765/mcp",
                    "planner_tools": ["read_power"],
                }
            }
        )
    with pytest.raises(ValueError, match="sync_with_robot must be a boolean"):
        parse_mcp_servers(
            {"power": {"url": "http://127.0.0.1:8765/mcp", "sync_with_robot": "yes"}}
        )
    with pytest.raises(ValueError, match="headers values must be strings"):
        parse_mcp_servers(
            {"power": {"url": "http://127.0.0.1:8765/mcp", "headers": {"X-Id": 1}}}
        )


@pytest.mark.parametrize(
    "bad_timeout",
    [0, -5, True, False, "30", float("inf"), float("nan")],
)
def test_parse_rejects_invalid_timeout_s(bad_timeout: object) -> None:
    with pytest.raises(ValueError, match="timeout_s must be a positive number"):
        parse_mcp_servers(
            {"power": {"url": "http://127.0.0.1:8765/mcp", "timeout_s": bad_timeout}}
        )


def test_load_mcp_config_reads_only_mcp_servers_section(tmp_path: Path) -> None:
    config = tmp_path / "mcp.yaml"
    config.write_text(
        "other_section:\n"
        "  ignored: ${oc.env:DOES_NOT_EXIST_OK}\n"
        "mcp_servers:\n"
        "  power:\n"
        "    url: http://127.0.0.1:8765/mcp\n"
        "    tools: [read_power_meter]\n"
    )
    assert load_mcp_config(None) == ()
    specs = load_mcp_config(config)
    assert len(specs) == 1
    assert specs[0].name == "power"
    assert specs[0].tools == ("read_power_meter",)


def test_load_mcp_config_rejects_missing_or_non_mapping(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="MCP config not found"):
        load_mcp_config(tmp_path / "missing.yaml")

    bad = tmp_path / "list.yaml"
    bad.write_text("- item1\n- item2\n")
    with pytest.raises(ValueError, match="must be a mapping"):
        load_mcp_config(bad)


def test_merge_mcp_servers_rejects_duplicates() -> None:
    a = McpServerSpec(name="power", url="http://127.0.0.1:8765/mcp")
    b = McpServerSpec(name="dock", url="http://127.0.0.1:8766/mcp")
    assert merge_mcp_servers((a,), (b,)) == (a, b)
    with pytest.raises(ValueError, match="duplicate MCP server name 'power'"):
        merge_mcp_servers((a,), (a,))


def test_public_spec_redacts_headers_and_url_query_fragment() -> None:
    spec = McpServerSpec(
        name="power",
        url="https://power.example/mcp?token=secret#frag",
        headers={"Authorization": "Bearer secret", "X-Team": "lab"},
        tools=("read_power_meter",),
        sync_with_robot=True,
        timeout_s=30.0,
    )
    recorded = public_spec(spec)
    assert recorded == {
        "name": "power",
        "url": f"https://power.example/mcp?{REDACTED_VALUE}#{REDACTED_VALUE}",
        "headers": {
            "Authorization": REDACTED_VALUE,
            "X-Team": REDACTED_VALUE,
        },
        "tools": ("read_power_meter",),
        "sync_with_robot": True,
        "timeout_s": 30.0,
    }
