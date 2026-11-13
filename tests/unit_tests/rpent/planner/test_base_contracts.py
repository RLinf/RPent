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

from rpent.planner.base import (
    add_mcp_prefix,
    display_tool_name,
    strip_mcp_prefix,
)


def test_add_mcp_prefix_namespaces_bare_names_once() -> None:
    assert add_mcp_prefix("finish") == "mcp__rpent__finish"
    assert add_mcp_prefix("mcp__rpent__finish") == "mcp__rpent__finish"


def test_strip_mcp_prefix_recognizes_rpent_tools_only() -> None:
    assert strip_mcp_prefix("mcp__rpent__finish") == "finish"
    # An external server's ``finish`` keeps its namespace, so finish detection
    # never mistakes it for RPent's own.
    assert strip_mcp_prefix("mcp__web__finish") == "mcp__web__finish"
    assert strip_mcp_prefix("Read") == "Read"


def test_display_tool_name_shortens_mcp_names() -> None:
    assert display_tool_name("mcp__rpent__finish") == "finish"
    assert display_tool_name("mcp__web__search") == "web:search"
    assert display_tool_name("Read") == "Read"
    # Non-MCP names, including pydantic-ai's ``<server>_<tool>``, pass through.
    assert display_tool_name("web_search") == "web_search"
