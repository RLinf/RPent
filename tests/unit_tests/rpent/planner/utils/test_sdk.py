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

from dataclasses import dataclass
from types import SimpleNamespace

from pydantic import BaseModel, Field, RootModel

from rpent.planner.utils.sdk import get_field, int_field, json_object, jsonable


def test_field_access_preserves_defaults_and_nullable_counters():
    for event in ({"tokens": None}, SimpleNamespace(tokens=None)):
        assert get_field(event, "missing", "fallback") == "fallback"
        assert get_field(event, "tokens", 99) is None
        assert int_field(event, "tokens") == 0
        assert int_field(event, "missing") == 0
    assert int_field({"tokens": "12"}, "tokens") == 12


def test_event_serialization_supports_sdk_shapes_and_binary_summaries():
    @dataclass
    class Block:
        data: bytes

    class Update(BaseModel):
        session_update: str = Field(alias="sessionUpdate")
        optional: str | None = None

    update = RootModel(Update(sessionUpdate="agent_message_chunk"))
    assert jsonable(update) == {"session_update": "agent_message_chunk"}
    assert jsonable(update, by_alias=True) == {"sessionUpdate": "agent_message_chunk"}
    assert jsonable({1: (Block(b"pixels"),)}) == {
        "1": [{"data": {"type": "bytes", "size": 6}}]
    }


def test_json_object_rejects_nonobjects_and_malformed_data():
    data = {"status": "success"}
    assert json_object(data) is data
    assert json_object('{"status": "success"}') == data
    for value in ("not JSON", "[]", "null", "true", None, 1):
        assert json_object(value) is None
