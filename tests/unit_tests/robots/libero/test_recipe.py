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

import json


def export(toolkit):
    name = toolkit.write_recipe("cell")
    assert name == "cell_recipe.jsonl"
    return [
        json.loads(line)
        for line in (toolkit._task_output_dir / name).read_text().splitlines()
    ]


def test_common_files_finish_validation_and_execution_errors_are_excluded(
    make_toolkit, monkeypatch
):
    toolkit, env, _ = make_toolkit()
    path = str(toolkit._task_output_dir / "note.txt")
    for name, arguments in [
        ("write_text_file", {"path": path, "content": "note"}),
        ("read_text_file", {"path": path}),
        ("list_dir", {}),
        ("read_image", {"name": "agentview_policy.png", "step": 0}),
    ]:
        assert not toolkit.execute_tool(name, arguments).is_error
    for name, arguments in [
        ("unknown", {}),
        ("set_gripper", {"steps": "invalid"}),
        ("segment", {}),
    ]:
        assert toolkit.execute_tool(name, arguments).is_error

    def fail(action):
        raise RuntimeError()

    with monkeypatch.context() as patch:
        patch.setattr(env, "step", fail)
        failed = toolkit.execute_tool("set_gripper", {"steps": 1})
    assert failed.is_error and failed.error == ""
    assert not toolkit.execute_tool("set_gripper", {"steps": 1}).is_error
    assert not toolkit.execute_tool(
        "finish", {"status": "success", "summary": "done"}
    ).is_error
    assert [call["action"] for call in export(toolkit)] == ["set_gripper"]


def test_reset_is_kept_with_both_attempts_and_refused_finish_is_excluded(make_toolkit):
    toolkit, _, _ = make_toolkit(mode="exploration", attempts=2)
    assert toolkit.execute_tool(
        "finish", {"status": "stuck", "summary": "first"}
    ).is_error
    for name, arguments in [
        ("segment", {"prompt": "bowl"}),
        ("set_gripper", {"steps": 1}),
        ("reset", {"reason": "try again"}),
        ("view_env_state", {}),
        ("segment", {"prompt": "lid"}),
        ("set_gripper", {"steps": 1}),
    ]:
        result = toolkit.execute_tool(name, arguments)
        assert not result.is_error, result.error
    recipe = export(toolkit)
    assert recipe[1] == {"action": "set_gripper", "gripper": -1.0, "steps": 1}
    assert [call["action"] for call in recipe] == [
        "segment",
        "set_gripper",
        "reset",
        "view_env_state",
        "segment",
        "set_gripper",
    ]


def test_recipe_does_not_infer_tool_errors_from_environment_success(make_toolkit):
    toolkit, env, _ = make_toolkit()
    result = toolkit.execute_tool(
        "pi0_doubled", {"prompt": "touch bowl", "max_chunks": 1}
    )
    assert not result.is_error and result.data["log"]["result"]["success"] is False
    assert not env.terminated
    assert [call["action"] for call in export(toolkit)] == ["pi0_doubled"]
