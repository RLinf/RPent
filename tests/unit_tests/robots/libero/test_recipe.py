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


def test_action_and_perception_commands_keep_completion_order_and_full_parameters(
    make_toolkit,
):
    toolkit, env, _ = make_toolkit()
    calls = [
        ("view_env_state", {"step": 0}),
        (
            "segment",
            {"prompt": "bowl", "camera": "wrist", "step": 0, "min_score": 0.35},
        ),
        ("back_project", {"row": 4, "col": 4, "step": 0}),
        ("set_gripper", {"steps": "2"}),
        ("segment", {"prompt": "lid", "camera": "agentview", "step": 0}),
    ]
    expected = []
    for name, arguments in calls:
        result = toolkit.execute_tool(name, arguments)
        assert not result.is_error, result.error
        args = toolkit._tools[name].args_schema.model_validate(arguments)
        expected.append({"action": name, **args.model_dump(mode="json")})
    assert not env.terminated
    assert len(toolkit.state.records()) == 2
    assert export(toolkit) == expected
    assert export(toolkit) == expected
    assert not (toolkit.state._output_dir / "cell_recipe.jsonl").exists()


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
    assert [call["action"] for call in export(toolkit)] == [
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


def test_capture_failure_excludes_the_call(make_toolkit, monkeypatch):
    toolkit, env, _ = make_toolkit()

    def fail():
        raise RuntimeError("capture failed")

    with monkeypatch.context() as patch:
        patch.setattr(env, "raw_obs", fail)
        assert toolkit.execute_tool("set_gripper", {"steps": 1}).is_error
    assert not toolkit.execute_tool("view_env_state", {}).is_error
    assert [call["action"] for call in export(toolkit)] == ["view_env_state"]
