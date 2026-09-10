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


def test_recipe_exports_only_the_successful_attempt_after_reset(make_toolkit):
    toolkit, env, _ = make_toolkit(mode="exploration", attempts=2)
    assert not toolkit.execute_tool("set_gripper", {"steps": 1}).is_error
    assert toolkit.write_recipe("unsolved") == ""
    assert not toolkit.execute_tool("reset", {"reason": "try again"}).is_error
    env.terminated = True
    assert not toolkit.execute_tool("set_gripper", {"steps": 2}).is_error

    name = toolkit.write_recipe("cell")
    assert name == "cell_recipe.jsonl"
    recipe = toolkit._task_output_dir / name
    assert [json.loads(line) for line in recipe.read_text().splitlines()] == [
        {"action": "set_gripper", "gripper": -1.0, "steps": 2}
    ]


def test_recipe_excludes_failed_actions_in_a_successful_attempt(
    make_toolkit, monkeypatch
):
    toolkit, env, _ = make_toolkit()

    def fail(action):
        raise RuntimeError("action failed")

    with monkeypatch.context() as patch:
        patch.setattr(env, "step", fail)
        assert toolkit.execute_tool("set_gripper", {"steps": 1}).is_error
    env.terminated = True
    assert not toolkit.execute_tool("set_gripper", {"steps": 2}).is_error
    recipe = toolkit._task_output_dir / toolkit.write_recipe("cell")
    assert [json.loads(line)["steps"] for line in recipe.read_text().splitlines()] == [
        2
    ]
