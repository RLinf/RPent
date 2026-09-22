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

from types import SimpleNamespace

import pytest

from robots.robodojo import env_server, tools
from robots.robodojo.access import public_observation
from robots.robodojo.language import resolve_instruction, validate_instruction


def environment(description=None):
    return SimpleNamespace(
        obs_manager=SimpleNamespace(
            desc_manager=SimpleNamespace(get_one_description=lambda: [description])
        ),
        get_obs=lambda env_idx: {"instruction": "<target>", "state": {}, "vision": {}},
    )


def test_official_manager_is_shared_by_rpc_observation_and_eval(monkeypatch):
    env = environment("Pick up the mint green scissors by 10 cm.")
    monkeypatch.setattr(env_server, "_record_obs_frame", lambda *args: None)
    rpc = env_server.RoboDojoEnvFacade.get_task_language(SimpleNamespace(env=env))
    obs = env_server._obs_dict(env, None)
    assert rpc == obs["instruction"] == public_observation(obs)["instruction"]
    assert rpc == "Pick up the mint green scissors by 10 cm."


@pytest.mark.parametrize("description", [None, "", "Pick <target>."])
def test_invalid_manager_language_is_rejected(description):
    with pytest.raises(ValueError, match="instruction"):
        resolve_instruction(environment(description))


def test_missing_manager_is_not_silently_replaced():
    env = environment()
    del env.obs_manager
    with pytest.raises(AttributeError, match="obs_manager"):
        resolve_instruction(env)


def test_description_manager_errors_are_preserved():
    env = environment()

    def missing():
        raise KeyError("target")

    env.obs_manager.desc_manager.get_one_description = missing
    with pytest.raises(KeyError, match="target"):
        resolve_instruction(env)


@pytest.mark.parametrize("value", ["", "Pick <target>.", "Pick <target", None])
def test_unfilled_language_is_never_valid(value):
    with pytest.raises(ValueError, match="instruction"):
        validate_instruction(value)


def test_pick_defaults_to_official_language_and_rejects_template_before_policy():
    seen = []
    obs = {
        "state": {
            f"{arm}_{field}": value
            for arm in ("left", "right")
            for field, value in (("ee_pose", [0, 0, 1]), ("ee_joint_state", [1]))
        }
    }
    primitives = SimpleNamespace(
        env=SimpleNamespace(
            get_task_language=lambda: "Pick the blue car.", get_obs=lambda: obs
        ),
        vla_client=SimpleNamespace(
            predict=lambda value: seen.append(value["instruction"]) or []
        ),
        _check_cancelled=lambda: None,
    )
    result = tools.pi0_pick(primitives, None, max_chunks=1)
    assert seen == ["Pick the blue car."]
    assert result["instruction"] == seen[0]
    with pytest.raises(ValueError, match="unresolved"):
        tools.pi0_pick(primitives, None, "Pick <target>.", max_chunks=1)
    assert len(seen) == 1
