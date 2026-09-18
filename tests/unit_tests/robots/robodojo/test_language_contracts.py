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


def environment(description=None, labels=None):
    return SimpleNamespace(
        obs_manager=SimpleNamespace(
            desc_manager=SimpleNamespace(get_one_description=lambda: [description])
        ),
        gen_instruction=lambda index: [
            "Pick <target> beside <reference>; hold <target>."
        ],
        scene_manager=SimpleNamespace(
            layout_manager=SimpleNamespace(
                get_label_descriptions=lambda label, env_idx: (labels or {}).get(
                    label, []
                )
            )
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
def test_label_fallback_fills_every_occurrence(description):
    env = environment(description, {"target": ["blue car"], "reference": ["black bin"]})
    assert resolve_instruction(env) == "Pick blue car beside black bin; hold blue car."


def test_missing_manager_uses_label_fallback():
    env = environment(labels={"target": ["blue car"], "reference": ["black bin"]})
    del env.obs_manager
    assert "blue car" in resolve_instruction(env)


def test_label_lookup_exception_has_resolution_context():
    env = environment()

    def missing(**kwargs):
        raise KeyError("target")

    env.scene_manager.layout_manager.get_label_descriptions = missing
    with pytest.raises(RuntimeError, match="Cannot resolve.*target"):
        resolve_instruction(env)


@pytest.mark.parametrize("label", [[], ["<unknown>"], [""]])
def test_unresolvable_language_fails_clearly(label):
    with pytest.raises(RuntimeError, match="Cannot resolve RoboDojo task instruction"):
        resolve_instruction(environment(labels={"target": label}))


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
