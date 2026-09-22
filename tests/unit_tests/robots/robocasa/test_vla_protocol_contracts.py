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

"""Offline contracts for RoboCasa live-task VLA execution."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from robots.robocasa.primitives import RoboCasaPrimitives
from robots.robocasa.prompt_bundle import system_prompt
from robots.robocasa.vla_server import _normalize_legacy_processor_geometry
from rpent.prompt.utils import format_prompt
from rpent.tools import ToolResult, iter_tools


class _RecordingRldx:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def run(
        self,
        prompt: str,
        max_chunks: int,
        n_action_steps: int,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "prompt": prompt,
                "max_chunks": max_chunks,
                "n_action_steps": n_action_steps,
                **kwargs,
            }
        )
        return {"ok": True, "prompt": prompt, "status": "cap"}


def _fake_primitives(task_language: str) -> tuple[RoboCasaPrimitives, _RecordingRldx]:
    rldx = _RecordingRldx()
    primitives = RoboCasaPrimitives.__new__(RoboCasaPrimitives)
    primitives.env = SimpleNamespace(
        current_raw_obs={"language": task_language},
        get_task_language=lambda: task_language,
    )
    primitives._rldx = rldx
    primitives._vla_desync = True
    primitives._recording = False
    primitives.record_frame = lambda: None
    return primitives, rldx


def test_prompt_requires_live_task_language_and_fresh_geometry(tmp_path: Path) -> None:
    rendered = format_prompt(
        system_prompt(),
        variables={
            "task_name": "OpenDrawer",
            "memory_dir": str(tmp_path / "results"),
        },
    )

    assert "complete live task_language" in rendered
    assert "Never shorten, paraphrase" in rendered
    assert "Only re-stage after 2-3 consecutive calls" in rendered
    assert "contact nor task progress" in rendered
    assert (
        "Historical entries may name vla_act, use_prompt, or atomic prompts" in rendered
    )
    assert "Never replay stored xyz, xy, pixels, base poses" in rendered
    assert "{{" not in rendered


def test_vla_tool_schema_hides_historical_prompt_override() -> None:
    vla_specs = {
        tool.name: {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema,
        }
        for tool in iter_tools(RoboCasaPrimitives)
        if tool.name.startswith("rldx_")
    }

    assert set(vla_specs) == {"rldx_skill", "rldx_arm"}
    for spec in vla_specs.values():
        schema = spec["input_schema"]
        assert schema["required"] == ["prompt"]
        assert "use_prompt" not in schema["properties"]
        assert "complete live task_language" in spec["description"]


@pytest.mark.parametrize("tool_name", ["rldx_skill", "rldx_arm"])
def test_vla_requires_prompt_before_execution(tool_name: str) -> None:
    primitives, rldx = _fake_primitives("Open the drawer.")
    declared = getattr(primitives, tool_name)

    with pytest.raises(ValidationError):
        declared.args_schema.model_validate({})
    with pytest.raises(TypeError, match="prompt"):
        declared()
    assert rldx.calls == []


def test_vla_always_uses_live_task_language_and_preserves_continuity() -> None:
    task_language = "Pick the squash up and place it in the microwave."
    primitives, rldx = _fake_primitives(task_language)

    first = primitives.rldx_skill(
        prompt="Pick the squash up.",
        use_prompt=True,
        max_chunks=3,
    )
    second = primitives.rldx_arm(
        prompt="Place it in the microwave.",
        use_prompt=True,
        max_chunks=4,
    )

    assert [call["prompt"] for call in rldx.calls] == [task_language, task_language]
    assert rldx.calls[0]["force_reset"] is True
    assert rldx.calls[1]["force_reset"] is False
    assert primitives._vla_desync is False
    assert first.data["effective_prompt"] == task_language
    assert first.data["effective_max_chunks"] == 3
    assert first.data["prompt_overridden"] is True
    assert first.data["requested_prompt"] == "Pick the squash up."
    assert second.data["effective_prompt"] == task_language
    assert second.data["effective_max_chunks"] == 4
    assert second.data["prompt_overridden"] is True


def test_gpu_policy_chain_arguments_execute_the_current_vla_tool(tmp_path, monkeypatch):
    from rpent.dashboard.events import NullDashboardEventSink
    from rpent.memory import MemoryManager
    from rpent.session import EnvState
    from rpent.tools import Toolkit
    from tests.e2e_tests.robocasa import scenario

    primitives, rldx = _fake_primitives("Open the drawer")
    toolkit = Toolkit(
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(tmp_path / "memory"),
        state=EnvState(tmp_path / "state"),
    )
    toolkit.add_tool(primitives.rldx_skill)
    monkeypatch.setattr(
        toolkit, "get_env_state", lambda **kwargs: ToolResult(data=kwargs["result"])
    )
    for variable in ("RLDX_MAX_CHUNKS", "RLDX_ACTION_STEPS_PER_CHUNK"):
        monkeypatch.delenv(variable, raising=False)

    def execute_chain(*, action, **kwargs):
        result = toolkit.execute_tool(action.name, action.arguments)
        assert not result.is_error, result.to_dict()
        assert len(rldx.calls) == 1
        assert rldx.calls[0]["prompt"] == "Open the drawer"
        assert rldx.calls[0]["max_chunks"] == 1
        assert rldx.calls[0]["n_action_steps"] == 1

    monkeypatch.setattr(scenario, "run_scripted_policy_chain", execute_chain)
    scenario._policy_chain(
        tmp_path,
        Namespace(
            task_name="OpenDrawer",
            split="target",
            seed=0,
            vla_model_path="unused",
            cuda_device=0,
        ),
    )


def test_environment_max_chunks_locks_the_formal_protocol(monkeypatch) -> None:
    task_language = "Open the left drawer."
    primitives, rldx = _fake_primitives(task_language)
    monkeypatch.setenv("RLDX_MAX_CHUNKS", "40")

    result = primitives.rldx_skill(prompt=task_language, max_chunks=70)

    assert rldx.calls[0]["max_chunks"] == 40
    assert result.data["effective_max_chunks"] == 40


def test_ordinary_robocasa_keeps_default_max_chunks_at_70(monkeypatch) -> None:
    task_language = "Open the left drawer."
    primitives, rldx = _fake_primitives(task_language)
    monkeypatch.delenv("RLDX_MAX_CHUNKS", raising=False)

    result = primitives.rldx_skill(prompt=task_language)

    assert rldx.calls[0]["max_chunks"] == 70
    assert result.data["effective_max_chunks"] == 70


def test_environment_locks_all_formal_rldx_runtime_values(monkeypatch) -> None:
    task_language = "Open the left drawer."
    primitives, rldx = _fake_primitives(task_language)
    monkeypatch.setenv("RLDX_MAX_CHUNKS", "40")
    monkeypatch.setenv("RLDX_ACTION_STEPS_PER_CHUNK", "8")
    monkeypatch.setenv("RLDX_SETTLE_PATIENCE", "999")

    result = primitives.rldx_skill(
        prompt=task_language,
        max_chunks=2,
        n_action_steps=1,
        settle_patience=2,
    )

    assert rldx.calls[0]["max_chunks"] == 40
    assert rldx.calls[0]["n_action_steps"] == 8
    assert rldx.calls[0]["settle_patience"] == 999
    assert result.data["effective_max_chunks"] == 40
    assert result.data["effective_n_action_steps"] == 8
    assert result.data["effective_settle_patience"] == 999


def test_invalid_vla_budgets_return_errors_without_executing_rldx(
    monkeypatch,
) -> None:
    monkeypatch.delenv("RLDX_MAX_CHUNKS", raising=False)
    monkeypatch.delenv("RLDX_ACTION_STEPS_PER_CHUNK", raising=False)
    monkeypatch.delenv("RLDX_SETTLE_PATIENCE", raising=False)

    for arguments, parameter in (
        ({"max_chunks": 0}, "max_chunks"),
        ({"n_action_steps": 0}, "n_action_steps"),
        ({"settle_patience": 0}, "settle_patience"),
    ):
        primitives, rldx = _fake_primitives("Open the left drawer.")

        result = primitives.rldx_skill(
            prompt="Open the left drawer.",
            **arguments,
        )

        assert result.to_dict() == {
            "error": f"{parameter} must be positive; VLA was not executed"
        }
        assert rldx.calls == []


def test_matching_vla_prompt_is_reported_without_override() -> None:
    task_language = "Open the left drawer."
    primitives, rldx = _fake_primitives(task_language)

    result = primitives.rldx_skill(prompt=task_language, use_prompt=False)

    assert rldx.calls[0]["prompt"] == task_language
    assert result.data["effective_prompt"] == task_language
    assert result.data["prompt_overridden"] is False
    assert "requested_prompt" not in result.data


def test_vla_does_not_run_without_environment_task_language() -> None:
    primitives, rldx = _fake_primitives("")

    result = primitives.rldx_skill(prompt="atomic fallback", use_prompt=True)

    assert "task language is unavailable" in result.error
    assert rldx.calls == []
    assert primitives._vla_desync is True


def test_legacy_rldx_processor_null_geometry_uses_release_defaults(monkeypatch) -> None:
    processor = SimpleNamespace(
        image_max_area=None,
        image_resize_m=None,
        random_crop_fraction=None,
        random_rotation_angle=None,
        color_jitter_params=None,
    )
    calls = []

    def fake_build(candidate):
        calls.append((candidate.image_max_area, candidate.image_resize_m))
        return "train-transform", "eval-transform"

    monkeypatch.setattr(
        "robots.robocasa.vla_server._build_processor_image_transforms",
        fake_build,
    )

    assert _normalize_legacy_processor_geometry(processor) is True
    assert processor.image_max_area == 65536
    assert processor.image_resize_m == 32
    assert processor.train_image_transform == "train-transform"
    assert processor.eval_image_transform == "eval-transform"
    assert calls == [(65536, 32)]


def test_current_rldx_processor_geometry_is_not_rebuilt(monkeypatch) -> None:
    processor = SimpleNamespace(image_max_area=131072, image_resize_m=64)

    def unexpected_build(candidate):
        raise AssertionError(f"unexpected transform rebuild for {candidate!r}")

    monkeypatch.setattr(
        "robots.robocasa.vla_server._build_processor_image_transforms",
        unexpected_build,
    )

    assert _normalize_legacy_processor_geometry(processor) is False
    assert processor.image_max_area == 131072
    assert processor.image_resize_m == 64


def test_scripted_grasp_composes_native_results_and_reports_failed_stage():
    import numpy as np

    class ScriptedPrimitives(RoboCasaPrimitives):
        def __init__(self):
            self.env = SimpleNamespace(
                gripper_qpos=np.array([0.1]), eef_pos=np.zeros(3)
            )
            self.targets = []
            self.gripper_commands = []

        def set_gripper(self, gripper, steps):
            self.gripper_commands.append((gripper, steps))
            return ToolResult(data={"ok": True})

        def move_to(self, xyz, **kwargs):
            self.targets.append(np.asarray(xyz).tolist())
            return ToolResult(data={"ok": len(self.targets) < 2, "steps": 200})

    primitives = ScriptedPrimitives()
    result = primitives.scripted_grasp([0.1, 0.2, 0.3])
    assert result.data == {"ok": False, "steps": 200, "stage": "descent"}
    assert primitives.gripper_commands == [(-1.0, 4)]
    assert np.allclose(primitives.targets, [[0.1, 0.2, 0.4], [0.1, 0.2, 0.3]])
