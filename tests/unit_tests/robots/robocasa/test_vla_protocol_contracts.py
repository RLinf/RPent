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

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from robots.robocasa.primitives import RoboCasaPrimitives
from robots.robocasa.prompt_bundle import system_prompt
from robots.robocasa.tools import TOOLS_SPEC
from robots.robocasa.vla_server import _normalize_legacy_processor_geometry
from rpent.prompt.utils import format_prompt


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
        spec["name"]: spec for spec in TOOLS_SPEC if spec["name"].startswith("rldx_")
    }

    assert set(vla_specs) == {"rldx_skill", "rldx_arm"}
    for spec in vla_specs.values():
        schema = spec["input_schema"]
        assert schema["required"] == ["prompt"]
        assert "use_prompt" not in schema["properties"]
        assert "complete live task_language" in spec["description"]


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
    assert first["effective_prompt"] == task_language
    assert first["prompt_overridden"] is True
    assert first["requested_prompt"] == "Pick the squash up."
    assert second["effective_prompt"] == task_language
    assert second["prompt_overridden"] is True


def test_matching_vla_prompt_is_reported_without_override() -> None:
    task_language = "Open the left drawer."
    primitives, rldx = _fake_primitives(task_language)

    result = primitives.rldx_skill(prompt=task_language, use_prompt=False)

    assert rldx.calls[0]["prompt"] == task_language
    assert result["effective_prompt"] == task_language
    assert result["prompt_overridden"] is False
    assert "requested_prompt" not in result


def test_vla_does_not_run_without_environment_task_language() -> None:
    primitives, rldx = _fake_primitives("")

    result = primitives.rldx_skill(prompt="atomic fallback", use_prompt=True)

    assert "task language is unavailable" in result["error"]
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
