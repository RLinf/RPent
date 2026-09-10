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

import numpy as np
import pytest

from robots.robocasa import tools
from robots.robocasa.prompt_bundle import system_prompt
from robots.robocasa.vla_server import _normalize_legacy_processor_geometry
from rpent.prompt.utils import format_prompt
from rpent.tools import ToolContext


def _record_vla(run):
    calls = []

    def record(prompt, max_chunks, n_action_steps, **kwargs):
        calls.append(
            {
                "prompt": prompt,
                "max_chunks": max_chunks,
                "n_action_steps": n_action_steps,
                **kwargs,
            }
        )
        return {"ok": True, "prompt": prompt, "status": "cap"}

    run.toolkit._robot._rldx.run = record
    return calls


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


def test_vla_schema_exposes_budgets_and_hides_legacy_prompt_override():
    for t in (tools.rldx_skill, tools.rldx_arm):
        schema = t.input_schema
        assert schema["required"] == ["prompt"]
        assert set(schema["properties"]) == {
            "prompt",
            "base_clip",
            "max_chunks",
            "force_reset",
            "n_action_steps",
            "settle_patience",
            "settle_eps",
        }
        assert "complete live task_language" in t.description


def test_vla_uses_live_language_and_run_defaults(make_toolkit):
    run = make_toolkit()
    calls = _record_vla(run)
    first = run.toolkit.execute_tool(
        "rldx_skill", {"prompt": "atomic pick", "use_prompt": True}
    )
    second = run.toolkit.execute_tool("rldx_arm", {"prompt": "atomic place"})
    assert not first.is_error and not second.is_error
    assert [call["prompt"] for call in calls] == [run.env.language] * 2
    assert [call["force_reset"] for call in calls] == [True, False]
    assert [call["base_clip"] for call in calls] == [None, 0.1]
    for result in (first, second):
        data = result.data["log"]["result"]
        assert data["effective_max_chunks"] == 70
        assert data["effective_n_action_steps"] == 8
        assert data["effective_settle_patience"] == 999
        assert data["effective_prompt"] == run.env.language
        assert data["prompt_overridden"] is True
    assert first.data["log"]["result"]["requested_prompt"] == "atomic pick"
    assert run.toolkit._robot._vla_desync is False
    for name in ("rldx_skill", "rldx_arm"):
        result = run.toolkit.execute_tool(
            name,
            {
                "prompt": run.env.language,
                "max_chunks": 3,
                "n_action_steps": 4,
                "settle_patience": 5,
            },
        )
        assert not result.is_error
        data = result.data["log"]["result"]
        assert data["effective_max_chunks"] == calls[-1]["max_chunks"] == 3
        assert data["effective_n_action_steps"] == calls[-1]["n_action_steps"] == 4
        assert data["effective_settle_patience"] == calls[-1]["settle_patience"] == 5


def test_environment_overrides_call_budgets_at_execution(make_toolkit, monkeypatch):
    run = make_toolkit(
        RLDX_MAX_CHUNKS=40, RLDX_ACTION_STEPS_PER_CHUNK=6, RLDX_SETTLE_PATIENCE=99
    )
    calls = _record_vla(run)
    for name in ("rldx_skill", "rldx_arm"):
        result = run.toolkit.execute_tool(
            name,
            {
                "prompt": run.env.language,
                "max_chunks": 2,
                "n_action_steps": 1,
                "settle_patience": 2,
            },
        )
        assert not result.is_error
        data = result.data["log"]["result"]
        assert data["effective_max_chunks"] == 40
        assert data["effective_n_action_steps"] == 6
        assert data["effective_settle_patience"] == 99
        assert data["prompt_overridden"] is False
        assert "requested_prompt" not in data
        assert result.data["log"]["command"]["max_chunks"] == 2
    assert all(
        (call["max_chunks"], call["n_action_steps"], call["settle_patience"])
        == (40, 6, 99)
        for call in calls
    )
    monkeypatch.setenv("RLDX_MAX_CHUNKS", "5")
    result = run.toolkit.execute_tool(
        "rldx_skill", {"prompt": run.env.language, "max_chunks": 2}
    )
    assert not result.is_error
    assert result.data["log"]["result"]["effective_max_chunks"] == 5
    assert calls[-1]["max_chunks"] == 5


@pytest.mark.parametrize(
    ("variable", "parameter"),
    [
        ("RLDX_MAX_CHUNKS", "max_chunks"),
        ("RLDX_ACTION_STEPS_PER_CHUNK", "n_action_steps"),
        ("RLDX_SETTLE_PATIENCE", "settle_patience"),
    ],
)
@pytest.mark.parametrize("source", ["environment", "arguments"])
def test_invalid_effective_budgets_fail_without_vla_execution(
    make_toolkit, variable, parameter, source
):
    run = make_toolkit(**({variable: 0} if source == "environment" else {}))
    calls = _record_vla(run)
    arguments = {"prompt": run.env.language}
    if source == "arguments":
        arguments[parameter] = 0
    result = run.toolkit.execute_tool("rldx_skill", arguments)
    assert result.error == f"{parameter} must be positive; VLA was not executed"
    assert not calls
    assert not run.env.actions
    assert run.toolkit._robot._vla_desync is True


def test_vla_rejects_missing_live_language(make_toolkit):
    run = make_toolkit()
    run.env.language = ""
    calls = _record_vla(run)
    result = run.toolkit.execute_tool("rldx_skill", {"prompt": "atomic fallback"})
    assert result.is_error and "task language is unavailable" in result.error
    assert result.data["log"]["result"]["effective_prompt"] == ""
    assert not calls
    assert run.toolkit._robot._vla_desync is True


def test_rollout_records_once_and_preserves_or_reseeds_history(make_toolkit):
    run = make_toolkit(RLDX_MAX_CHUNKS=1)
    tk = run.toolkit
    for name in ("rldx_skill", "rldx_arm"):
        result = tk.execute_tool(name, {"prompt": run.env.language})
        assert not result.is_error
        assert result.data["log"]["result"]["steps_applied"] == 2
        # Hitting the chunk cap with no task success is an ordinary VLA result.
        assert result.data["log"]["result"]["status"] == "cap"
    assert len(run.env.actions) == len(tk._frames) == 4
    assert [int(frame[0, 0, 0]) for frame in tk._frames] == [1, 2, 3, 4]
    np.testing.assert_allclose(run.env.actions[0][7:11], 1)
    np.testing.assert_allclose(run.env.actions[2][7:11], 0.1)
    assert [options["reset_memory"] for _, options in run.model.calls] == [
        [True],
        [False],
    ]
    before_manual = run.model.calls[1][0]["video.robot0_agentview_left"]
    assert before_manual.shape == (1, 2, 4, 4, 3)
    assert before_manual[0, :, 0, 0, 0].tolist() == [0, 2]
    tk.execute_tool("release", {"steps": 1})
    assert tk._robot._vla_desync is True
    tk.execute_tool("rldx_arm", {"prompt": run.env.language})
    assert run.model.calls[2][1]["reset_memory"] == [True]
    assert run.model.calls[2][0]["video.robot0_agentview_left"][
        0, :, 0, 0, 0
    ].tolist() == [5, 5]
    tk.execute_tool("rldx_arm", {"prompt": run.env.language, "force_reset": True})
    assert run.model.calls[3][1]["reset_memory"] == [True]
    assert len(run.env.actions) == len(tk._frames) == 9


def test_vla_checks_current_call_cancellation_after_inference(
    make_toolkit, monkeypatch
):
    run = make_toolkit(RLDX_MAX_CHUNKS=1)
    contexts = []
    original = ToolContext.check_cancelled

    def checkpoint(ctx):
        contexts.append(ctx)
        original(ctx)

    monkeypatch.setattr(ToolContext, "check_cancelled", checkpoint)
    run.model.on_predict = lambda: contexts[-1]._cancel_event.set()
    result = run.toolkit.execute_tool("rldx_skill", {"prompt": run.env.language})
    assert result.is_error and "cancelled" in result.error
    assert len(run.model.calls) == 1
    assert not run.env.actions and not run.toolkit._frames
    run.model.on_predict = lambda: None
    result = run.toolkit.execute_tool("rldx_skill", {"prompt": run.env.language})
    assert not result.is_error
    assert len(run.env.actions) == len(run.toolkit._frames) == 2


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
