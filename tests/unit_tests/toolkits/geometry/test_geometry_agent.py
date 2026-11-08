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

"""Geometry delegation uses real artifact/tool paths with offline model outputs."""

import argparse
from types import SimpleNamespace

import numpy as np
import pytest

from robots.libero import robot_spec
from robots.libero.tools import LiberoPrimitives
from rpent.robots.components.molmo_client import MolmoResult
from rpent.robots.components.sam3_client import Sam3Result
from rpent.session import EnvState
from rpent.tools.toolkit import ToolCancelled
from toolkits.geometry.geometry_agent import GeometryAgent


@pytest.fixture
def scene(tmp_path):
    state = EnvState(tmp_path / "episode")
    rows, cols = np.indices((6, 8))
    world = np.stack((cols * 0.01, rows * 0.02, 0.8 + rows * 0.01), axis=-1)
    with state.record_step(state={}):
        state.save("agentview.png", np.zeros((6, 8, 3), dtype=np.uint8))
        state.save("agentview_world.npz", world.astype(np.float32))
        state.save("agentview_depth.npz", np.ones((6, 8), dtype=np.float32))
    return state


def agent_for(state, *, point=(5, 3), masks=None, ground=None):
    calls = []

    def segment(image, **kwargs):
        calls.append(kwargs)
        mask = masks[len(calls) - 1] if masks is not None else np.ones((6, 8), bool)
        return Sam3Result(found=mask is not None, mask=mask, score=0.9)

    primitives = LiberoPrimitives(
        env=object(),
        model=object(),
        sam3_client=SimpleNamespace(segment=segment),
        check_cancelled=lambda: None,
    )
    selected = MolmoResult(found=point is not None, point_xy=point, image_size=(8, 6))
    grounder = SimpleNamespace(ground=ground or (lambda image, query: selected))
    agent = GeometryAgent(
        state=state,
        segment=primitives.segment,
        check_cancelled=lambda: None,
        grounder=grounder,
    )
    return agent, calls


def test_surface_point_uses_grounded_row_col_and_persists_evidence(scene):
    agent, sam_calls = agent_for(scene)
    result = agent.localize("the visible cup handle").data
    assert result["status"] == "usable"
    assert result["pixel_rc"] == [3, 5]
    assert result["world_xyz"] == [0.05, 0.06, 0.83]
    assert result["definition"] == "visible_material_surface_point"
    assert result["quality"]["calibrated_error_m"] is None
    assert result["calls"] == {"grounding": 1, "segment": 0, "back_project": 1}
    assert sam_calls == [] and len(scene.records()) == 1
    saved = scene.load(result["geometry_artifact"], step=0)
    assert saved["pixel_rc"] == [3, 5] and saved["step"] == 0


def test_wrong_text_mask_retries_once_with_grounded_positive_point(scene):
    wrong = np.ones((6, 8), dtype=bool)
    wrong[3, 5] = False
    right = np.zeros((6, 8), dtype=bool)
    right[1:5, 3:7] = True
    agent, calls = agent_for(scene, masks=[wrong, right])
    result = agent.localize("the yellow mug", measurement="visible_region").data
    assert result["status"] == "usable"
    assert len(calls) == 2 and calls[1]["point"] == [3, 5]
    assert calls[0]["text_prompt"] == "the yellow mug"
    assert all(call["min_score"] == 0.2 for call in calls)
    assert not result["segment_attempts"][0]["contains_target_pixel"]
    assert result["segment_attempts"][1]["contains_target_pixel"]
    mask = scene.load(result["mask_artifact"], step=0)
    cloud = scene.load(result["pointcloud_artifact"], step=0)
    assert mask.dtype == bool and np.array_equal(mask, right)
    np.testing.assert_allclose(cloud, scene.load("agentview_world.npz", step=0)[right])
    np.testing.assert_allclose(result["world_xyz"], np.median(cloud, axis=0))
    assert result["definition"] == "visible_region_coordinate_median"
    assert result["quality"]["n_valid"] == 16
    assert result["sam_score"] == 0.9 and len(scene.records()) == 1


def test_matching_text_mask_does_not_retry(scene):
    agent, calls = agent_for(scene)
    assert (
        agent.localize("cup", measurement="visible_region").data["status"] == "usable"
    )
    assert len(calls) == 1


def test_segmentation_budget_requests_new_view_instead_of_synonym_search(scene):
    agent, calls = agent_for(scene, masks=[None, None])
    data = agent.localize("cup", measurement="visible_region").data
    assert data["status"] == "need_observation" and data["world_xyz"] is None
    assert len(calls) == 2 and data["calls"]["grounding"] == 1
    assert data["observation_request"]["execution_owner"] == "parent_planner"
    assert data["surface_point_xyz"] == [0.05, 0.06, 0.83]


def test_unidentified_target_never_runs_sam_or_projection(scene):
    agent, calls = agent_for(scene, point=None)
    data = agent.localize("hidden hinge").data
    assert data["status"] == "need_observation" and data["world_xyz"] is None
    assert data["evidence_status"] == "unknown"
    assert data["calls"] == {"grounding": 1, "segment": 0, "back_project": 0}
    assert calls == []


def test_visual_service_failure_retains_failed_request_evidence(scene):
    import httpx

    def unavailable(image, query):
        raise httpx.ConnectError("vision endpoint unavailable")

    agent, calls = agent_for(scene, ground=unavailable)
    data = agent.localize("cup").data
    assert data["status"] == "error" and data["world_xyz"] is None
    assert data["calls"]["grounding"] == 1 and calls == []
    saved = scene.load(data["geometry_artifact"], step=0)
    assert saved["error"] == "vision endpoint unavailable"


def test_unavailable_step_does_not_start_perception(scene):
    agent, calls = agent_for(scene)
    data = agent.localize("cup", step=100).data
    assert data["status"] == "error" and data["code"] == "observation_unavailable"
    assert calls == []


@pytest.mark.parametrize("point", [(8, 2), (-1, 2), (2, float("nan"))])
def test_invalid_visual_points_cannot_index_the_world_map(scene, point):
    agent, calls = agent_for(scene, point=point)
    data = agent.localize("cup").data
    assert data["status"] == "error" and data["world_xyz"] is None
    assert data["calls"]["back_project"] == 0 and calls == []


def test_shape_mismatch_fails_before_model_request(scene):
    scene.save("agentview_world.npz", np.ones((6, 7, 3)), step=0)
    agent, _ = agent_for(scene)
    data = agent.localize("cup").data
    assert "shapes" in data["error"] and data["calls"]["grounding"] == 0


def test_invalid_point_depth_cannot_become_a_coordinate(scene):
    depth = np.ones((6, 8), dtype=np.float32)
    depth[3, 5] = np.nan
    scene.save("agentview_depth.npz", depth, step=0)
    agent, _ = agent_for(scene)
    data = agent.localize("cup").data
    assert data["status"] == "need_observation" and data["world_xyz"] is None


def test_insufficient_depth_in_correct_mask_needs_view_without_retry(scene):
    depth = np.zeros((6, 8), dtype=np.float32)
    depth[3, 5] = 1
    scene.save("agentview_depth.npz", depth, step=0)
    agent, calls = agent_for(scene)
    data = agent.localize("cup", measurement="visible_region").data
    assert data["reason"] == "too_few_valid_region_points"
    assert data["quality"]["n_valid"] == 1 and len(calls) == 1
    assert data["world_xyz"] is None


def test_latest_frame_is_pinned_before_visual_inference(scene):
    def ground(image, query):
        with scene.record_step(state={}):
            scene.save("agentview.png", np.zeros((6, 8, 3), dtype=np.uint8))
            scene.save("agentview_world.npz", np.full((6, 8, 3), 5.0))
            scene.save("agentview_depth.npz", np.ones((6, 8)))
        return MolmoResult(found=True, point_xy=(5, 3), image_size=(8, 6))

    agent, _ = agent_for(scene, ground=ground)
    data = agent.localize("cup", measurement="visible_region").data
    assert data["step"] == 0 and data["world_xyz"][2] < 1
    assert scene.load(data["geometry_artifact"], step=0)["step"] == 0


def test_high_resolution_measurement_preserves_float16_provenance(scene):
    scene.save("agentview_high.png", np.zeros((6, 8, 3), dtype=np.uint8), step=0)
    world = np.full((6, 8, 3), 0.9237, dtype=np.float16)
    scene.save("agentview_world_high.npz", world, step=0)
    agent, _ = agent_for(scene)
    data = agent.localize("cup", measurement="visible_region").data
    assert data["source_dtype"] == "float16" and data["resolution"] == "high"
    np.testing.assert_array_equal(data["world_xyz"], world[0, 0].astype(float))
    assert data["quality"]["calibrated_error_m"] is None


def test_mask_persistence_failure_is_not_reported_as_usable(scene, monkeypatch):
    save = scene.save

    def fail_mask(name, value, **kwargs):
        return None if name.startswith("segment_mask_") else save(name, value, **kwargs)

    monkeypatch.setattr(scene, "save", fail_mask)
    agent, _ = agent_for(scene)
    data = agent.localize("cup", measurement="visible_region").data
    assert data["status"] == "error" and "persist segment mask" in data["error"]
    attempt = data["segment_attempts"][0]
    assert (
        "persist segment mask"
        in scene.load(attempt["segment_artifact"], step=0)["error"]
    )


def test_cancellation_after_visual_inference_prevents_internal_tool_calls(scene):
    agent, calls = agent_for(scene)
    checks = 0

    def cancel():
        nonlocal checks
        checks += 1
        if checks == 3:
            raise ToolCancelled("cancelled")

    agent._check_cancelled = cancel
    with pytest.raises(ToolCancelled):
        agent.localize("cup", measurement="visible_region")
    assert calls == []


def test_real_visual_adapter_and_geometry_run_offline(scene, monkeypatch):
    from pydantic_ai.models.test import TestModel

    import rpent.planner.base as base

    model = TestModel(custom_output_args={"point_xy": [5, 3]})
    monkeypatch.setattr(base, "build_api_model", lambda *a: model)
    agent, _ = agent_for(scene)
    agent._grounder = None
    agent._model = "test:geometry"
    assert agent.localize("cup handle").data["world_xyz"] == [0.05, 0.06, 0.83]
    assert model.last_model_request_parameters.function_tools == []


def test_parent_dispatch_delegates_without_capturing_or_advancing_state(
    scene, tmp_path, monkeypatch
):
    from pydantic_ai.models.test import TestModel

    import rpent.planner.base as base
    from robots.libero.toolkit import LiberoToolkit
    from rpent.dashboard.events import NullDashboardEventSink
    from rpent.memory import MemoryManager
    from rpent.utils import templates

    def init(self, **kwargs):
        self._state = scene
        self._primitives = LiberoPrimitives(
            env=object(),
            model=object(),
            sam3_client=object(),
            check_cancelled=self.raise_if_cancelled,
        )

    monkeypatch.setattr(LiberoToolkit, "init_primitives", init)
    monkeypatch.setattr(
        templates, "default_variables", lambda: {"output_dir": str(tmp_path / "parent")}
    )
    monkeypatch.setattr(
        base,
        "build_api_model",
        lambda *a: TestModel(custom_output_args={"point_xy": [5, 3]}),
    )
    parent = LiberoToolkit(
        runtime_kwargs={},
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(tmp_path / "memory"),
        geometry_model="test:geometry",
        state_output_dir=tmp_path / "parent",
    )

    def unexpected_capture(**kwargs):
        pytest.fail("localize must not capture a fresh environment state")

    monkeypatch.setattr(parent, "get_env_state", unexpected_capture)
    data = parent.execute_tool("localize", {"query": "cup handle"}).data
    assert data["world_xyz"] == [0.05, 0.06, 0.83]
    assert len(scene.records()) == 1
    assert {t.name for t in parent.list_tools()}.isdisjoint({"segment", "back_project"})


def test_recipe_exports_delegate_request_without_internal_sam_calls(scene, tmp_path):
    import json

    from robots.libero.tools import write_recipe_from_states

    wrong = np.ones((6, 8), dtype=bool)
    wrong[3, 5] = False
    agent, _ = agent_for(scene, masks=[wrong, np.ones((6, 8), dtype=bool)])
    agent.localize("the yellow cup", measurement="visible_region")
    with scene.record_step(state={}, terminated=True):
        pass
    name = write_recipe_from_states(scene, "geometry", output_dir=tmp_path)
    commands = [json.loads(line) for line in (tmp_path / name).read_text().splitlines()]
    assert commands == [
        {
            "action": "localize",
            "query": "the yellow cup",
            "camera": "agentview",
            "measurement": "visible_region",
        }
    ]


@pytest.mark.parametrize(
    "planner,model,expected",
    [
        ("api", "test:vision", "test:vision"),
        ("codex", "test", "codex:test"),
        ("codex", None, "codex"),
    ],
)
def test_geometry_model_inherits_supported_planner_configuration(
    monkeypatch, planner, model, expected
):
    monkeypatch.delenv("CODEX_MODEL", raising=False)
    parser = argparse.ArgumentParser()
    for option in (
        "output-dir",
        "memory-profile",
        "memory-dir",
        "planner",
        "model",
        "base-url",
    ):
        parser.add_argument("--" + option)
    robot_spec.get_robot_spec().add_cli_args(parser, False)
    argv = [
        "--suite",
        "libero_10_swap",
        "--task",
        "9",
        "--planner",
        planner,
        "--base-url",
        "https://provider.test/v1",
    ]
    if model:
        argv += ["--model", model]
    args = parser.parse_args(argv)
    config = robot_spec.get_robot_spec().parse_config(args)
    assert config.task_desc["geometry_model"] == expected
    assert config.task_desc["geometry_base_url"] == "https://provider.test/v1"
    args.geometry_model = "test:independent"
    args.geometry_base_url = "https://geometry.test/v1"
    config = robot_spec.get_robot_spec().parse_config(args)
    assert config.task_desc["geometry_model"] == "test:independent"
    assert config.task_desc["geometry_base_url"] == "https://geometry.test/v1"


@pytest.mark.parametrize(
    "profile,mode,policy",
    [
        ("hf", "eval", "vla"),
        ("local", "eval", "vla"),
        ("local", "explore", "vla"),
        ("local", "eval", "wam"),
    ],
)
def test_active_prompts_delegate_localization_without_old_tool_names(
    profile, mode, policy
):
    spec = robot_spec.get_robot_spec()
    variables = {
        "memory_profile": profile,
        "mode": mode,
        "policy_kind": policy,
        "memory_dir": "/offline/memory",
        "memory_inbox": "/offline/inbox",
        "output_dir": "/offline/output",
        "recipe_tag": "test",
        "reference_tag": "test_s0",
        "suite": "libero_10_swap",
        "task": 9,
        "seed": 1,
        "session_number": 1,
        "session_max": 2,
    }
    for name in ("system", "user"):
        prompt = spec.prompts.render(name, variables=variables)
        assert "localize" in prompt
        assert "back_project" not in prompt and "`segment`" not in prompt
