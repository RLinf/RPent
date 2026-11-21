# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.
"""Behavioral regression tests for strict calls, batches and image episodes."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robots.rl import tools
from robots.rl.toolkit import RLToolkit
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from rpent.utils import templates
from tests.unit_tests.robots.rl.test_libero_rl_contracts import (
    FakeEnv,
    FakeSam3,
    bar_mask,
)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(tools, "IMAGE_SIZE", 32)
    monkeypatch.setattr(
        templates, "default_variables", lambda: {"output_dir": "/offline/output"}
    )


def make(tmp_path, workflow="episode_image", action_mode="single", **kwargs):
    env = kwargs.pop("env", FakeEnv())
    tk = RLToolkit(
        runtime_kwargs={"env": env, "sam3_client": FakeSam3(bar_mask())},
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(tmp_path / "memory"),
        observation_mode="adaptive",
        output_dir=tmp_path / "run",
        workflow=workflow,
        action_mode=action_mode,
        **kwargs,
    )
    return tk, env


@pytest.mark.parametrize(
    "workflow,mode,expected",
    [
        (
            "episode",
            "single",
            {"move_to", "set_gripper", "view_env", "segment", "back_project", "finish"},
        ),
        (
            "episode",
            "batch",
            {"execute_actions", "view_env", "segment", "back_project", "finish"},
        ),
        (
            "episode_image",
            "single",
            {"move_to", "set_gripper", "segment", "back_project", "finish"},
        ),
        (
            "episode_image",
            "batch",
            {"execute_actions", "segment", "back_project", "finish"},
        ),
    ],
)
def test_exact_tools_and_initial_images(tmp_path, workflow, mode, expected):
    tk, _ = make(tmp_path, workflow, mode)
    assert {t.name for t in tk.list_tools()} == expected
    assert len(tk.initial_observation().images) == (
        2 if workflow == "episode_image" else 0
    )
    if workflow == "episode_image":
        for name in ["segment", "back_project"]:
            assert "step" not in tk._tools[name].input_schema["properties"]


@pytest.mark.parametrize(
    "name,args",
    [
        ("segment", {}),
        ("segment", {"prompt": " "}),
        ("segment", {"prompt": "cube", "point": [1, 2]}),
        ("segment", {"point": [-1, 1]}),
        ("segment", {"prompt": "cube", "step": 0}),
        ("back_project", {}),
        ("back_project", {"pixel": [1, 2], "box": [0, 0, 3, 3]}),
        ("back_project", {"pixel": [1, 2], "z_range_m": [0, 1]}),
        ("back_project", {"box": [5, 2, 3, 4]}),
        ("back_project", {"box": [0, 0, 4, 4], "z_range_m": [2, 1]}),
        ("move_to", {}),
        ("move_to", {"orientation_rpy": [0, 0]}),
        ("move_to", {"xyz": [math.nan, 0, 1]}),
        ("move_to", {"target_yaw": 0}),
        ("set_gripper", {"opening": 2}),
    ],
)
def test_invalid_queries_never_advance(tmp_path, name, args):
    tk, env = make(tmp_path)
    result = tk.execute_tool(name, args)
    assert result.is_error
    assert env.actions == []
    assert tk.state.latest_step == 0


def test_batch_validates_every_item_before_any_physics(tmp_path):
    tk, env = make(tmp_path, action_mode="batch")
    result = tk.execute_tool(
        "execute_actions",
        {
            "actions": [
                {"action": "move_to", "xyz": [0.2, 0, 1]},
                {"action": "set_gripper", "opening": 7},
            ]
        },
    )
    assert result.is_error and not env.actions
    assert tk.state.latest_step == 0
    assert tk.execute_tool("move_to", {"xyz": [0.2, 0, 1]}).is_error
    assert tk.execute_tool("execute_actions", {"actions": []}).is_error
    assert tk.execute_tool(
        "execute_actions", {"actions": [{"action": "set_gripper", "opening": 1}] * 4}
    ).is_error


def test_batch_stops_at_contact_and_returns_only_final_images(tmp_path):
    tk, env = make(tmp_path, action_mode="batch", env=FakeEnv(block_gap=0.04))
    result = tk.execute_tool(
        "execute_actions",
        {
            "actions": [
                {"action": "move_to", "xyz": [0.2, -0.1, 1.0]},
                {"action": "set_gripper", "opening": 0},
                {"action": "move_to", "xyz": [0.4, -0.1, 1.0]},
            ]
        },
    )
    assert result.data["executed_count"] == 2 and result.data["stopped_at"] == 1
    assert result.data["stop_reason"] == "blocked"
    assert result.data["results"][0]["status"] == "reached"
    assert result.data["results"][1]["status"] == "partial"
    assert len(result.images) == 2
    assert env.xyz == pytest.approx([0.2, -0.1, 1.0], abs=0.004)
    assert tk.state.latest_step == 2


def test_success_stops_batch_without_advancing_remaining_actions(tmp_path):
    tk, env = make(tmp_path, action_mode="batch", env=FakeEnv(success_below_z=0.95))
    result = tk.execute_tool(
        "execute_actions",
        {
            "actions": [
                {"action": "move_to", "xyz": [0.1, -0.2, 0.9]},
                {"action": "set_gripper", "opening": 0},
            ]
        },
    )
    assert result.data["executed_count"] == 1
    assert result.data["stop_reason"] == "terminated"
    assert result.data["final_state"]["terminated"]


def test_latest_mask_expires_after_motion(tmp_path):
    tk, _ = make(tmp_path)
    result = tk.execute_tool("segment", {"prompt": "cube"})
    assert "found" not in result.data and "score" not in result.data
    mask = result.data["mask_id"]
    assert "world_xyz" in tk.execute_tool("back_project", {"mask_id": mask}).data
    moved = tk.execute_tool("move_to", {"xyz": [0.2, -0.1, 1.0]})
    assert len(moved.images) == 2
    stale = tk.execute_tool("back_project", {"mask_id": mask})
    assert stale.is_error and "older observation" in stale.to_text()


def test_world_commands_convert_through_rotated_controller_frame(tmp_path):
    class RotatedBase(FakeEnv):
        def _observe(self):
            super()._observe()
            self.frame = Rotation.from_euler("z", 0.7).as_matrix()
            self.last_obs["control_state"]["control_to_world_rotation"] = self.frame

        def step(self, action):
            converted = np.asarray(action).copy()
            converted[:3] = self.frame @ converted[:3]
            converted[3:6] = self.frame @ converted[3:6]
            return super().step(converted)

    tk, env = make(tmp_path, env=RotatedBase())
    target = [math.pi, 0.2, 0.4]
    result = tk.execute_tool(
        "move_to", {"xyz": [0.2, 0, 1.1], "orientation_rpy": target}
    )
    assert result.data["status"] == "reached"
    assert np.linalg.norm(env.xyz - [0.2, 0, 1.1]) < 0.004
    assert (Rotation.from_euler("xyz", target) * env.rotation.inv()).magnitude() < 0.02


@pytest.mark.parametrize("size", [512, 256])
def test_small_vlm_pixels_validate_before_native_depth_conversion(
    tmp_path, monkeypatch, size
):
    monkeypatch.setattr(tools, "IMAGE_SIZE", 1024)
    tk, _ = make(tmp_path, image_size=size)
    coordinate = size - 5
    result = tk.execute_tool("back_project", {"pixel": [coordinate, coordinate]})
    assert not result.is_error
    native = int((coordinate + 0.5) * 1024 / size)
    assert result.data["world_xyz"] == pytest.approx(
        [(native - 512) / 1024, (native - 512) / 1024, 1], abs=0.001
    )
    assert tk.execute_tool("back_project", {"pixel": [size, coordinate]}).is_error


def test_segmentation_not_found_is_text_and_raw_detection_is_preserved(tmp_path):
    from rpent.robots.components.sam3_client import Sam3Result

    tk, _ = make(tmp_path)
    tk._primitives._sam3.segment = lambda *a, **kw: Sam3Result(found=False, score=0.1)
    result = tk.execute_tool("segment", {"prompt": "missing target"})
    assert result.data["code"] == "not_found"
    assert result.to_text().startswith("No target was detected")
    assert "found" not in result.data and "score" not in result.data
    raw = tk.state.load("sam3_detection_1.json", step=0)
    assert raw["found"] is False and raw["score"] == 0.1


def test_mask_parameter_cannot_select_different_camera(tmp_path):
    tk, _ = make(tmp_path)
    mask = tk.execute_tool("segment", {"prompt": "cube"}).data["mask_id"]
    result = tk.execute_tool("back_project", {"mask_id": mask, "camera": "wrist"})
    assert result.is_error


def test_snapshot_contracts_reject_blank_entries_before_execution():
    from robots.rl.contracts import strict_tool
    from robots.rl.snapshot import Snapshot, SubtaskPlan

    for declaration, arguments in [
        (Snapshot.change_query, {"queries": ["  "]}),
        (Snapshot.update_todo, {"items": ["  "]}),
        (SubtaskPlan.update_todo, {"items": [{"text": "  ", "status": "pending"}]}),
    ]:
        contract = strict_tool(declaration, image_size=1024)
        with pytest.raises(ValueError):
            contract.validate_arguments(arguments)
        assert "Example:" in contract.description


def test_required_tool_branches_preserve_array_shapes_and_acceptance():
    from jsonschema import Draft202012Validator

    from robots.rl.contracts import strict_tool
    from robots.rl.tools import LiberoRLPrimitives

    contract = strict_tool(LiberoRLPrimitives.move_to, image_size=1024)
    schema = contract.input_schema
    target = {"xyz": [0.1, 0.2, 0.3], "orientation_rpy": [3.14159265, 0, 0]}
    contract.validate_arguments(target)
    contract.validate_arguments({"orientation_rpy": target["orientation_rpy"]})
    contract.validate_arguments(
        {"xyz": None, "orientation_rpy": target["orientation_rpy"]}
    )
    with pytest.raises(ValueError):
        contract.validate_arguments({"xyz": None, "orientation_rpy": None})
    for branch in schema["anyOf"]:
        # Reproduce the affected decoder's shallow branch selection.
        selected = {**schema, **branch}
        selected.pop("anyOf")
        Draft202012Validator(selected).validate(target)
        assert not Draft202012Validator(selected).is_valid({"xyz": []})
        assert branch["properties"]["xyz"]["items"] == {"type": "number"}
        assert branch["properties"]["xyz"]["minItems"] == 3


def test_segment_branch_preserves_pixel_bounds_for_required_tool_decoding():
    from robots.rl.contracts import strict_tool
    from robots.rl.tools import LiberoRLPrimitives

    contract = strict_tool(LiberoRLPrimitives.segment, image_size=1024)
    point_branch = contract.input_schema["oneOf"][1]
    assert point_branch["properties"]["point"]["minItems"] == 2
    assert point_branch["properties"]["point"]["items"]["maximum"] == 1023
    contract.validate_arguments({"point": [100, 200], "step": 0})
    with pytest.raises(ValueError):
        contract.validate_arguments({"point": [], "step": 0})


def test_nonblank_patterns_match_complete_strings_in_decoding_grammars():
    import re

    from robots.rl.contracts import strict_tool
    from robots.rl.tools import LiberoRLPrimitives

    for owner, field, value in [
        (LiberoRLPrimitives.segment, "prompt", "milk carton"),
        (LiberoRLPrimitives.back_project, "mask_id", "m_0_agentview_1"),
    ]:
        schema = strict_tool(owner, image_size=1024).input_schema
        pattern = schema["properties"][field]["pattern"]
        assert re.fullmatch(pattern, value)
        assert re.fullmatch(pattern, chr(10) + "word" + chr(9))
        assert not re.search(pattern, " " + chr(9) + chr(10))


def test_sam_audit_preserves_exact_mask_score_threshold_and_service_error(tmp_path):
    from rpent.robots.components.sam3_client import Sam3Result

    tk, _ = make(tmp_path)
    mask = bar_mask()
    score = 0.987654321
    tk._primitives._sam3.segment = lambda *a, **kw: Sam3Result(
        found=True, score=score, box=[1.25, 2.5, 8.75, 9.5], mask=mask
    )
    visible = tk.execute_tool("segment", {"prompt": "red cup"})
    assert "score" not in visible.data and "found" not in visible.data
    raw = tk.state.load("sam3_detection_1.json", step=0)
    assert raw["score"] == score and raw["min_score"] == 0.2
    assert raw["box_xyxy"] == [1.25, 2.5, 8.75, 9.5]
    assert raw["query"] == {"prompt": "red cup", "point": None}
    assert raw["category"] == "detected"
    np.testing.assert_array_equal(tk.state.load(raw["mask"], step=0).astype(bool), mask)

    def failed(*a, **kw):
        raise ConnectionError("SAM worker unavailable")

    tk._primitives._sam3.segment = failed
    failure = tk.execute_tool("segment", {"prompt": "red cup"})
    assert failure.is_error and "segmentation_failed" in failure.to_text()
    raw = tk.state.load("sam3_detection_2.json", step=0)
    assert raw["category"] == "service_error"
    assert raw["error"]["type"] == "ConnectionError"


def test_zap_keeps_raw_detections_out_of_model_feedback(tmp_path):
    from robots.rl.snapshot import Snapshot

    tk, _ = make(tmp_path)
    sam = tk._primitives._sam3
    sam.segment_queries = lambda image, queries: [
        [sam.segment(image, text_prompt=query)] for query in queries
    ]
    snapshot = Snapshot(
        tk._primitives, environment="libero", task="test task", image_size=32
    )
    snapshot.queries = ["red cup"]
    observed = snapshot.observe(tk._robot_state())
    assert "score" not in str(observed.data) and "found" not in str(observed.data)
    raw = tk.state.load("sam3_detection_1.json", step=0)
    assert raw["min_score"] == 0.2 and raw["category"] == "detected"
    assert raw["score"] is not None
    assert raw["mask_shape"] == list(bar_mask().shape)


@pytest.mark.parametrize(
    "model_name,enabled", [("qwen3.8-27b", True), ("other", False)]
)
def test_empty_qwen_tool_has_bounded_native_grammar_and_same_contract(
    tmp_path, model_name, enabled
):
    import asyncio
    from types import SimpleNamespace

    from pydantic_ai.models import ModelRequestParameters
    from pydantic_ai.tools import ToolDefinition

    from robots.rl.workflow import WorkflowCapability
    from rpent.planner.qwen import qwen_model_settings

    tk, env = make(tmp_path)
    definitions = [
        ToolDefinition(name=t.name, parameters_json_schema=t.input_schema)
        for t in tk.list_tools()
    ]
    request = SimpleNamespace(
        model=SimpleNamespace(model_name=model_name),
        model_request_parameters=ModelRequestParameters(
            function_tools=definitions, allow_text_output=False
        ),
        messages=[],
        model_settings=qwen_model_settings(max_tokens=32768, reasoning_effort="xhigh"),
    )
    context = SimpleNamespace(usage=SimpleNamespace(requests=0))
    cap = WorkflowCapability(tk, "system", "task", 100, timeout_s=7200)
    result = asyncio.run(cap.before_model_request(context, request))
    extra = result.model_settings["extra_body"]
    assert extra["chat_template_kwargs"]["enable_thinking"] is True
    assert result.model_settings["max_tokens"] == 32768
    if enabled:
        assert extra["tool_choice"] == "auto"
        structural = extra["response_format"]["format"]
        assert structural["at_least_one"] and structural["stop_after_first"]
        for tag, original in zip(structural["tags"], definitions, strict=True):
            assert tag["content"]["json_schema"] == original.parameters_json_schema
            assert tag["content"]["max_whitespace_cnt"] == 16
        assert not result.model_request_parameters.allow_text_output
    else:
        assert "response_format" not in extra and "tool_choice" not in extra
    assert not env.actions
