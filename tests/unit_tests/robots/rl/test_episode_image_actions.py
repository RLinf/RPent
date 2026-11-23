# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.
"""Behavioral regression tests for strict calls, batches and explicit episode perception."""

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
    execute_with_si_values,
)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(tools, "IMAGE_SIZE", 32)
    monkeypatch.setattr(
        templates, "default_variables", lambda: {"output_dir": "/offline/output"}
    )


def make(tmp_path, **kwargs):
    env = kwargs.pop("env", FakeEnv())
    tk = RLToolkit(
        runtime_kwargs={"env": env, "sam3_client": FakeSam3(bar_mask())},
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(tmp_path / "memory"),
        output_dir=tmp_path / "run",
        **kwargs,
    )
    return tk, env


def test_exact_tools_and_initial_images(tmp_path):
    tk, _ = make(tmp_path)
    assert {t.name for t in tk.list_tools()} == {
        "execute_actions",
        "view_env",
        "segment",
        "back_project",
    }
    assert not tk.initial_observation().images


@pytest.mark.parametrize(
    "name,args",
    [
        ("segment", {}),
        ("segment", {"step": 0, "prompt": " "}),
        ("segment", {"step": 0, "prompt": "cube", "point": [1, 2]}),
        ("segment", {"step": 0, "point": [-1, 1]}),
        ("back_project", {}),
        ("back_project", {"step": 0, "pixel": [1, 2], "box": [0, 0, 3, 3]}),
        ("back_project", {"step": 0, "pixel": [1, 2], "z_range_m": [0, 1]}),
        ("back_project", {"step": 0, "box": [5, 2, 3, 4]}),
        ("back_project", {"step": 0, "box": [0, 0, 4, 4], "z_range_m": [2, 1]}),
        ("execute_actions", {"actions": [{"action": "move_to", **{}}]}),
        (
            "execute_actions",
            {"actions": [{"action": "move_to", **{"orientation_rpy": [0, 0]}}]},
        ),
        (
            "execute_actions",
            {"actions": [{"action": "move_to", **{"xyz": [math.nan, 0, 1]}}]},
        ),
        ("execute_actions", {"actions": [{"action": "move_to", **{"target_yaw": 0}}]}),
        ("execute_actions", {"actions": [{"action": "set_gripper", **{"opening": 2}}]}),
    ],
)
def test_invalid_queries_never_advance(tmp_path, name, args):
    tk, env = make(tmp_path)
    result = tk.execute_tool(name, args)
    assert result.is_error
    assert env.actions == []
    assert tk.state.latest_step == 0


def test_batch_validates_every_item_before_any_physics(tmp_path):
    tk, env = make(tmp_path)
    result = execute_with_si_values(
        tk,
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
    assert execute_with_si_values(tk, "move_to", {"xyz": [0.2, 0, 1]}).is_error
    assert execute_with_si_values(tk, "execute_actions", {"actions": []}).is_error
    assert execute_with_si_values(
        tk,
        "execute_actions",
        {"actions": [{"action": "set_gripper", "opening": 1}] * 4},
    ).is_error


def test_batch_stops_at_contact_and_requires_explicit_view(tmp_path):
    tk, env = make(tmp_path, env=FakeEnv(block_gap=0.04))
    result = execute_with_si_values(
        tk,
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
    assert not result.images
    assert env.xyz == pytest.approx([0.2, -0.1, 1.0], abs=0.004)
    assert tk.state.latest_step == 2


def test_success_stops_batch_without_advancing_remaining_actions(tmp_path):
    tk, env = make(tmp_path, env=FakeEnv(success_below_z=0.95))
    result = execute_with_si_values(
        tk,
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
    result = execute_with_si_values(
        tk,
        "execute_actions",
        {
            "actions": [
                {"action": "move_to", "xyz": [0.2, 0, 1.1], "orientation_rpy": target}
            ]
        },
    )
    assert result.data["results"][0]["status"] == "reached"
    assert np.linalg.norm(env.xyz - [0.2, 0, 1.1]) < 0.004
    assert (Rotation.from_euler("xyz", target) * env.rotation.inv()).magnitude() < 0.02


@pytest.mark.parametrize("size", [512, 256])
def test_small_vlm_pixels_validate_before_native_depth_conversion(
    tmp_path, monkeypatch, size
):
    monkeypatch.setattr(tools, "IMAGE_SIZE", 1024)
    tk, _ = make(tmp_path, image_size=size)
    coordinate = size - 5
    result = execute_with_si_values(
        tk, "back_project", {"step": 0, "pixel": [coordinate, coordinate]}
    )
    assert not result.is_error
    native = int((coordinate + 0.5) * 1024 / size)
    assert result.data["world_xyz"] == pytest.approx(
        [(native - 512) / 1024, (native - 512) / 1024, 1], abs=0.001
    )
    assert execute_with_si_values(
        tk, "back_project", {"step": 0, "pixel": [size, coordinate]}
    ).is_error


def test_segmentation_not_found_is_text_and_raw_detection_is_preserved(tmp_path):
    from rpent.robots.components.sam3_client import Sam3Result

    tk, _ = make(tmp_path)
    tk._primitives._sam3.segment = lambda *a, **kw: Sam3Result(found=False, score=0.1)
    result = execute_with_si_values(
        tk, "segment", {"step": 0, "prompt": "missing target"}
    )
    assert result.data["code"] == "not_found"
    assert result.data["camera"] == "agentview"
    assert result.data["step"] == 0
    assert result.data["position_unit"] == "cm"
    assert "found" not in result.data and "score" not in result.data
    raw = tk.state.load("sam3_detection_1.json", step=0)
    assert raw["found"] is False and raw["score"] == 0.1


def test_mask_parameter_cannot_select_different_camera(tmp_path):
    tk, _ = make(tmp_path)
    mask = execute_with_si_values(tk, "segment", {"step": 0, "prompt": "cube"}).data[
        "mask_id"
    ]
    result = execute_with_si_values(
        tk, "back_project", {"mask_id": mask, "camera": "wrist"}
    )
    assert result.is_error


def test_required_tool_branches_preserve_array_shapes_and_acceptance():
    from jsonschema import Draft202012Validator

    from robots.rl.contracts import strict_tool
    from robots.rl.tools import LiberoRLPrimitives

    contract = strict_tool(LiberoRLPrimitives.move_to)
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

    contract = strict_tool(LiberoRLPrimitives.segment)
    point_branch = contract.input_schema["oneOf"][1]
    assert point_branch["properties"]["point"]["minItems"] == 2
    assert point_branch["properties"]["point"]["items"]["minimum"] == 0
    assert "maximum" not in point_branch["properties"]["point"]["items"]
    contract.validate_arguments({"step": 0, "point": [100, 200]})
    with pytest.raises(ValueError):
        contract.validate_arguments({"step": 0, "point": []})


def test_nonblank_patterns_match_complete_strings_in_decoding_grammars():
    import re

    from robots.rl.contracts import strict_tool
    from robots.rl.tools import LiberoRLPrimitives

    for owner, field, value in [
        (LiberoRLPrimitives.segment, "prompt", "milk carton"),
        (LiberoRLPrimitives.back_project, "mask_id", "m_0_agentview_1"),
    ]:
        schema = strict_tool(owner).input_schema
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
    visible = execute_with_si_values(tk, "segment", {"step": 0, "prompt": "red cup"})
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
    failure = execute_with_si_values(tk, "segment", {"step": 0, "prompt": "red cup"})
    assert failure.is_error and "segmentation_failed" in failure.to_text()
    raw = tk.state.load("sam3_detection_2.json", step=0)
    assert raw["category"] == "service_error"
    assert raw["error"]["type"] == "ConnectionError"


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
            function_tools=definitions, allow_text_output=True
        ),
        messages=[],
        model_settings=qwen_model_settings(max_tokens=32768, reasoning_effort="xhigh"),
    )
    context = SimpleNamespace(usage=SimpleNamespace(requests=0))
    cap = WorkflowCapability(tk)
    result = asyncio.run(cap.before_model_request(context, request))
    extra = result.model_settings["extra_body"]
    assert extra["chat_template_kwargs"]["enable_thinking"] is True
    assert result.model_settings["max_tokens"] == 32768
    if enabled:
        assert extra["tool_choice"] == "auto"
        structural = extra["response_format"]["format"]
        assert not structural["at_least_one"] and structural["stop_after_first"]
        for tag, original in zip(structural["tags"], definitions, strict=True):
            assert tag["content"]["json_schema"] == original.parameters_json_schema
            assert tag["content"]["max_whitespace_cnt"] == 16
        assert result.model_request_parameters.allow_text_output
    else:
        assert "response_format" not in extra and "tool_choice" not in extra
    assert not env.actions


def test_general_fractional_grammar_keeps_zero_expressible_and_declared_limits():
    from copy import deepcopy

    from pydantic_ai.tools import ToolDefinition

    from robots.rl.qwen_grammar import bounded_tool_format

    schema = {
        "type": "object",
        "properties": {
            "yaw_rate": {"type": "number", "minimum": -0.5, "maximum": 0.5},
            "vx": {"type": "number", "minimum": -35, "maximum": 35},
        },
        "required": ["yaw_rate", "vx"],
        "additionalProperties": False,
    }
    original = deepcopy(schema)
    tools = [ToolDefinition(name="move_base", parameters_json_schema=schema)]
    normal = bounded_tool_format(tools)["format"]["tags"][0]["content"]["json_schema"]
    adjusted = bounded_tool_format(tools, fractional_bounds_at_execution=True)
    generated = adjusted["format"]["tags"][0]["content"]["json_schema"]
    assert normal == schema == original
    assert generated["properties"]["yaw_rate"] == {"type": "number"}
    assert generated["properties"]["vx"] == original["properties"]["vx"]
    assert generated["required"] == original["required"]
