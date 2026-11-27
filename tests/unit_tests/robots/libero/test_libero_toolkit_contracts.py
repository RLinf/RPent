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

"""Offline contracts for the LIBERO toolkit."""

from __future__ import annotations

import argparse
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from robots.libero import robot_spec, toolkit
from robots.libero.observations import LiberoGeometryObservations
from robots.libero.prompts import evaluate
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from rpent.prompt.utils import Numbered
from rpent.robots import RunConfig
from rpent.robots.components.sam3_client import Sam3Result
from rpent.session import EnvState
from rpent.tools import Toolkit, ToolResult
from rpent.tools.common import CommonTools
from rpent.utils import templates

COMMON_TOOLS = {"read_text_file", "write_text_file", "list_dir", "finish"}

EVALUATION_TOOLS = COMMON_TOOLS | {
    "view_env_state",
    "move_to",
    "pi0_pick",
    "pi0_doubled",
    "release",
    "set_gripper",
    "rotate_wrist",
    "rotate_pitch",
    "move_pose",
    "view_camera_meta",
    "segment",
    "back_project",
}


def _record(step_idx: int = 0) -> SimpleNamespace:
    return SimpleNamespace(step_idx=step_idx, terminated=False)


def _tool_names(robot_toolkit: Toolkit) -> set[str]:
    return {definition.name for definition in robot_toolkit.list_tools()}


def _readonly_names(robot_toolkit: Toolkit) -> set[str]:
    return {
        definition.name
        for definition in robot_toolkit.list_tools()
        if definition.readonly
    }


def _run_config(memory_dir: Path, *, recipe_tag: str = "cell-s0") -> RunConfig:
    return RunConfig(
        recipe_tag=recipe_tag,
        output_dir=memory_dir.parent / "run",
        prompt_vars={"memory_dir": str(memory_dir)},
        task_desc={},
    )


@pytest.mark.parametrize("backend", ["cosmos-policy", "fast-wam"])
def test_wam_configuration_and_toolkit_use_local_memory(tmp_path, monkeypatch, backend):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir")
    parser.add_argument("--memory-profile")
    parser.add_argument("--memory-dir")
    parser.add_argument("--planner", default="api")
    parser.add_argument("--explore", action="store_true")
    spec = robot_spec.get_robot_spec()
    spec.add_cli_args(parser, False)
    args = parser.parse_args(
        [
            "--suite",
            "libero_spatial_task",
            "--task",
            "0",
            "--am-backend",
            backend,
            "--am-endpoint",
            "http://localhost:8116",
        ]
    )
    config = spec.parse_config(args)
    assert args.libero_type == "pro" and args.memory_profile == "local"
    assert Path(config.prompt_vars["memory_dir"]).name == backend
    assert "wam" in {
        item["name"] for item in spec.resolve_dashboard(args)["runtime_components"]
    }
    for name in ("system", "user"):
        prompt = spec.prompts.render(
            name, variables={**config.prompt_vars, "output_dir": str(tmp_path)}
        )
        assert "wam_act" in prompt and "pi0_pick" not in prompt
    monkeypatch.setattr(
        toolkit, "LiberoToolkit", lambda **kwargs: SimpleNamespace(**kwargs)
    )
    instance = robot_spec.get_toolkit(
        runtime_kwargs={},
        dashboard_events=NullDashboardEventSink(),
        config=config,
    )
    assert isinstance(instance.memory, MemoryManager)
    assert instance.memory.root == Path(config.prompt_vars["memory_dir"]).resolve()
    args.memory_dir = str(tmp_path / "custom-memory")
    assert spec.parse_config(args).prompt_vars["memory_dir"] == args.memory_dir
    args.memory_profile = "hf"
    with pytest.raises(ValueError, match="no compatible HF corpus"):
        spec.parse_config(args)


def test_toolkit_factory_configures_memory_access_by_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: list[dict[str, Any]] = []

    def fake_toolkit(**kwargs: Any) -> SimpleNamespace:
        captured.append(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(toolkit, "LiberoToolkit", fake_toolkit)
    memory_dir = tmp_path / "libero-memory"
    config = _run_config(memory_dir)

    evaluation = robot_spec.get_toolkit(
        runtime_kwargs={"env": "evaluation"},
        dashboard_events=NullDashboardEventSink(),
        config=config,
    )
    exploration = robot_spec.get_toolkit(
        runtime_kwargs={"env": "exploration"},
        dashboard_events=NullDashboardEventSink(),
        config=config,
        mode="exploration",
        attempts_per_session=2,
        state_output_dir=tmp_path / "state",
    )

    assert evaluation.memory.root == memory_dir.resolve()
    assert exploration.memory.root == memory_dir.resolve()
    evaluation_write = CommonTools(memory=evaluation.memory).write_text_file
    exploration_write = CommonTools(memory=exploration.memory).write_text_file
    own_draft = memory_dir / "_internal" / "inbox" / config.recipe_tag / "draft.md"
    with pytest.raises(PermissionError, match="writing to memory is denied"):
        evaluation_write(str(own_draft), "draft")
    assert exploration_write(str(own_draft), "draft").data["bytes_written"] == 5
    assert captured[0]["mode"] == "evaluation"
    assert captured[1]["mode"] == "exploration"
    assert captured[1]["attempts_per_session"] == 2


@pytest.mark.parametrize("enabled", [False, True])
def test_generic_localization_is_registered_with_libero_observations(
    tmp_path, monkeypatch, fake_single_arm_primitives, enabled
):
    fake = fake_single_arm_primitives.for_robot(toolkit.libero_tools.LiberoPrimitives)
    monkeypatch.setattr(toolkit.libero_tools, "LiberoPrimitives", fake)
    monkeypatch.setattr(toolkit.libero_tools, "dump_state", lambda *a, **kw: _record())
    monkeypatch.setattr(
        templates, "default_variables", lambda: {"output_dir": str(tmp_path)}
    )
    rows, cols = np.indices((40, 40))
    mask = (abs(rows - 20) <= 8) & (abs(cols - 22) <= 10)
    points = np.stack((cols * 0.002, rows * 0.002, 0.4 + mask * 0.04), axis=-1)
    pose = np.diag([1.0, -1.0, -1.0, 1.0])
    pose[2, 3] = 1.0
    sam = SimpleNamespace(
        segment=lambda *a, **kw: Sam3Result(found=True, mask=mask, score=0.9),
        segment_candidates=lambda *a, **kw: [
            Sam3Result(found=True, mask=mask, score=0.9)
        ],
    )
    tk = toolkit.LiberoToolkit(
        runtime_kwargs={"env": object(), "sam3_client": sam},
        generic_tools=True,
        geometry_grounder=object() if enabled else None,
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(tmp_path / "memory"),
        state_output_dir=tmp_path / "state",
    )
    assert _tool_names(tk) == EVALUATION_TOOLS | ({"geometry"} if enabled else set())
    assert {"back_project", "segment"} <= _readonly_names(tk)
    if enabled:
        assert "geometry" in _readonly_names(tk)
        schema = tk._tools["geometry"].input_schema
        required = {"query", "camera", "object_prompt", "required_geometry", "feature"}
        assert set(schema["required"]) == required
        assert set(schema["properties"]) == required | {"step"}
        quantities = schema["properties"]["required_geometry"]["items"]["enum"]
        assert set(quantities) == {
            "ring_center",
            "disk_center",
            "rectangle_center",
            "base_center",
        }
        assert set(schema["properties"]["feature"]["enum"]) == {
            "upper_inner_ring",
            "upper_outer_ring",
            "upper_face",
            "supported_round_base",
        }

    with tk.state.record_step(state={}) as step:
        tk.state.save("agentview_policy.png", np.ones((4, 4, 3), np.uint8), step=step)
        tk.state.save(
            "agentview_high.png", np.zeros((*mask.shape, 3), np.uint8), step=step
        )
        tk.state.save("wrist_high.png", np.zeros((600, 800, 3), np.uint8), step=step)
        tk.state.save("agentview_world_high.npz", points, step=step)
        tk.state.save(
            "agentview_metadata.json",
            {
                "height": mask.shape[0],
                "width": mask.shape[1],
                "intrinsic_K": np.eye(3).tolist(),
                "extrinsic_cam2world": pose.tolist(),
            },
            step=step,
        )
    if enabled:
        target = {
            "camera": "agentview",
            "query": "box",
            "object_prompt": "box",
            "step": step,
        }
        request = {
            **target,
            "required_geometry": ["rectangle_center"],
            "feature": "upper_face",
        }
        for args, invalid_fields in (
            (target, {"required_geometry", "feature"}),
            ({**request, "measurement": "surface_point"}, {"measurement"}),
            ({**request, "measurement": "visible_region"}, {"measurement"}),
            ({**request, "feature": "largest_face"}, {"feature"}),
            (
                {
                    **request,
                    "required_geometry": ["cylinder_base_center"],
                    "feature": "supported_base",
                },
                {"required_geometry", "feature"},
            ),
        ):
            errors = tk.execute_tool("geometry", args).data["errors"]
            assert {error["loc"][0] for error in errors} == invalid_fields
        measured = tk.execute_tool("geometry", request).data
        assert measured["status"] == "usable" and measured["camera"] == "agentview"
        np.testing.assert_allclose(
            measured["geometry"]["rectangle_center"]["value"],
            points[20, 22],
            atol=0.002,
        )
        rejected = tk.execute_tool("geometry", {**request, "camera": "wrist"}).data
        assert (
            rejected["code"] == "unsupported_camera"
            and "agentview" in rejected["error"]
        )
    expected_view = toolkit.libero_tools.view_env_state(step, state=tk.state)
    assert tk.execute_tool("view_env_state", {"step": step}) == expected_view
    after_action = tk.get_env_state(
        command={"action": "release"}, result={}, elapsed_s=0.5
    )
    assert after_action == ToolResult(
        data={**expected_view.data, "agent_elapsed_s": 0.5}, images=expected_view.images
    )

    selected = tk.execute_tool(
        "back_project",
        {
            "camera": "agentview",
            "step": step,
            "selection": {"image_shape": [2, 2], "points_rc": [[1, 0]]},
        },
    )
    assert selected.data["samples_xyz"] == [points[30, 10].tolist()]
    segmented = tk.execute_tool(
        "segment", {"camera": "agentview", "step": step, "prompt": "box"}
    )
    assert (
        segmented.data["found"]
        and segmented.data["evidence_status"] == "identity_unchecked"
    )
    assert tk.state.latest_step == step


def test_pinned_wrist_sample_uses_its_own_calibration(tmp_path):
    state = EnvState(tmp_path)
    for x in (0.0, 0.1):
        pose = np.eye(4)
        pose[0, 3] = x
        with state.record_step(state={}) as step:
            state.save("wrist_high.png", np.zeros((8, 8, 3), dtype=np.uint8), step=step)
            xyz = np.ones((8, 8, 3))
            state.save("wrist_world_high.npz", xyz, step=step)
            state.save(
                "wrist_metadata.json",
                {
                    "width": 4,
                    "height": 4,
                    "intrinsic_K": [[4, 0, 2], [0, 4, 2], [0, 0, 1]],
                    "extrinsic_cam2world": pose.tolist(),
                },
                step=step,
            )
    obs = LiberoGeometryObservations(state)
    historical = obs.get_observation(camera="wrist", step=0)
    latest = obs.get_observation(camera="wrist", step=-1)
    assert historical.camera_to_world[0, 3] == 0.0
    assert latest.camera_to_world[0, 3] == 0.1
    assert historical.world_frame_id == latest.world_frame_id
    np.testing.assert_array_equal(latest.intrinsics, [[8, 0, 4], [0, 8, 4], [0, 0, 1]])
    with pytest.raises(ValueError, match="calibrated RGB-D pair"):
        obs.get_observation(camera="agentview", step=0)


def test_geometry_requires_images_for_the_api_planner(tmp_path):
    from rpent.cli.main import _build_argparser

    spec = robot_spec.get_robot_spec()
    parser = _build_argparser()
    spec.add_cli_args(parser, False)
    base = [
        "--suite",
        "libero_spatial_task",
        "--task",
        "0",
        "--output-dir",
        str(tmp_path),
    ]
    geometry = ["--geometry", "--molmo-endpoint", "http://localhost:8118"]
    args = parser.parse_args([*base, "--planner", "api", "--no-images", *geometry])
    for configure in (spec.parse_config, spec.resolve_dashboard):
        with pytest.raises(ValueError, match=r"--geometry.*--no-images"):
            configure(args)
    rgbd = parser.parse_args([*base, "--planner", "api", "--no-images", "--rgbd-tools"])
    assert spec.parse_config(rgbd).prompt_vars["generic_tools"]
    sdk = parser.parse_args([*base, "--planner", "codex", "--no-images", *geometry])
    assert spec.parse_config(sdk).prompt_vars["geometry"]


def test_generic_tools_and_geometry_are_opt_in(
    tmp_path, monkeypatch, fake_single_arm_primitives
):
    parser = argparse.ArgumentParser()
    parser.add_argument("--planner", default="api")
    parser.add_argument("--memory-profile")
    parser.add_argument("--memory-dir")
    parser.add_argument("--output-dir")
    spec = robot_spec.get_robot_spec()
    spec.add_cli_args(parser, False)
    primitives = fake_single_arm_primitives.for_robot(
        toolkit.libero_tools.LiberoPrimitives
    )
    monkeypatch.setattr(toolkit.libero_tools, "LiberoPrimitives", primitives)
    monkeypatch.setattr(toolkit.libero_tools, "dump_state", lambda *a, **kw: _record())
    monkeypatch.setattr(robot_spec, "get_memory_dir", lambda _: tmp_path / "memory")
    monkeypatch.setattr(templates, "get_output_dir", lambda: tmp_path)
    arguments = [
        "--suite",
        "libero_spatial_task",
        "--task",
        "0",
        "--output-dir",
        str(tmp_path / "run"),
    ]
    for flags, generic, geometry in [
        ([], False, False),
        (["--no-rgbd-tools", "--no-geometry"], False, False),
        (["--rgbd-tools"], True, False),
        (["--geometry", "--molmo-endpoint", "http://localhost:8118"], True, True),
    ]:
        args = parser.parse_args(arguments + flags)
        config = spec.parse_config(args)
        assert config.prompt_vars["generic_tools"] is generic
        assert config.prompt_vars["geometry"] is geometry
        prompt = spec.prompts.render(
            "system", variables={**config.prompt_vars, "output_dir": str(tmp_path)}
        )
        assert ("points_rc" in prompt and "box_rc" in prompt) is generic
        assert ('back_project({"row":' in prompt) is not generic
        assert ("mutually exclusive" in prompt) is not generic
        assert ('geometry({"query":' in prompt) is geometry
        assert ('"object_prompt":"blue cup"' in prompt) is geometry
        assert ('"feature":"upper_inner_ring"' in prompt) is geometry
        assert ("insufficient_evidence" in prompt) is geometry
        assert ("highlighted object matches the full query" in prompt) is geometry
        assert ("bare table next to it" in prompt) is not geometry
        assert ("measure the actual support surface" in prompt) is geometry
        assert ("nudge and re-observe" in prompt) is not geometry
        assert (
            "two-camera relation protocol for disambiguating identical objects"
            in prompt
        ) is not geometry

        tk = robot_spec.get_toolkit(
            config=config,
            runtime_kwargs={
                "env": object(),
                "sam3_client": object(),
                "geometry_grounder": object() if geometry else None,
            },
            dashboard_events=NullDashboardEventSink(),
            state_output_dir=tmp_path / "state",
        )
        schemas = {tool.name: tool.args_schema for tool in tk.list_tools()}
        examples = re.findall(
            r"\b(back_project|segment|geometry)\((\{.*?\})\)", prompt, re.S
        )
        expected = {"back_project", "segment"} | ({"geometry"} if geometry else set())
        assert {name for name, _ in examples} == expected
        for name, example in examples:
            # The default prompt uses ROW, COL and NN for the selected pixel and step.
            payload = json.loads(re.sub(r"\b(?:ROW|COL|NN)\b", "0", example))
            schemas[name].model_validate(payload, strict=True)
            if generic and name == "segment" and "point" in payload:
                assert "image_shape" in payload


@pytest.mark.parametrize("guide", ["strict_hybrid_guide.md", "pro_hybrid_guide.md"])
def test_shared_guides_leave_localization_call_formats_to_the_prompt(guide):
    text = (Path(robot_spec.__file__).with_name("guides") / guide).read_text()
    assert not re.search(
        r'\b(?:back_project|segment|geometry)\(|"row"\s*:|resolution["`]?[=:]|\brow_range\b',
        text,
    )


@pytest.mark.parametrize(
    "settings",
    [
        {"mode": "eval", "memory_profile": "hf"},
        {"mode": "eval", "memory_profile": "local"},
        {"mode": "explore"},
        {"policy_kind": "wam"},
    ],
)
def test_rgbd_menu_preserves_the_planners_task_and_memory_instructions(settings):
    prompts = robot_spec.get_robot_spec().prompts
    baseline = prompts.system(settings)
    original_user = prompts.user(settings)

    def task_sections(prompt):
        # Only localization and the workflow's segment aid name tool arguments.
        return {
            key: (
                tuple(
                    item.replace(evaluate.GENERIC_SEGMENT_AID, evaluate.SEGMENT_AID)
                    for item in value.items
                )
                if isinstance(value, Numbered)
                else value
            )
            for key, value in prompt.items()
            if not key.startswith(("LOCALIZATION", "FIRST-STEP ALGORITHM"))
        }

    rgbd = {**settings, "generic_tools": True}
    rgbd_prompt = prompts.system(rgbd)
    assert task_sections(rgbd_prompt) == task_sections(baseline)
    assert prompts.user(rgbd) == original_user

    variables = {**rgbd, "geometry": True}
    actual = prompts.system(variables)
    user = prompts.user(variables)
    localization = next(
        value for key, value in actual.items() if key.startswith("LOCALIZATION")
    )
    assert "coarse localization" in localization and "samples_xyz" in localization
    for key in original_user.keys() - {"MODE"}:
        assert user[key] == original_user[key]
    if settings.get("policy_kind") == "wam":
        assert "geometry" in actual["WORKFLOW"]
        assert actual["OUTCOME"] == baseline["OUTCOME"]
        return

    heading = "FIRST-STEP ALGORITHM"
    if settings.get("memory_profile") != "local":
        heading += " — agentview = IDENTITY, wrist = GEOMETRY"
    assert baseline[heading] == rgbd_prompt[heading] == evaluate.PERCEPTION_ALGORITHM
    algorithm = actual["FIRST-STEP ALGORITHM"]
    assert "agentview" in algorithm and "wrist" not in algorithm
    assert "`geometry`" in user["MODE"]
    assert "Leave missing centers unresolved" in user["MODE"]
    assert "samples_xyz" in algorithm and "samples_xyz" in user["MODE"]
    rules = actual["RULES (NON-NEGOTIABLE)"]
    assert "`geometry`, `back_project` or `segment`" in rules
    assert "you get object positions ONLY by camera images, depth, and" not in rules
    workflow = " ".join(" ".join(actual["WORKFLOW"].items).split())
    assert "`back_project` for median xyz" not in workflow
    assert "localization table" in workflow
    assert "wrist-refined target before pi0_pick" not in next(
        value for key, value in actual.items() if key.startswith("PROVEN LEVERS")
    )
    for key in baseline:
        if key.startswith(("ROLE", "MEMORY", "YOUR GOAL", "OUTPUT DISCIPLINE")):
            assert actual[key] == baseline[key]


def test_toolkit_modes_construct_with_fake_primitives(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_single_arm_primitives: type[Any],
) -> None:
    fake_single_arm_primitives = fake_single_arm_primitives.for_robot(
        toolkit.libero_tools.LiberoPrimitives
    )
    dumped: list[Any] = []
    monkeypatch.setattr(
        templates, "default_variables", lambda: {"output_dir": "/offline/output"}
    )
    monkeypatch.setattr(
        toolkit.libero_tools,
        "LiberoPrimitives",
        fake_single_arm_primitives,
    )
    monkeypatch.setattr(
        toolkit.libero_tools,
        "dump_state",
        lambda primitives, state, log: dumped.append(primitives) or _record(),
    )

    evaluation = toolkit.LiberoToolkit(
        runtime_kwargs={"env": object()},
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(tmp_path / "evaluation-memory"),
        mode="evaluation",
        state_output_dir=tmp_path / "evaluation",
    )
    exploration = toolkit.LiberoToolkit(
        runtime_kwargs={"env": object()},
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(
            tmp_path / "exploration-memory",
            memory_access="inbox_write",
            inbox_cell_tag="offline-cell",
        ),
        mode="exploration",
        attempts_per_session=3,
        state_output_dir=tmp_path / "exploration",
    )

    assert _tool_names(evaluation) == EVALUATION_TOOLS
    assert _tool_names(exploration) == EVALUATION_TOOLS | {"reset"}
    assert _readonly_names(evaluation) == COMMON_TOOLS | {
        "view_env_state",
        "view_camera_meta",
        "back_project",
        "segment",
    }
    assert _readonly_names(exploration) == _readonly_names(evaluation)
    assert len(dumped) == 2
    assert all(
        instance.reset_calls == 1 for instance in fake_single_arm_primitives.instances
    )
    assert all(
        instance.recording_started for instance in fake_single_arm_primitives.instances
    )
    assert all(
        callable(instance.kwargs["check_cancelled"])
        for instance in fake_single_arm_primitives.instances
    )

    refused = exploration.execute_tool(
        "finish", {"status": "failure", "summary": "first attempt"}
    )
    assert refused.data["error"] == "finish refused"
    assert refused.data.get("_finish", False) is False

    exploration.get_env_state = lambda *, command, result, elapsed_s: ToolResult(
        data=dict(result)
    )
    assert (
        exploration.execute_tool("reset", {"reason": "new approach"}).data["attempt"]
        == 2
    )
    assert (
        exploration.execute_tool("reset", {"reason": "third approach"}).data["attempt"]
        == 3
    )
    allowed = exploration.execute_tool(
        "finish", {"status": "failure", "summary": "budget spent"}
    )
    assert allowed.data.get("_finish", False) is True


def test_parallel_segments_preserve_artifacts_without_capturing_state(
    tmp_path, monkeypatch
):
    state = EnvState(tmp_path / "state")
    with state.record_step(state={}):
        state.save("agentview.png", np.zeros((4, 4, 3), dtype=np.uint8))
        state.save("agentview_world.npz", np.ones((4, 4, 3)))
    calls = 4
    inference_barrier = threading.Barrier(calls, timeout=5)
    projection_barrier = threading.Barrier(calls, timeout=5)
    project = toolkit.libero_tools._mask_to_world

    def project_together(mask, world_map):
        projection_barrier.wait()
        return project(mask, world_map)

    monkeypatch.setattr(toolkit.libero_tools, "_mask_to_world", project_together)

    def segment(image, **kwargs):
        # Every inference must enter before any returns: segment stays readonly.
        inference_barrier.wait()
        return Sam3Result(found=True, mask=np.ones((4, 4), dtype=bool), score=1.0)

    primitives = toolkit.libero_tools.LiberoPrimitives(
        env=object(),
        model=object(),
        sam3_client=SimpleNamespace(segment=segment),
        check_cancelled=lambda: None,
    )
    robot_toolkit = Toolkit(
        dashboard_events=NullDashboardEventSink(),
        state=state,
        memory=MemoryManager(tmp_path / "memory"),
    )
    definition = primitives.segment
    robot_toolkit.add_tool(definition.with_handler(partial(definition, state=state)))
    captured = []

    def capture(**kwargs):
        captured.append(kwargs)
        return ToolResult(data={})

    robot_toolkit.get_env_state = capture
    prompts = [f"object {index}" for index in range(calls)]
    with ThreadPoolExecutor(max_workers=calls) as pool:
        results = list(
            pool.map(
                lambda prompt: robot_toolkit.execute_tool(
                    "segment", {"prompt": prompt}
                ),
                prompts,
            )
        )

    assert captured == []
    assert len(state.records()) == 1
    assert all(not result.is_error for result in results)
    assert {result.data["segment_artifact"] for result in results} == {
        f"segment_{index:02d}.json" for index in range(calls)
    }
    for prompt, result in zip(prompts, results):
        artifact = state.load(result.data["segment_artifact"], step=0)
        assert artifact["prompt"] == prompt
        assert artifact["source_step"] == 0
        assert result.images == [
            state.load_bytes(result.data["overlay_artifact"], step=0)
        ]

    # A failed JSON write must not release an index for another call to reuse.
    monkeypatch.setattr(toolkit.libero_tools, "_mask_to_world", project)
    primitives._sam3_client.segment = lambda *args, **kwargs: Sam3Result(
        found=True, mask=np.ones((4, 4), dtype=bool), score=1.0
    )
    save = state.save

    def fail_segment_save(name, value, **kwargs):
        if name == f"segment_{calls:02d}.json":
            return None
        return save(name, value, **kwargs)

    monkeypatch.setattr(state, "save", fail_segment_save)
    failed = robot_toolkit.execute_tool("segment", {"prompt": "failed"})
    assert failed.data["code"] == "segment_artifact_save_failed"
    result = robot_toolkit.execute_tool("segment", {"prompt": "next"})
    assert result.data["segment_artifact"] == f"segment_{calls + 1:02d}.json"
    assert state.load(result.data["segment_artifact"], step=0)["prompt"] == "next"
