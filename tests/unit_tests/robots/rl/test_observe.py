# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.
"""Episode observe graphs: source identity, atomic gates, dependencies and concurrency."""

from __future__ import annotations

import copy
import json
import threading

import pytest
from pydantic_ai.models.function import DeltaToolCall, FunctionModel

from robots.rl import tools
from robots.rl.prompt_bundle import PROMPT_VARIANTS, system_prompt, user_prompt
from robots.rl.workflow import WorkflowCapability
from rpent.dashboard.events import NullDashboardEventSink
from rpent.planner.api_loop import ApiAgentLoop
from rpent.utils import templates
from tests.unit_tests.robots.rl.test_episode_image_actions import make
from tests.unit_tests.robots.rl.test_libero_rl_contracts import FakeSam3, bar_mask


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    monkeypatch.setattr(tools, "IMAGE_SIZE", 32)
    monkeypatch.setattr(
        templates, "default_variables", lambda: {"output_dir": str(tmp_path)}
    )


def toolkit(tmp_path, **kwargs):
    return make(tmp_path, workflow="episode", perception_mode="batch", **kwargs)[0]


def call(tk, *items):
    return tk.execute_tool("observe", {"calls": list(items)})


def results(value):
    assert not value.is_error, value.to_text()
    return {item["id"]: item["result"] for item in value.data["results"]}


def test_tool_sets_and_initial_handles(tmp_path):
    tk = toolkit(tmp_path)
    assert {d.name for d in tk.list_tools()} == {
        "observe",
        "move_to",
        "set_gripper",
        "finish",
    }
    initial = tk.initial_observation()
    assert not initial.images
    handles = initial.data["state"]["image_sources"]
    viewed = results(call(tk, {"id": "v", "op": "view_env"}))["v"]
    assert {meta["camera"]: meta["image_id"] for meta in viewed["images"]} == handles
    batch = toolkit(tmp_path / "batch", action_mode="batch")
    assert {d.name for d in batch.list_tools()} == {
        "observe",
        "execute_actions",
        "finish",
    }


def test_views_deduplicate_images_and_keep_stable_ids(tmp_path):
    tk = toolkit(tmp_path)
    value = call(
        tk,
        {"id": "a", "op": "view_env"},
        {"id": "b", "op": "view_env", "camera": "wrist"},
    )
    assert len(value.images) == 2
    data = results(value)
    assert data["b"]["images"][0]["image_index"] == 1
    assert data["a"]["images"][1]["image_id"] == data["b"]["images"][0]["image_id"]
    assert (
        results(call(tk, {"id": "c", "op": "view_env"}))["c"]["images"]
        == data["a"]["images"]
    )


def test_forward_refs_parallel_segments_and_exact_mask_binding(tmp_path):
    tk = toolkit(tmp_path)
    barrier = threading.Barrier(2, timeout=5)

    class ConcurrentSam(FakeSam3):
        def segment(self, *args, **kwargs):
            barrier.wait()
            return super().segment(*args, **kwargs)

    tk._primitives._sam3 = ConcurrentSam(bar_mask())
    value = call(
        tk,
        {"id": "xyz", "op": "back_project", "source": "@cup"},
        {"id": "cup", "op": "segment", "source": "@v.agentview", "prompt": "cup"},
        {"id": "plate", "op": "segment", "source": "@v.wrist", "prompt": "plate"},
        {"id": "v", "op": "view_env"},
    )
    data = results(value)
    assert data["cup"]["mask_id"] != data["plate"]["mask_id"]
    assert data["xyz"]["image_id"] == data["cup"]["image_id"]
    assert data["xyz"]["world_xyz"] == data["cup"]["world_xyz"]
    assert [r["id"] for r in value.data["results"]] == ["xyz", "cup", "plate", "v"]
    assert tk._primitives.env.actions == []
    audit = json.loads((tk.output_dir / "observe/0001.json").read_text())
    nodes = audit["nodes"]
    assert max(nodes["cup"]["started_at"], nodes["plate"]["started_at"]) < min(
        nodes["cup"]["finished_at"], nodes["plate"]["finished_at"]
    )
    assert nodes["xyz"]["started_at"] >= nodes["cup"]["finished_at"]


def test_ready_dependent_does_not_wait_for_unrelated_slow_segment(tmp_path):
    tk = toolkit(tmp_path, observe_concurrency=2)
    projected = threading.Event()

    class PipelineSam(FakeSam3):
        def segment(self, *args, text_prompt=None, **kwargs):
            if text_prompt == "slow":
                assert projected.wait(5), (
                    "projection was blocked behind unrelated SAM work"
                )
            return super().segment(*args, text_prompt=text_prompt, **kwargs)

    tk._primitives._sam3 = PipelineSam(bar_mask())
    original = tk._primitives.back_project
    # Replace the instance tool with a small callable only for primitive execution.

    def project(**kwargs):
        result = original(**kwargs)
        projected.set()
        return result

    tk._primitives.back_project = project
    image = tk.observer.publish_current()["agentview"]
    value = call(
        tk,
        {"id": "fast", "op": "segment", "source": image, "prompt": "fast"},
        {"id": "slow", "op": "segment", "source": image, "prompt": "slow"},
        {"id": "p", "op": "back_project", "source": "@fast"},
    )
    assert isinstance(results(value)["p"], dict)


def test_issued_images_and_masks_survive_motion_and_local_aliases_do_not(tmp_path):
    tk = toolkit(tmp_path)
    old = tk.initial_observation().data["state"]["image_sources"]["agentview"]
    segmented = results(
        call(tk, {"id": "cup", "op": "segment", "source": old, "prompt": "cup"})
    )["cup"]
    mask = segmented["mask_id"]
    moved = tk.execute_tool("move_to", {"xyz": [0.2, -0.1, 1.0]})
    assert moved.data["image_sources"]["agentview"] != old
    assert tk.state.latest_step > 0
    data = results(
        call(
            tk,
            {"id": "old", "op": "segment", "source": old, "prompt": "cup"},
            {"id": "mask", "op": "back_project", "source": mask},
        )
    )
    assert data["old"]["step"] == data["mask"]["step"] == 0
    assert data["old"]["image_id"] == data["mask"]["image_id"] == old
    assert call(tk, {"id": "x", "op": "back_project", "source": "@cup"}).is_error
    other = toolkit(tmp_path / "other")
    assert call(
        other, {"id": "x", "op": "segment", "source": old, "prompt": "cup"}
    ).is_error


def test_batch_actions_publish_the_final_and_intermediate_image_handles(tmp_path):
    tk = toolkit(tmp_path, action_mode="batch")
    value = tk.execute_tool(
        "execute_actions",
        {
            "actions": [
                {"action": "move_to", "xyz": [0.2, -0.1, 1.0]},
                {"action": "move_to", "xyz": [0.25, -0.1, 1.0]},
            ]
        },
    )
    assert value.data["results"][0]["image_sources"] != value.data["image_sources"]
    assert value.data["results"][-1]["image_sources"] == value.data["image_sources"]
    assert all(
        handle in tk.observer.images
        for row in value.data["results"]
        for handle in row["image_sources"].values()
    )


@pytest.mark.parametrize(
    "bad",
    [
        [],
        [{"id": "v", "op": "view_env", "source": "anything"}],
        [{"id": "a", "op": "view_env"}, {"id": "a", "op": "view_env"}],
        [{"id": "v", "op": "view_env", "camera": "unknown"}],
        [
            {"id": "s", "op": "segment", "source": "@v", "prompt": "cup"},
            {"id": "v", "op": "view_env"},
        ],
        [
            {"id": "s", "op": "segment", "source": "@v.wrist", "prompt": "cup"},
            {"id": "v", "op": "view_env", "camera": "agentview"},
        ],
        [{"id": "s", "op": "segment", "source": "@missing", "prompt": "cup"}],
        [{"id": "s", "op": "segment", "source": "@s", "prompt": "cup"}],
        [
            {"id": "v", "op": "view_env"},
            {
                "id": "p",
                "op": "back_project",
                "source": "@v.agentview",
                "pixel": [1, 1],
            },
            {"id": "s", "op": "segment", "source": "@p", "prompt": "cup"},
        ],
        [
            {"id": "v", "op": "view_env"},
            {"id": "s", "op": "segment", "source": "@v.agentview", "prompt": "cup"},
            {"id": "p", "op": "back_project", "source": "@s", "pixel": [1, 1]},
        ],
        [
            {"id": "a", "op": "segment", "source": "@b", "prompt": "cup"},
            {"id": "b", "op": "segment", "source": "@a", "prompt": "cup"},
        ],
        [
            {
                "id": "a",
                "op": "segment",
                "source": "@v.agentview",
                "prompt": "cup",
                "point": [1, 2],
            },
            {"id": "v", "op": "view_env"},
        ],
        [
            {"id": "a", "op": "segment", "source": "@v.agentview", "prompt": " "},
            {"id": "v", "op": "view_env"},
        ],
        [
            {
                "id": "a",
                "op": "segment",
                "source": "@v.agentview",
                "prompt": "cup",
                "step": 0,
            },
            {"id": "v", "op": "view_env"},
        ],
        [
            {"id": "p", "op": "back_project", "source": "@v.agentview"},
            {"id": "v", "op": "view_env"},
        ],
        [
            {
                "id": "p",
                "op": "back_project",
                "source": "@v.agentview",
                "pixel": [1024, 1],
            },
            {"id": "v", "op": "view_env"},
        ],
        [
            {
                "id": "p",
                "op": "back_project",
                "source": "@v.agentview",
                "box": [5, 2, 3, 4],
            },
            {"id": "v", "op": "view_env"},
        ],
        [
            {
                "id": "p",
                "op": "back_project",
                "source": "@v.agentview",
                "pixel": [1, 2],
                "z_range_m": [0, 1],
            },
            {"id": "v", "op": "view_env"},
        ],
        [
            {
                "id": "p",
                "op": "back_project",
                "source": "@v.agentview",
                "box": [0, 0, 4, 4],
                "z_range_m": [float("nan"), 1],
            },
            {"id": "v", "op": "view_env"},
        ],
    ],
)
def test_invalid_whole_graph_runs_nothing(tmp_path, bad):
    tk = toolkit(tmp_path)
    count = 0

    class ForbiddenSam(FakeSam3):
        def segment(self, *a, **kw):
            nonlocal count
            count += 1
            return super().segment(*a, **kw)

    tk._primitives._sam3 = ForbiddenSam(bar_mask())
    value = call(tk, *bad)
    assert value.is_error and count == 0
    assert not list((tk.output_dir / "observe").glob("[0-9]*.json"))
    assert tk._primitives.env.actions == []


def test_failed_sources_skip_only_dependents_and_preserve_other_results(tmp_path):
    from rpent.robots.components.sam3_client import Sam3Result

    tk = toolkit(tmp_path)

    class MissingSam(FakeSam3):
        def segment(self, *args, text_prompt=None, **kwargs):
            if text_prompt == "missing":
                return Sam3Result(
                    found=False, score=None, box=None, mask=None, mask_shape=None
                )
            if text_prompt == "broken":
                raise RuntimeError("service unavailable")
            return super().segment(*args, text_prompt=text_prompt, **kwargs)

    tk._primitives._sam3 = MissingSam(bar_mask())
    image = tk.observer.publish_current()["agentview"]
    value = call(
        tk,
        {"id": "missing", "op": "segment", "source": image, "prompt": "missing"},
        {"id": "broken", "op": "segment", "source": image, "prompt": "broken"},
        {"id": "ok", "op": "segment", "source": image, "prompt": "cup"},
        {"id": "dependent", "op": "back_project", "source": "@missing"},
        {
            "id": "independent",
            "op": "back_project",
            "source": image,
            "pixel": [500, 400],
        },
    )
    data = results(value)
    assert "No target" in data["missing"] and "segmentation_failed" in data["broken"]
    assert "Skipped" in data["dependent"]
    assert isinstance(data["ok"], dict) and isinstance(data["independent"], dict)
    assert "score" not in data["ok"] and "found" not in data["ok"]


@pytest.mark.parametrize("size", [1024, 512, 256])
def test_image_coordinate_mapping_uses_source_size_and_native_depth(tmp_path, size):
    tk = toolkit(tmp_path, image_size=size)
    seen = []

    class PointSam(FakeSam3):
        def segment(self, *args, point=None, **kwargs):
            seen.append(point)
            return super().segment(*args, point=point, **kwargs)

    tk._primitives._sam3 = PointSam(bar_mask())
    image_id = tk.observer.publish_current()["agentview"]
    data = results(
        call(
            tk,
            {"id": "s", "op": "segment", "source": image_id, "point": [size - 1, 0]},
            {
                "id": "p",
                "op": "back_project",
                "source": image_id,
                "pixel": [size - 1, 0],
            },
        )
    )
    assert seen == [[31, 0]]
    world = tk.state.load("agentview_world.npz", step=0)
    assert data["p"]["world_xyz"] == pytest.approx(world[31, 0], abs=0.0001)
    assert data["s"]["box"] == [size // 4, size // 8, 3 * size // 8, 7 * size // 8]
    assert call(
        tk, {"id": "p", "op": "back_project", "source": image_id, "pixel": [size, 0]}
    ).is_error


def test_model_retries_invalid_graph_before_any_tool_runs(tmp_path):
    tk = toolkit(tmp_path)
    requests = []
    script = [
        [
            {"id": "v", "op": "view_env"},
            {"id": "bad", "op": "segment", "source": "@missing", "prompt": "cup"},
        ],
        [
            {"id": "v", "op": "view_env"},
            {"id": "cup", "op": "segment", "source": "@v.agentview", "prompt": "cup"},
        ],
    ]

    async def stream(messages, info):
        requests.append(copy.deepcopy(messages))
        if len(requests) < 3:
            yield {
                0: DeltaToolCall(
                    name="observe",
                    json_args=json.dumps({"calls": script[len(requests) - 1]}),
                )
            }
        else:
            yield {0: DeltaToolCall(name="finish", json_args="{}")}

    planner = ApiAgentLoop(
        model=FunctionModel(stream_function=stream),
        dashboard_events=NullDashboardEventSink(),
        timeout_s=10,
        compaction=None,
        capabilities_factory=lambda toolkit, system, user, turns: [
            WorkflowCapability(toolkit, system, user, turns, timeout_s=10)
        ],
    )
    result = planner.solve(
        system_prompt="Use tools.", user_message="Task.", toolkit=tk, max_turns=3
    )
    assert result.error is None and result.stats["turns_used"] == 3
    assert tk.workflow_stats["invalid_tool_responses"] == 1
    assert tk.workflow_stats["observe_calls"] == 1
    assert "No tool executed" in str(requests[1])


@pytest.mark.parametrize("variant", PROMPT_VARIANTS)
def test_prompts_explain_sources_tools_and_visibility(tmp_path, variant):
    variables = {"workflow": "episode", "prompt": variant, "perception_mode": "batch"}
    system, user = system_prompt(variables), user_prompt(variables)
    assert "suboperations" in system and "@v.agentview" in system
    assert "intermediate" in system and "mask_id" in system
    assert "not standalone tools" in system
    assert "Use view_env to obtain" not in system  # no old standalone workflow rule
    assert "image_sources" in user and "not displayed pictures" in user


@pytest.mark.parametrize(
    "workflow", ["episode_image", "zap_turn", "zap_episode", "zap_subtask"]
)
def test_observe_cannot_be_enabled_for_other_workflows(tmp_path, workflow):
    with pytest.raises(ValueError, match="episode"):
        make(tmp_path, workflow=workflow, perception_mode="batch")
    with pytest.raises(ValueError, match="episode"):
        system_prompt({"workflow": workflow, "perception_mode": "batch"})


def test_schema_bounds_and_list_limit_are_shared_with_executor(tmp_path):
    tk = toolkit(tmp_path, max_observe_calls=2, image_size=256)
    tool = next(t for t in tk.list_tools() if t.name == "observe")
    schema = tool.input_schema
    assert schema["properties"]["calls"]["maxItems"] == 2
    assert schema["$defs"]["ViewCall"]["properties"]["camera"]["enum"] == [
        "agentview",
        "wrist",
        "all",
    ]
    for model, field, bound in [
        ("SegmentCall", "point", 255),
        ("ProjectCall", "pixel", 255),
        ("ProjectCall", "box", 256),
    ]:
        prop = schema["$defs"][model]["properties"][field]
        assert prop["items"] == {"minimum": 0, "maximum": bound, "type": "integer"}
    image = tk.observer.publish_current()["agentview"]
    for bad in [
        {"id": "a", "op": "segment", "source": image, "point": [-1, 1]},
        {"id": "a", "op": "segment", "source": image, "point": [1.5, 1]},
        {"id": "a", "op": "segment", "source": image, "point": [True, 1]},
    ]:
        assert call(tk, bad).is_error
    assert call(tk, *[{"id": f"v{i}", "op": "view_env"} for i in range(3)]).is_error


def test_serial_concurrency_cap(tmp_path):
    tk = toolkit(tmp_path, observe_concurrency=1)
    value = call(
        tk,
        {"id": "v", "op": "view_env"},
        {"id": "a", "op": "segment", "source": "@v.agentview", "prompt": "cup"},
        {"id": "b", "op": "segment", "source": "@v.wrist", "prompt": "cup"},
    )
    assert all(isinstance(row["result"], dict) for row in value.data["results"])
    audit = json.loads((tk.output_dir / "observe/0001.json").read_text())
    a, b = audit["nodes"]["a"], audit["nodes"]["b"]
    assert b["started_at"] >= a["finished_at"]


@pytest.mark.parametrize(
    "environment,camera",
    [
        ("libero", "agentview"),
        ("robocasa", "agentview"),
        ("robotwin", "head"),
    ],
)
def test_observe_prompt_uses_the_simulators_actual_camera(environment, camera):
    text = system_prompt(
        {"environment": environment, "workflow": "episode", "perception_mode": "batch"}
    )
    assert f"@v.{camera}" in text
    if environment == "robotwin":
        assert "@v.agentview" not in text


def test_physical_action_cannot_interleave_with_observation(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    tk = toolkit(tmp_path)
    entered, release, stepped = threading.Event(), threading.Event(), threading.Event()

    class BlockingSam(FakeSam3):
        def segment(self, *args, **kwargs):
            entered.set()
            assert release.wait(5)
            return super().segment(*args, **kwargs)

    tk._primitives._sam3 = BlockingSam(bar_mask())
    original = tk._primitives.env.step

    def step(*args, **kwargs):
        stepped.set()
        return original(*args, **kwargs)

    tk._primitives.env.step = step
    image = tk.observer.publish_current()["agentview"]
    with ThreadPoolExecutor(max_workers=2) as workers:
        observed = workers.submit(
            call, tk, {"id": "cup", "op": "segment", "source": image, "prompt": "cup"}
        )
        assert entered.wait(5)
        moved = workers.submit(tk.execute_tool, "move_to", {"xyz": [0.2, -0.1, 1.0]})
        try:
            assert not stepped.wait(0.1), "a physical action entered a read observation"
        finally:
            release.set()
        assert not observed.result(timeout=5).is_error
        assert not moved.result(timeout=5).is_error
    assert stepped.is_set()
