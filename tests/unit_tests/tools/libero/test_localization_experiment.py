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

"""Offline checklist execution, immutable inputs and independent error accounting."""

import copy
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rpent.session import EnvState
from rpent.tools import ToolResult
from tools.libero import localization_experiment as experiment
from tools.libero.localization_checklist import (
    build_manifest,
    frame_id,
    load_manifest,
    manifest_digest,
    select_cases,
)
from tools.libero.localization_evaluation import (
    evaluate,
    truth_template,
    validate_truth,
)


def labelled_truth(manifest):
    truth = truth_template(manifest)
    for row in truth["annotations"]:
        row.update(
            evaluable=True,
            world_xyz=[0.1, 0.2, 0.8],
            annotation_source="independent analytic test labels",
        )
    return truth


def observation(root, manifest, case):
    path = root / case["case_id"]
    state = EnvState(path / "episode")
    with state.record_step(
        state={"object_truth": [9, 8, 7]}, extras={"secret_truth": "not an agent input"}
    ):
        state.save("agentview_high.png", np.zeros((12, 12, 3), dtype=np.uint8))
        state.save("agentview_world_high.npz", np.zeros((12, 12, 3), dtype=np.float32))
        state.save("agentview_depth_high.npz", np.ones((12, 12), dtype=np.float32))
        state.save(
            "agentview_metadata_high.json",
            {
                "intrinsic_K": np.eye(3).tolist(),
                "extrinsic_cam2world": np.eye(4).tolist(),
                "secret_truth": [9, 8, 7],
            },
        )
        state.save("truth.json", {"point": [9, 8, 7]})
    experiment._write_json(
        path / "observation.json",
        {
            "manifest_sha256": manifest_digest(manifest),
            "case_id": case["case_id"],
            "frame_id": frame_id(case["case_id"]),
            "step": 0,
            "camera": "agentview",
            "units": "m",
            "artifact_sha256": {
                name: experiment._hash(state.artifact_path(name, step=0))
                for name in state.get(0).artifacts
            },
        },
    )
    return path, state


def test_manifest_expands_every_markdown_row_and_reference():
    root = Path(__file__).resolve().parents[4]
    manifest = load_manifest(root / "test-cases.json")
    assert len(manifest["cases"]) == 80
    assert sum(len(case["requests"]) for case in manifest["cases"]) == 160
    assert Counter(
        request["reference"]
        for case in manifest["cases"]
        for request in case["requests"]
    ) == {
        "plate": 40,
        "ramekin": 20,
        "cheese": 20,
        "soup": 20,
        "black_bowl": 30,
        "wine_bottle": 10,
        "gray_white_mug": 5,
        "yellow_white_mug": 10,
        "microwave": 5,
    }
    markdown = (root / "选点清单.md").read_text()
    rows = [
        line.split("|")
        for line in markdown.splitlines()
        if line.startswith("| `libero_")
    ]
    assert len(rows) == 16
    for row in rows:
        suite, task = row[1].strip().strip("`"), int(row[2])
        cases = [
            case
            for case in manifest["cases"]
            if case["suite"] == suite and case["task"] == task
        ]
        assert [case["seed"] for case in cases] == [1, 2, 3, 4, 5]
        assert [
            manifest["references"][request["reference"]]["label"]
            for request in cases[0]["requests"]
        ] == [row[4].strip(), row[5].strip()]


def test_manifest_drift_and_unknown_pilot_cases_are_rejected(tmp_path):
    manifest = build_manifest()
    assert len(select_cases(manifest, limit=2)) == 2
    assert (
        select_cases(manifest, [manifest["cases"][-1]["case_id"]])
        == manifest["cases"][-1:]
    )
    with pytest.raises(ValueError, match="positive"):
        select_cases(manifest, limit=0)
    with pytest.raises(ValueError, match="known"):
        select_cases(manifest, ["other_seed"])
    manifest["cases"][0]["seed"] = 0
    experiment._write_json(tmp_path / "manifest.json", manifest)
    with pytest.raises(ValueError, match="fixed"):
        load_manifest(tmp_path / "manifest.json")


def test_sensor_replay_strips_truth_and_never_modifies_source(tmp_path):
    manifest = build_manifest()
    case = manifest["cases"][0]
    source, original = observation(tmp_path / "input", manifest, case)
    before = {
        str(path): path.read_bytes() for path in source.rglob("*") if path.is_file()
    }
    state, hashes = experiment.copy_observation(
        source, tmp_path / "inference", manifest, case
    )
    assert state.get(0).state == {} and state.get(0).extras == {}
    assert not state.exists("truth.json", step=0)
    assert "secret_truth" not in state.load("agentview_metadata_high.json", step=0)
    assert original.load("truth.json", step=0)["point"] == [9, 8, 7]
    assert len(hashes) == 4
    assert before == {
        str(path): path.read_bytes() for path in source.rglob("*") if path.is_file()
    }
    original.save("agentview_high.png", np.ones((12, 12, 3), dtype=np.uint8), step=0)
    with pytest.raises(ValueError, match="changed"):
        experiment.copy_observation(source, tmp_path / "rejected", manifest, case)


@pytest.mark.parametrize("permissive", [False, True])
def test_batch_runs_exactly_160_fixed_requests_and_never_uses_surface_fallback(
    tmp_path, monkeypatch, permissive
):
    from tools.libero import geometry_agent

    manifest = build_manifest()
    cases = manifest["cases"]
    args = SimpleNamespace(
        inputs=tmp_path / "inputs",
        output=tmp_path / "predictions",
        geometry_model="unused:test",
        geometry_base_url=None,
        permissive=permissive,
    )
    args.output.mkdir()
    for case in cases:
        observation(args.inputs, manifest, case)
    calls = []

    class Agent:
        def __init__(self, **kwargs):
            assert kwargs["state"].get(0).state == {}
            assert kwargs["permissive"] is permissive

        def localize(self, query, *, camera, step, required_geometry):
            calls.append((query, camera, step, required_geometry))
            field = required_geometry[0]
            value = None if field == "rectangle_center" else [0.1, 0.2, 0.8]
            return ToolResult(
                data={
                    "status": "need_observation" if value is None else "usable",
                    "world_xyz": [9, 8, 7],
                    "geometry": {field: {"value": value}},
                }
            )

    monkeypatch.setattr(geometry_agent, "GeometryAgent", Agent)
    predictions = experiment.infer_cases(args, manifest, cases, object())
    assert len(predictions) == len(calls) == 160
    assert len({row["request_id"] for row in predictions}) == 160
    assert all(camera == "agentview" and step == 0 for _, camera, step, _ in calls)
    assert Counter(call[3][0] for call in calls) == {
        "plate_center": 40,
        "inner_rim_center": 65,
        "rectangle_center": 25,
        "lid_center": 20,
        "body_segment_center": 10,
    }
    assert sum(row["world_xyz"] is None for row in predictions) == 25
    assert [9, 8, 7] not in [row["world_xyz"] for row in predictions]
    assert len((args.output / "results.jsonl").read_text().splitlines()) == 160


def test_missing_observation_records_both_target_errors_instead_of_skipping(tmp_path):
    manifest = build_manifest()
    case = manifest["cases"][0]
    args = SimpleNamespace(
        inputs=tmp_path / "missing",
        output=tmp_path,
        geometry_model="unused:test",
        geometry_base_url=None,
    )
    predictions = experiment.infer_cases(args, manifest, [case], object())
    assert len(predictions) == 2 and all(
        row["status"] == "error" for row in predictions
    )


def test_collection_selects_env_only_and_cleans_up_on_capture_failure(
    tmp_path, monkeypatch
):
    from robots.libero import tools

    manifest = build_manifest()
    cases = manifest["cases"][:2]
    calls, stopped = [], []

    def start(args, case, output, components):
        calls.append(components)
        return [SimpleNamespace(stop=lambda: stopped.append(case["case_id"]))], {
            "env": SimpleNamespace(reset=lambda: ({"states": np.zeros(8)}, {}))
        }

    monkeypatch.setattr(experiment, "_start_runtime", start)
    monkeypatch.setattr(
        tools,
        "dump_state",
        lambda *args: (_ for _ in ()).throw(RuntimeError("capture failed")),
    )
    args = SimpleNamespace(output=tmp_path / "collect", env_endpoint=None)
    summary = experiment.collect(args, manifest, cases)
    assert calls == [{"env"}, {"env"}] and len(stopped) == 2
    assert summary == {"selected_cases": 2, "captured": 0, "failed": 2}
    with pytest.raises(ValueError, match="pinned"):
        experiment.collect(
            SimpleNamespace(output=tmp_path / "other", env_endpoint="http://borrowed"),
            manifest,
            cases,
        )


def test_collection_preserves_native_rgb_depth_alignment_and_float32_world(
    tmp_path, monkeypatch
):
    manifest = build_manifest()
    case = manifest["cases"][0]
    calls = []

    class Env:
        terminated = truncated = False

        def reset(self):
            return {
                "states": np.zeros(8),
                "main_images": np.zeros((256, 256, 3), dtype=np.uint8),
            }, {}

        def raw_obs(self):
            return {
                "robot0_eef_pos": [0, 0, 1],
                "robot0_eef_quat": [0, 0, 0, 1],
                "robot0_gripper_qpos": [0, 0],
                "agentview_image": np.zeros((256, 256, 3), dtype=np.uint8),
                "agentview_depth": np.full((256, 256), 0.5),
            }

        def get_task_language(self):
            return "no policy actions in this capture"

        def get_camera_meta(self, camera_name, height, width):
            return {
                "intrinsic_K": [
                    [height, 0, width / 2],
                    [0, height, height / 2],
                    [0, 0, 1],
                ],
                "extrinsic_cam2world": np.eye(4).tolist(),
                "depth_near": 0.1,
                "depth_far": 10.0,
            }

        def render_camera(self, camera_name, height, width, depth):
            calls.append((camera_name, height, width, depth))
            raw_depth = np.broadcast_to(
                np.linspace(0.1, 0.9, height, dtype=np.float32)[:, None],
                (height, width),
            )
            rgb = np.broadcast_to(
                (np.arange(height) % 256).astype(np.uint8)[:, None, None],
                (height, width, 3),
            ).copy()
            return rgb, raw_depth

    monkeypatch.setattr(
        experiment, "_start_runtime", lambda *args: ([], {"env": Env()})
    )
    output = tmp_path / "captured"
    summary = experiment.collect(
        SimpleNamespace(output=output, env_endpoint=None), manifest, [case]
    )
    assert summary["captured"] == 1
    state = EnvState(output / case["case_id"] / "episode")
    depth = state.load("agentview_depth_high.npz", step=0)
    world = state.load("agentview_world_high.npz", step=0)
    rgb = state.load("agentview_high.png", step=0)
    assert depth.shape == (1024, 1024) and world.dtype == np.float32
    assert depth[0, 0] == pytest.approx(0.1 / (1 - 0.9 * 0.99), rel=1e-5)
    assert rgb[0, 0, 0] == 255
    np.testing.assert_allclose(world[:, :, 2], depth, atol=1e-6)
    assert all(
        height == width == 1024 and depth is True for _, height, width, depth in calls
    )


def test_inference_starts_sam_only_and_stops_owned_service(tmp_path, monkeypatch):
    manifest = build_manifest()
    cases = manifest["cases"][:1]
    calls = []

    def start(args, case, output, components):
        calls.append(components)
        return [SimpleNamespace(stop=lambda: calls.append("stopped"))], {
            "sam3_client": object()
        }

    monkeypatch.setattr(experiment, "_start_runtime", start)
    monkeypatch.setattr(
        experiment,
        "infer_cases",
        lambda *args: (_ for _ in ()).throw(RuntimeError("model failed")),
    )
    args = SimpleNamespace(
        output=tmp_path / "run", inputs=tmp_path, geometry_model="codex"
    )
    with pytest.raises(RuntimeError, match="model failed"):
        experiment.infer(args, manifest, cases)
    assert calls == [{"sam3"}, "stopped"]


def test_evaluation_counts_failures_and_unrun_in_denominator_but_never_as_zero_error():
    manifest = build_manifest()
    truth = labelled_truth(manifest)
    cases = manifest["cases"]
    predictions = []
    for case in cases:
        for request in case["requests"]:
            row = experiment._prediction_base(manifest, case, request)
            row.update(status="usable", world_xyz=[0.1, 0.2, 0.8])
            predictions.append(row)
    predictions[0].update(status="need_observation", world_xyz=None)
    predictions[1]["world_xyz"] = [0.12, 0.2, 0.8]
    predictions.pop()
    truth["annotations"][2].update(
        evaluable=False,
        world_xyz=None,
        not_evaluable_reason="independently labelled occluded",
    )
    metrics = evaluate(manifest, truth, predictions)["overall"]
    assert metrics["planned"] == 160 and metrics["evaluable"] == 159
    assert metrics["unrun"] == 1 and metrics["refused_or_failed_on_evaluable"] == 1
    assert metrics["returned_on_evaluable"] == 157
    assert metrics["returned_on_not_evaluable"] == 1
    assert metrics["success_within_tolerance"] == 156
    assert metrics["success_rate_on_all_evaluable"] == pytest.approx(156 / 159)
    assert metrics["conditional_error_m_on_returned_only"]["mean"] == pytest.approx(
        0.02 / 157
    )


@pytest.mark.parametrize(
    "corruption",
    ["unlabelled", "missing", "units", "duplicate", "definition", "frame", "nan"],
)
def test_invalid_truth_never_produces_an_accuracy_score(corruption):
    manifest = build_manifest()
    truth = labelled_truth(manifest)
    first = truth["annotations"][0]
    if corruption == "unlabelled":
        first["evaluable"] = None
    elif corruption == "missing":
        truth["annotations"].pop()
    elif corruption == "units":
        truth["units"] = "mm"
    elif corruption == "duplicate":
        truth["annotations"].append(copy.deepcopy(first))
    elif corruption == "definition":
        first["definition"] = "mask centroid"
    elif corruption == "frame":
        first["frame_id"] = "different-world"
    elif corruption == "nan":
        first["world_xyz"] = [float("nan"), 0, 0]
    with pytest.raises(ValueError):
        evaluate(manifest, truth, [])


def test_model_local_and_bottle_profile_truth_are_transformed_only_by_evaluator():
    manifest = build_manifest()
    truth = labelled_truth(manifest)
    bottle = next(
        row for row in truth["annotations"] if row["reference"] == "wine_bottle"
    )
    pose = np.eye(4)
    pose[:3, 3] = [0.1, 0.2, 0.8]
    bottle.update(
        world_xyz=None,
        barrel_profile={
            "axis": [1, 0, 0],
            "axis_origin": [0, 0, 0],
            "bounds_m": [-0.1, 0.1],
        },
        model_to_world=pose.tolist(),
    )
    first = truth["annotations"][0]
    first.update(
        world_xyz=None, model_local_xyz=[0, 0, 0], model_to_world=pose.tolist()
    )
    resolved = validate_truth(manifest, truth)
    assert resolved[bottle["request_id"]]["world_xyz"] == [0.1, 0.2, 0.8]
    assert resolved[first["request_id"]]["world_xyz"] == [0.1, 0.2, 0.8]


def test_cli_manifest_and_unlabelled_template_need_no_models(tmp_path):
    manifest = tmp_path / "manifest.json"
    assert experiment.main(["write-manifest", "--manifest", str(manifest)]) == 0
    assert experiment.main(["validate-manifest", "--manifest", str(manifest)]) == 0
    truth = tmp_path / "truth.json"
    assert (
        experiment.main(
            ["truth-template", "--manifest", str(manifest), "--output", str(truth)]
        )
        == 0
    )
    assert len(json.loads(truth.read_text())["annotations"]) == 160
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text("")
    assert (
        experiment.main(
            [
                "evaluate",
                "--manifest",
                str(manifest),
                "--truth",
                str(truth),
                "--predictions",
                str(predictions),
                "--output",
                str(tmp_path / "metrics"),
            ]
        )
        == 1
    )
    assert not (tmp_path / "metrics" / "metrics.json").exists()
