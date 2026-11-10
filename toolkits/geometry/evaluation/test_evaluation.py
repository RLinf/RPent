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

"""Offline execution checks; these are not real LIBERO accuracy experiments."""

from __future__ import annotations

import csv
import io
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import imageio.v3 as imageio
import numpy as np
import pytest

from rpent.robots.components.molmo_client import MolmoResult
from rpent.robots.components.sam3_client import Sam3Result
from toolkits.geometry.evaluation import experiment
from toolkits.geometry.evaluation.checklist import (
    build_manifest,
    frame_id,
    manifest_digest,
    select_cases,
)
from toolkits.geometry.evaluation.evaluation_scene import REFERENCE_OBJECTS
from toolkits.geometry.evaluation.libero import capture_case
from toolkits.geometry.evaluation.metrics import evaluate, truth_template
from toolkits.geometry.evaluation.observation import load_observation
from toolkits.geometry.evaluation.references import build_truth, mesh_fingerprint
from toolkits.geometry.evaluation.replay import (
    RecordingGrounder,
    RecordingSegmenter,
    ReplayGrounder,
    ReplaySegmenter,
    append_jsonl,
    read_jsonl,
)


def _case():
    manifest = build_manifest()
    return manifest, select_cases(manifest, ["libero_10_task_t9_s1"])


def _environment(case):
    rgb = np.zeros((128, 128, 3), dtype=np.uint8)
    rgb[0, 0] = [10, 20, 30]
    geom = {
        "mesh_name": "synthetic",
        "vertices_body": [[-0.04, 0, 0], [0.04, 0, 0], [0, 0.04, 0]],
        "faces": [[0, 1, 2]],
    }
    objects = {
        REFERENCE_OBJECTS[request["reference"]]: {
            "body_name": REFERENCE_OBJECTS[request["reference"]],
            "body_position_world": [0.005, 0, 0.8],
            "body_rotation_world": np.eye(3).tolist(),
            "visual_geoms": [geom],
        }
        for request in case["requests"]
    }
    snapshot = {"schema_version": 1, "units": "m", "objects": objects}
    env = SimpleNamespace(
        closed=False,
        resets=0,
        render_camera=lambda *a, **kw: (rgb, np.full((128, 128), 0.8)),
        get_camera_meta=lambda *a, **kw: {
            "intrinsic_K": [[800, 0, 64], [0, 800, 64], [0, 0, 1]],
            "extrinsic_cam2world": np.eye(4).tolist(),
        },
    )
    env.reset = lambda: setattr(env, "resets", env.resets + 1)
    env.env = SimpleNamespace(
        workers=[SimpleNamespace(get_env_attr=lambda name: deepcopy(snapshot))],
        close=lambda: setattr(env, "closed", True),
    )
    profiles = {
        "schema_version": 1,
        "units": "m",
        "frozen_at": "synthetic fixture",
        "references": {
            request["reference"]: {
                "body_local_xyz": [0, 0, 0],
                "compiled_mesh_name": "synthetic",
                "compiled_mesh_sha256": mesh_fingerprint(geom),
                "method": "explicit synthetic circular center",
            }
            for request in case["requests"]
        },
    }
    return env, profiles


def test_offline_cli_and_fixed_checklist(tmp_path):
    root = Path(__file__).resolve().parents[3]
    script = """
import sys
from toolkits.geometry.evaluation.experiment import main
from toolkits.geometry.evaluation.checklist import build_manifest, manifest_digest
manifest = build_manifest()
assert manifest['counts'] == {'task_rows': 16, 'cases': 80, 'requests': 160}
assert len({r['request_id'] for c in manifest['cases'] for r in c['requests']}) == 160
assert manifest_digest(manifest) == '27f89e9a7a1afe32cfbe4071968b3b8e1dddc2565ef5958d046858bf6bab6e93'
assert not {'mujoco', 'torch', 'rlinf', 'sam3'}.intersection(sys.modules)
assert 'robots.libero.env_server' not in sys.modules
main(['write-manifest', '--output', 'manifest.json'])
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(root), "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads((tmp_path / "manifest.json").read_text()) == build_manifest()


def test_cold_collector_initializes_graphics_before_loading_the_simulator(tmp_path):
    root = Path(__file__).resolve().parents[3]
    sdk = tmp_path / "sdk"
    package = sdk
    for name in ("rlinf", "envs", "sim", "libero"):
        package = package / name
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("")
    (package / "libero_env.py").write_text("""
import os
import sys
from types import ModuleType
assert os.environ.get("MUJOCO_GL") == "egl", "simulator imported before GL setup"
assert os.environ.get("PYOPENGL_PLATFORM") == "egl"
sys.modules["mujoco"] = ModuleType("mujoco")
class LiberoEnv:
    def __init__(self, **kwargs):
        self.options = kwargs
""")
    (package / "utils.py").write_text("""
from types import SimpleNamespace
class Suite:
    n_tasks = 16
    def get_task_init_states(self, task):
        return list(range(5))
benchmark = SimpleNamespace(get_benchmark=lambda name: Suite)
""")
    script = """
import sys
from types import SimpleNamespace
sys.modules["omegaconf"] = SimpleNamespace(
    OmegaConf=SimpleNamespace(create=lambda config: config)
)
from toolkits.geometry.evaluation.checklist import build_manifest, select_cases
from toolkits.geometry.evaluation.libero import make_environment
case = select_cases(build_manifest(), ["libero_10_task_t9_s1"])[0]
env = make_environment(case)
assert env.options["cfg"]["specific_reset_id"] == 46
assert env.options["cfg"]["auto_reset"] is False
assert env.options["num_envs"] == 1
assert env._rpent_evaluation_enabled
"""
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(sdk), str(root))),
        "PYTHONDONTWRITEBYTECODE": "1",
        "LIBERO_TYPE": "pro",
    }
    environment.pop("MUJOCO_GL", None)
    environment.pop("PYOPENGL_PLATFORM", None)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_capture_closes_its_environment_on_failure():
    _, cases = _case()
    env, _ = _environment(cases[0])

    def fail(*args, **kwargs):
        raise RuntimeError("render failed")

    env.render_camera = fail
    with pytest.raises(RuntimeError, match="render failed"):
        capture_case(cases[0], environment_factory=lambda case: env)
    assert env.closed and env.resets == 1


def test_initial_frame_record_replay_and_error_report(tmp_path, monkeypatch):
    from toolkits.geometry.evaluation import libero

    manifest, cases = _case()
    case = cases[0]
    env, profiles = _environment(case)
    capture = capture_case(
        case, height=128, width=128, environment_factory=lambda case: env
    )
    monkeypatch.setattr(libero, "capture_case", lambda *a, **kw: capture)
    inputs = tmp_path / "captures"
    assert experiment.collect_cases(manifest, cases, inputs)["captured"] == 1
    assert env.closed and env.resets == 1
    observation = load_observation(inputs / case["case_id"], manifest, case)
    assert observation.rgb[-1, 0].tolist() == [10, 20, 30]
    truth = build_truth(manifest, cases, inputs, profiles=profiles)
    assert truth["annotations"][0]["model_to_world"][0][3] == 0.005
    rows, cols = np.indices((128, 128))
    radii = np.hypot((cols - 64) * 0.001, (rows - 64) * 0.001)
    mask = (radii >= 0.032) & (radii <= 0.045)
    grounder = SimpleNamespace(
        ground=lambda image, query: MolmoResult(
            found="yellow" not in query,
            point_xy=None if "yellow" in query else (96, 64),
            image_size=(128, 128),
        )
    )
    segmenter = SimpleNamespace(
        segment=lambda *a, **kw: Sam3Result(found=True, mask=mask, score=0.95)
    )
    first = tmp_path / "live"
    experiment.fresh_output(first)
    predictions = experiment.infer_cases(
        manifest,
        cases,
        inputs,
        first,
        grounder=RecordingGrounder(grounder, first / "perception"),
        segmenter=RecordingSegmenter(segmenter, first / "perception"),
    )
    assert predictions[0]["status"] == "usable", predictions[0]
    assert predictions[1]["status"] == "need_observation"
    legacy_grounding = tmp_path / "grounding-results.jsonl"
    records = read_jsonl(first / "perception/grounding.jsonl")
    legacy_grounding.write_text(
        "\n".join(
            json.dumps({**record, "request_id": request["request_id"]})
            for record, request in zip(records, case["requests"], strict=True)
        )
    )
    frozen = tmp_path / "frozen"
    assert (
        experiment.main(
            [
                "freeze-perception",
                "--predictions",
                str(first),
                "--grounding",
                str(legacy_grounding),
                "--output",
                str(frozen),
            ]
        )
        == 0
    )
    second = tmp_path / "replayed"
    assert (
        experiment.main(
            [
                "infer",
                "--inputs",
                str(inputs),
                "--output",
                str(second),
                "--replay",
                str(frozen),
                "--case",
                case["case_id"],
            ]
        )
        == 0
    )
    replayed = read_jsonl(second / "results.jsonl")
    summary = json.loads((second / "summary.json").read_text())
    assert summary["fresh_grounding_calls"] == summary["fresh_segmentation_calls"] == 0
    assert [p["world_xyz"] for p in replayed] == [p["world_xyz"] for p in predictions]
    truth_path = tmp_path / "truth.json"
    truth_path.write_text(json.dumps(truth))
    output = tmp_path / "report"
    assert (
        experiment.main(
            [
                "evaluate",
                "--predictions",
                str(second / "results.jsonl"),
                "--truth",
                str(truth_path),
                "--case",
                case["case_id"],
                "--output",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads((output / "metrics.json").read_text())
    assert report["overall"]["planned"] == 2
    assert report["overall"]["return_rate"] == 0.5
    assert report["rows"][0]["error_mm"] == pytest.approx(5.0, abs=0.2)
    csv_rows = list(
        csv.DictReader(
            io.StringIO((output / "errors.csv").read_text(encoding="utf-8-sig"))
        )
    )
    assert len(csv_rows) == 2 and csv_rows[1]["error_mm"] == ""
    assert case["requests"][1]["request_id"] in (output / "report.md").read_text()
    states = json.loads((second / case["case_id"] / "episode/states.json").read_text())
    assert states["steps"][0]["state"] == {}
    assert "evaluator_snapshot.json" not in states["steps"][0]["artifacts"]
    assert not (second / case["case_id"] / "evaluator_snapshot.json").exists()
    # Observation mutations must not silently change a replayed measurement.
    rgb = inputs / case["case_id"] / "episode/agentview_high.png/00.png"
    rgb.write_bytes(b"changed")
    with pytest.raises(ValueError, match="sensor hash mismatch"):
        load_observation(inputs / case["case_id"], manifest, case)


def test_replay_rejects_changed_prompt_and_mask(tmp_path):
    image = imageio.imwrite(
        "<bytes>", np.zeros((8, 8, 3), dtype=np.uint8), extension=".png"
    )
    grounder = SimpleNamespace(
        ground=lambda *a: MolmoResult(found=True, point_xy=(3, 3), image_size=(8, 8))
    )
    segmenter = SimpleNamespace(
        segment=lambda *a, **kw: Sam3Result(
            found=True, mask=np.ones((8, 8), dtype=bool)
        )
    )
    RecordingGrounder(grounder, tmp_path).ground(image, "cup rim")
    RecordingSegmenter(segmenter, tmp_path).segment(image, text_prompt="cup rim")
    record = read_jsonl(tmp_path / "grounding.jsonl")[0]
    append_jsonl(
        tmp_path / "grounding.jsonl", {**record, "request_id": "another-case/A"}
    )
    assert ReplayGrounder(tmp_path).ground(image, "cup rim").point_xy == (3, 3)
    with pytest.raises(ValueError, match="prompt differs"):
        ReplayGrounder(tmp_path).ground(image, "bowl rim")
    with pytest.raises(ValueError, match="threshold differs"):
        ReplaySegmenter(tmp_path).segment(image, text_prompt="cup rim", min_score=0.3)
    (tmp_path / "mask_0000.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="mask changed"):
        ReplaySegmenter(tmp_path).segment(image, text_prompt="cup rim")


def test_missing_predictions_and_incompatible_reference_metadata():
    manifest, cases = _case()
    truth = truth_template(manifest, cases)
    for annotation in truth["annotations"]:
        annotation.update(
            evaluable=True,
            world_xyz=[0, 0, 0.8],
            annotation_source="synthetic independent point",
        )
    prediction = experiment.prediction_base(manifest, cases[0], cases[0]["requests"][0])
    prediction.update(status="usable", world_xyz=[0.003, 0.004, 0.8])
    report = evaluate(manifest, truth, [prediction], cases=cases)
    assert report["overall"]["unrun"] == 1
    assert report["overall"]["coverage_on_evaluable"] == 0.5
    assert report["overall"]["conditional_error_mm_on_returned_only"][
        "mean"
    ] == pytest.approx(5)
    wrong = {**prediction, "frame_id": "different frame"}
    with pytest.raises(ValueError, match="frame_id mismatch"):
        evaluate(manifest, truth, [wrong], cases=cases)
    with pytest.raises(ValueError, match="duplicate prediction"):
        evaluate(manifest, truth, [prediction, prediction], cases=cases)
    truth["annotations"][0]["definition"] = "body origin"
    with pytest.raises(ValueError, match="definition mismatch"):
        evaluate(manifest, truth, [prediction], cases=cases)


def test_reference_mesh_identity_is_checked(tmp_path):
    from rpent.evaluation.result import write_json_atomic

    manifest, cases = _case()
    env, profiles = _environment(cases[0])
    _, snapshot = capture_case(cases[0], environment_factory=lambda case: env)
    root = tmp_path / cases[0]["case_id"]
    root.mkdir()
    first = next(iter(snapshot["objects"].values()))
    first["visual_geoms"][0]["vertices_body"][0][0] += 0.01
    write_json_atomic(root / "evaluator_snapshot.json", snapshot)
    with pytest.raises(ValueError, match="compiled model differs"):
        build_truth(manifest, cases, tmp_path, profiles=profiles)
    assert frame_id(cases[0]["case_id"]) != "world"
    assert manifest_digest(manifest) == truth_template(manifest)["manifest_sha256"]


def test_snapshot_converts_visual_mesh_to_the_correct_body_frame():
    from toolkits.geometry.evaluation.evaluation_scene import snapshot_simulator

    angle = 0.4
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    position = np.array([0.2, -0.1, 0.9])
    local_offset = np.array([0.01, 0.02, 0.03])
    vertices = np.array([[0, 0, 0], [0.04, 0, 0], [0, 0.04, 0]])
    model = SimpleNamespace(
        geom_bodyid=np.array([0]),
        geom_group=np.array([1]),
        geom_dataid=np.array([0]),
        geom_type=np.array([7]),
        mesh_vertadr=np.array([0]),
        mesh_vertnum=np.array([3]),
        mesh_vert=vertices,
        mesh_faceadr=np.array([0]),
        mesh_facenum=np.array([1]),
        mesh_face=np.array([[0, 1, 2]]),
        geom_id2name=lambda i: "visual",
        mesh_id2name=lambda i: "model",
        body_id2name=lambda i: "plate_1_main",
    )
    data = SimpleNamespace(
        body_xpos=position[None],
        body_xmat=rotation[None],
        geom_xpos=(position + rotation @ local_offset)[None],
        geom_xmat=rotation[None],
    )
    snapshot = snapshot_simulator(
        SimpleNamespace(
            obj_body_id={"plate_1": 0},
            sim=SimpleNamespace(model=model, data=data),
        )
    )
    obj = snapshot["objects"]["plate_1"]
    np.testing.assert_allclose(
        obj["visual_geoms"][0]["vertices_body"], vertices + local_offset
    )
    np.testing.assert_allclose(obj["body_position_world"], position)
    np.testing.assert_allclose(obj["body_rotation_world"], rotation)
