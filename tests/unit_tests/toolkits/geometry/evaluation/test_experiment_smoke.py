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

"""Run the fixed-frame experiment with fake external services, without accuracy gates."""

import json
from types import SimpleNamespace

import numpy as np

from rpent.robots.components.molmo_client import MolmoResult
from rpent.robots.components.sam3_client import Sam3Result
from toolkits.geometry.evaluation import experiment
from toolkits.geometry.evaluation.checklist import build_manifest
from toolkits.geometry.evaluation.evaluation_scene import ReferenceEnvAdapter


class _Env:
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
        return "geometry smoke fixture"

    def get_camera_meta(self, camera_name, height, width):
        return {
            "intrinsic_K": [[height, 0, width / 2], [0, height, height / 2], [0, 0, 1]],
            "extrinsic_cam2world": np.eye(4).tolist(),
            "depth_near": 0.1,
            "depth_far": 10.0,
        }

    def render_camera(self, camera_name, height, width, depth):
        return (
            np.zeros((height, width, 3), dtype=np.uint8),
            np.full((height, width), 0.5, dtype=np.float32),
        )


def test_fixed_frame_experiment_smoke(tmp_path, monkeypatch):
    manifest = build_manifest()
    assert len(manifest["cases"]) == 80
    cases = manifest["cases"][:1]
    rows, cols = np.indices((1024, 1024))
    mask = np.hypot(rows - 512, cols - 512) <= 100
    sam = SimpleNamespace(
        segment=lambda *a, **kw: Sam3Result(found=True, mask=mask, score=0.95)
    )
    grounder = SimpleNamespace(
        ground=lambda *a: MolmoResult(
            found=True, point_xy=(512, 512), image_size=(1024, 1024)
        )
    )
    lifecycle = []

    def start(args, case, output, components):
        lifecycle.append(components)
        assert components in ({"env"}, {"sam3"})
        daemon = SimpleNamespace(stop=lambda: lifecycle.append("stopped"))
        runtime = {"env": _Env()} if components == {"env"} else {"sam3_client": sam}
        return [daemon], runtime

    monkeypatch.setattr(experiment, "_start_runtime", start)
    inputs = tmp_path / "observations"
    collected = experiment.collect(
        SimpleNamespace(output=inputs, env_endpoint=None), manifest, cases
    )
    assert collected["captured"] == 1 and collected["failed"] == 0

    output = tmp_path / "predictions"
    summary = experiment.infer(
        SimpleNamespace(
            inputs=inputs,
            output=output,
            geometry_model="test:geometry",
            geometry_base_url=None,
            permissive=True,
        ),
        manifest,
        cases,
        grounder=grounder,
    )
    assert summary["recorded"] == summary["returned_centers"] == 2, summary
    predictions = [
        json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()
    ]
    assert all(row["step"] == 0 and row["camera"] == "agentview" for row in predictions)
    assert all(np.isfinite(row["world_xyz"]).all() for row in predictions)
    assert lifecycle == [{"env"}, "stopped", {"sam3"}, "stopped"]

    truth_path = tmp_path / "truth.json"
    assert experiment.main(["truth-template", "--output", str(truth_path)]) == 0
    truth = json.loads(truth_path.read_text())
    assert len(truth["annotations"]) == 160
    for row in truth["annotations"]:
        row.update(
            evaluable=True,
            world_xyz=[0, 0, 1],
            annotation_source="synthetic smoke fixture; no accuracy claim",
        )
    truth_path.write_text(json.dumps(truth))
    report_path = tmp_path / "report"
    assert (
        experiment.main(
            [
                "evaluate",
                "--truth",
                str(truth_path),
                "--predictions",
                str(output / "results.jsonl"),
                "--output",
                str(report_path),
                "--case",
                cases[0]["case_id"],
            ]
        )
        == 0
    )
    report = json.loads((report_path / "metrics.json").read_text())
    assert report["overall"]["planned"] == len(report["rows"]) == 2
    assert all(np.isfinite(row["error_m"]) for row in report["rows"])
    assert len((report_path / "errors.csv").read_text().splitlines()) == 3


def test_reference_snapshot_smoke():
    model = SimpleNamespace(
        geom_bodyid=np.array([0]),
        geom_group=np.array([1]),
        geom_dataid=np.array([0]),
        geom_type=np.array([7]),
        mesh_vertadr=np.array([0]),
        mesh_vertnum=np.array([3]),
        mesh_vert=np.array([[0, 0, 0], [0.03, 0, 0], [0, 0.04, 0]]),
        mesh_faceadr=np.array([0]),
        mesh_facenum=np.array([1]),
        mesh_face=np.array([[0, 1, 2]]),
        geom_id2name=lambda _: "plate_visual",
        mesh_id2name=lambda _: "plate_mesh",
        body_id2name=lambda _: "plate_body",
    )
    data = SimpleNamespace(
        body_xpos=np.array([[0, 0, 0.8]]),
        body_xmat=np.array([np.eye(3)]),
        geom_xpos=np.array([[0, 0, 0.8]]),
        geom_xmat=np.array([np.eye(3)]),
    )
    env = SimpleNamespace(
        sim=SimpleNamespace(model=model, data=data), obj_body_id={"plate_1": 0}
    )
    snapshot = ReferenceEnvAdapter(env).rpent_reference_snapshot
    assert snapshot["objects"]["plate_1"]["visual_geoms"]
    assert json.loads(json.dumps(snapshot, allow_nan=False))["units"] == "m"
