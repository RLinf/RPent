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

"""Resolve independently fixed model points at each captured simulator pose."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from toolkits.geometry.evaluation.checklist import frame_id, manifest_digest
from toolkits.geometry.evaluation.libero import REFERENCE_OBJECTS, file_digest
from toolkits.geometry.evaluation.metrics import truth_template, validate_truth

REFERENCE_PROFILES = {
    "schema_version": 1,
    "units": "m",
    "created_at": "2026-11-06T03:51:26.175850+00:00",
    "source": "complete native LIBERO visual meshes with XML mesh scale; no perception "
    "outputs used",
    "references": {
        "plate": {
            "body_local_xyz": [
                -4.0113735641348266e-05,
                2.3098927379942535e-05,
                0.004225116473237365,
            ],
            "method": "complete-model circular symmetry axis intersected "
            "with the native central support triangles; height "
            "is the actual support surface at that axis, not "
            "the raised rim",
            "quality": {
                "plane_residual_p95_m": 0.0016432064946922224,
                "n_plane_samples": 471,
                "plane_normal_body": [
                    -0.027345147828212426,
                    0.0005791567420560076,
                    0.999625883752377,
                ],
            },
            "mesh_source_sha256": "7d3278b1cefb2ed496094cad18f0c66004ee65ab933122ec8d5efdc78daa96b4",
            "compiled_mesh_name": "model_vis",
            "compiled_mesh_sha256": "8709977241e2f43e5f150f4891f4e8495cb2091c057e0c65609c86042703d673",
            "raw_to_compiled_max_deviation_m": 7.4406750496030584e-09,
            "asset_xml_sha256": "72b4de23b2daa3169843b784998c0c3c30a6559c6c85917c6b86e3d39fa241e6",
        },
        "ramekin": {
            "body_local_xyz": [
                -2.2530517233048224e-05,
                -0.0001603486413628215,
                0.04264924979589367,
            ],
            "method": "complete triangle-mesh radial sections; inner "
            "side within 0.2 mm of the upper lip ridge in "
            "each section; best-fit opening circle plane and "
            "center",
            "quality": {
                "radius_m": 0.04213666176878962,
                "model_circle_residual_p95_m": 0.0006356637756466616,
                "n_reference_samples": 180,
                "radial_sections_observed": 180,
                "upper_band_height_m": 0.00519960012,
                "inner_ridge_height_tolerance_m": 0.0002,
            },
            "mesh_source_sha256": "9ef1877c027112d27fca3e8b16e7c226fb02f52345750b8d8c8217ad0c41c5eb",
            "compiled_mesh_name": "glazed_rim_porcelain_ramekin_vis",
            "compiled_mesh_sha256": "3ca4be4d19f362f0768aabf2695098b6a42793e8c67e264920e87e86d97678d2",
            "raw_to_compiled_max_deviation_m": 3.863531683507841e-09,
            "asset_xml_sha256": "61cf1b665e52faf68a98b044da3fc2b42f2303485b6445743741fbf99d3f9bcf",
        },
        "cheese": {
            "body_local_xyz": [
                0.00016336850000000028,
                -3.2334999999999656e-05,
                0.009144379843612523,
            ],
            "method": "complete main rectangular top-face extents "
            "evaluated on its fitted native face plane; "
            "microwave door, button and feet excluded",
            "quality": {
                "plane_residual_p95_m": 0.000219427167580824,
                "n_plane_samples": 9258,
                "plane_normal_body": [
                    0.00044497010365831337,
                    -0.021344136380563452,
                    0.9997720889501641,
                ],
            },
            "mesh_source_sha256": "0110fcd0351384174e70956ae83c4b30a677bc5673aa8107213b73334efbdf3b",
            "compiled_mesh_name": "cream_cheese_vis",
            "compiled_mesh_sha256": "4ca854a9f2737bbed062fae01f21943d8e05319e49e87dd8ce558503ad7d4156",
            "raw_to_compiled_max_deviation_m": 4.2944324260371236e-09,
            "asset_xml_sha256": "a1bbeb0321d9d20d723aca62baa54f6012bf74c0af5d4f767a58ca0cdde50d95",
        },
        "soup": {
            "body_local_xyz": [
                -0.00040530661435838125,
                0.03337160242613327,
                -0.0035824159091846497,
            ],
            "method": "positive model-Y sealed end-cap plane and complete "
            "projected circular outline; the intrinsic lid "
            "identity is fixed and follows body rotation",
            "quality": {
                "plane_residual_p95_m": 0.00029719073998569167,
                "n_plane_samples": 4147,
                "radius_m": 0.028913108023974732,
                "model_circle_residual_p95_m": 0.001165821602730573,
                "n_reference_samples": 56,
                "plane_normal_body": [
                    -0.00014863491120393007,
                    0.9999395849093959,
                    0.010991107263983403,
                ],
            },
            "mesh_source_sha256": "a9f36a95f4ad97cb141f134251114d46cc4f3950229d76ac815e751cc0cf121e",
            "compiled_mesh_name": "textured_vis",
            "compiled_mesh_sha256": "fa1850324f6762c85b5d477d67447d822b578b9c7a21028f9c1a2fd64daa798e",
            "raw_to_compiled_max_deviation_m": 5.7917014895582376e-09,
            "asset_xml_sha256": "09a230f3cc5cbb437190adb9d2212ad20edfb5904bda8094a5db7acb2595afc3",
        },
        "black_bowl": {
            "body_local_xyz": [
                -3.7287833468191217e-06,
                -5.2425992253362524e-05,
                0.051705128378899914,
            ],
            "method": "complete triangle-mesh radial sections; "
            "inner side within 0.2 mm of the upper lip "
            "ridge in each section; best-fit opening "
            "circle plane and center",
            "quality": {
                "radius_m": 0.054661183283665586,
                "model_circle_residual_p95_m": 0.0004326819173051847,
                "n_reference_samples": 180,
                "radial_sections_observed": 180,
                "upper_band_height_m": 0.00631797612,
                "inner_ridge_height_tolerance_m": 0.0002,
            },
            "mesh_source_sha256": "6474d536a740e995c3a2eaeb74c61e14da94131b0b222c250b61b30b13d11229",
            "compiled_mesh_name": "akita_black_bowl_vis",
            "compiled_mesh_sha256": "94ce0d1b0ef33bc779a07f107db5667f9f9389752f7c998748f8d2bfeab06fc2",
            "raw_to_compiled_max_deviation_m": 6.242374269741382e-09,
            "asset_xml_sha256": "18c1074cfa09baea739bb75928f9bd2bd80e22ac18655f6a27f005dbf77ccfda",
        },
        "wine_bottle": {
            "body_local_xyz": [
                1.3937629601945828e-09,
                -8.454488300816033e-05,
                0.03135691565395895,
            ],
            "method": "midpoint of the fixed full-model "
            "near-cylindrical outer barrel interval; "
            "base curvature, shoulder, neck and mouth "
            "excluded",
            "quality": {
                "barrel_bounds_body_z_m": [0.00717993162170088, 0.05553389968621701],
                "radius_m": 0.02160093852464459,
                "section_spacing_m": 0.00014652717595307918,
                "n_sections": 1024,
                "criterion": "longest complete-model "
                "radius plateau within 1.5 "
                "percent of dominant radius "
                "and absolute dr/dz at most "
                "0.03",
            },
            "mesh_source_sha256": "92a7a897bdac34f47161cb418fbe6f7fc9056d2f0e93647beb8139cffeed270a",
            "compiled_mesh_name": "wine_bottle_red_main_vis",
            "compiled_mesh_sha256": "a34143a785526c080ed6dcbf63bb2cfc6faed6f88a898c969c72b493834a4b9b",
            "raw_to_compiled_max_deviation_m": 6.864590521908002e-09,
            "asset_xml_sha256": "9bdae6cb59f9260544cb802d4babe32c23b9dd66e0a5fb2bdb4662b468ce1a17",
        },
        "gray_white_mug": {
            "body_local_xyz": [
                0.016489495041088747,
                -0.0018143471594326943,
                0.11426321582817268,
            ],
            "method": "complete triangle-mesh radial sections; "
            "inner side within 0.2 mm of the upper "
            "lip ridge in each section; best-fit "
            "opening circle plane and center",
            "quality": {
                "radius_m": 0.045941139684101746,
                "model_circle_residual_p95_m": 0.0004981037076688279,
                "n_reference_samples": 180,
                "radial_sections_observed": 180,
                "upper_band_height_m": 0.01375488024,
                "inner_ridge_height_tolerance_m": 0.0002,
            },
            "mesh_source_sha256": "e0dc3f0a0585d172c6ff22e49173774e37bc8a4a8e74d8c583ecb7eb314ffc13",
            "compiled_mesh_name": "porcelain_mug_vis",
            "compiled_mesh_sha256": "2ec19268e02e9b948d700b6c5b224b0b094bdcb6ce3d40baa455ff3713666dd1",
            "raw_to_compiled_max_deviation_m": 6.3354281181205305e-09,
            "asset_xml_sha256": "cd72828fc40d20ccd8d3fb08ae37fbc87cb9957309eb4e3d04244554491f753e",
        },
        "yellow_white_mug": {
            "body_local_xyz": [
                -0.01785442948570012,
                -0.0009000509997171457,
                0.1037523430813633,
            ],
            "method": "complete triangle-mesh radial "
            "sections; inner side within 0.2 mm of "
            "the upper lip ridge in each section; "
            "best-fit opening circle plane and "
            "center",
            "quality": {
                "radius_m": 0.04852629787350218,
                "model_circle_residual_p95_m": 0.0005183658842933172,
                "n_reference_samples": 180,
                "radial_sections_observed": 180,
                "upper_band_height_m": 0.01257527988,
                "inner_ridge_height_tolerance_m": 0.0002,
            },
            "mesh_source_sha256": "aeabcc3a57b84549568a1d6907542ef9e65ea126c101f6f3985deb8b55d3aa79",
            "compiled_mesh_name": "white_yellow_mug_vis",
            "compiled_mesh_sha256": "5ee47bb29427f1559a21ea23e618e24bbf7251074f236b46aa595a2288220a59",
            "raw_to_compiled_max_deviation_m": 7.670717026322836e-09,
            "asset_xml_sha256": "134bfdb2e605b60609b63c4bae60d833ad027902973dfb6b912768e0870b435e",
        },
        "microwave": {
            "body_local_xyz": [
                2.8282500000004207e-05,
                0.0009668209999999997,
                0.1869113890000001,
            ],
            "method": "complete main rectangular top-face extents "
            "evaluated on its fitted native face plane; "
            "microwave door, button and feet excluded",
            "quality": {
                "plane_residual_p95_m": 8.326672684688674e-17,
                "n_plane_samples": 27,
                "plane_normal_body": [0.0, -1.9469413143858213e-32, 1.0],
            },
            "mesh_source_sha256": "0825383a47b711561328d1db63ee9aa15c2c182f362298c315ee2cb2cd0fba1d",
            "compiled_mesh_name": "micro",
            "compiled_mesh_sha256": "d1a18823945aa6c207f044b1fc978d5fc52859f6798e3fe76630b11cd40b912a",
            "raw_to_compiled_max_deviation_m": 1.37683179600436e-08,
            "asset_xml_sha256": "d06c1e44acb830c529a843e566deedfc5693a6e11f68a3d1d59d49021890fbdf",
        },
    },
    "frozen_at": "2026-11-06T05:18:54.014566+00:00",
    "compiled_catalog_sha256": "72d363cce1c4d79933f05e5db1298d5ae99f6c26c0b09d21354e396d44bdf7cd",
    "reference_convention": "Complete native model geometry fixed before any formal "
    "perception calls; all 160 existing target points are "
    "evaluable regardless of visibility; world coordinates come "
    "from same initial-frame MuJoCo body FK",
}


def mesh_fingerprint(geom: dict[str, Any]) -> str:
    """Match the frozen catalog's body-local mesh fingerprint at nanometre rounding."""
    vertices = np.asarray(geom["vertices_body"], dtype=float)
    faces = np.asarray(geom["faces"], dtype=np.int64)
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or not np.isfinite(vertices).all()
        or faces.ndim != 2
        or faces.shape[1] != 3
        or not len(vertices)
        or not len(faces)
        or faces.min() < 0
        or faces.max() >= len(vertices)
    ):
        raise ValueError("invalid compiled reference mesh")
    return hashlib.sha256(np.round(vertices, 9).tobytes() + faces.tobytes()).hexdigest()


def build_truth(
    manifest: dict[str, Any],
    cases: list[dict[str, Any]],
    inputs: Path,
    *,
    profiles: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Check model identity, then transform fixed local points using same-frame FK.

    A model point remains evaluable when occluded. This function never reads
    masks, selected pixels or predictions and never substitutes a body origin.
    """
    profiles = REFERENCE_PROFILES if profiles is None else profiles
    if (
        profiles.get("schema_version") != 1
        or profiles.get("units") != "m"
        or not profiles.get("frozen_at")
    ):
        raise ValueError(
            "reference profiles must be frozen metre-based model annotations"
        )
    truth = truth_template(manifest, cases)
    annotations = {row["request_id"]: row for row in truth["annotations"]}
    profile_hash = manifest_digest(profiles)
    truth["profile_sha256"] = profile_hash
    for case in cases:
        path = inputs / case["case_id"] / "evaluator_snapshot.json"
        snapshot = json.loads(path.read_text())
        snapshot_hash = file_digest(path)
        for key, expected in {
            "schema_version": 1,
            "case_id": case["case_id"],
            "frame_id": frame_id(case["case_id"]),
            "units": "m",
            "step": 0,
            "camera": "agentview",
        }.items():
            if snapshot.get(key) != expected:
                raise ValueError(
                    f"reference snapshot {key} mismatch: {case['case_id']}"
                )
        for request in case["requests"]:
            name = request["reference"]
            profile = profiles["references"][name]
            obj = snapshot["objects"][REFERENCE_OBJECTS[name]]
            matches = [
                geom
                for geom in obj["visual_geoms"]
                if geom["mesh_name"] == profile["compiled_mesh_name"]
                and mesh_fingerprint(geom) == profile["compiled_mesh_sha256"]
            ]
            if not matches:
                raise ValueError(
                    f"compiled model differs from frozen reference: {name}"
                )
            pose = np.eye(4)
            pose[:3, :3] = obj["body_rotation_world"]
            pose[:3, 3] = obj["body_position_world"]
            annotations[request["request_id"]].update(
                evaluable=True,
                model_local_xyz=profile["body_local_xyz"],
                model_to_world=pose.tolist(),
                annotation_source=(
                    f"Frozen complete-model profile {profile_hash}; {profile['method']}; "
                    f"same-frame MuJoCo FK; snapshot {snapshot_hash}"
                ),
                profile_sha256=profile_hash,
                snapshot_sha256=snapshot_hash,
                body_name=obj["body_name"],
            )
    validate_truth(manifest, truth, cases)
    return truth
