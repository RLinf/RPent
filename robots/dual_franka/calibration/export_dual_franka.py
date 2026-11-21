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
"""Export a fixed-camera candidate as easy_handeye-compatible YAML for Dual Franka."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import yaml
from common import Record, load_candidate, rigid_transform
from scipy.spatial.transform import Rotation


def export(candidate_path: Path, output: Path) -> Record:
    """Export a camera-to-right-base transform without overwriting an existing file.

    Args:
        candidate_path: Fixed-camera candidate beside its report and samples.
        output: New YAML file referenced by perception.calibration.

    Returns:
        YAML content with RPent-compatible parameters and transformation fields.
    """
    candidate = load_candidate(candidate_path)
    if candidate["calibration_mode"] != "eye_to_hand" or candidate["arm"] != "right":
        raise ValueError(
            "Dual Franka projection requires a fixed-camera eye_to_hand candidate in right_base; wrist and left-base candidates cannot be used"
        )
    matrix = rigid_transform(candidate["T_right_base_camera"])
    parameters = dict(candidate["parameters"])
    if (
        parameters.get("eye_on_hand") is not False
        or parameters.get("robot_base_frame") != "right_base"
    ):
        raise ValueError(
            "Candidate parameters do not describe a fixed camera in right_base"
        )
    values = np.r_[matrix[:3, 3], Rotation.from_matrix(matrix[:3, :3]).as_quat()]
    result = {
        "parameters": parameters,
        "transformation": dict(
            zip(("x", "y", "z", "qx", "qy", "qz", "qw"), map(float, values))
        ),
        "calibration_status": candidate["status"],
        "camera_serial": candidate["camera_serial"],
        "source_candidate": str(candidate_path.resolve()),
    }
    text = yaml.safe_dump(result, sort_keys=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(text)
    return result


def main() -> None:
    """Write YAML without changing the active robot configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    export(args.candidate, args.output)
    logging.getLogger(__name__).info(
        "Exported %s; validate physically before configuring RPent to use it",
        args.output,
    )


if __name__ == "__main__":
    main()
