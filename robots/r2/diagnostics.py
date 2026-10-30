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

"""Check an R2 env server without moving the robot.

Reads the server's limits, the robot's health and TCP poses, and one image
per camera, and saves the images for review.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio

from robots.r2.env_client import R2EnvClient
from rpent.utils.rpc import make_rpc_client
from rpent.utils.rpc.client_utils import wait_for_ready


def check(endpoint: str, output_dir: Path) -> dict[str, Any]:
    """Connect without the reset, observe once, and save each camera image."""
    rpc = make_rpc_client(endpoint)
    wait_for_ready(rpc, timeout_s=60.0)
    client = R2EnvClient(rpc, reset_on_connect=False)
    observation = client.observe()
    output_dir.mkdir(parents=True, exist_ok=True)
    cameras: dict[str, Any] = {}
    for name, view in observation["views"].items():
        if "error" in view:
            cameras[name] = {"error": view["error"]}
            continue
        imageio.imwrite(output_dir / f"{name}.png", view["rgb"])
        depth = view.get("depth_mm")
        meta = observation["camera_meta"]["cameras"][name]
        cameras[name] = {
            "image_shape": list(view["rgb"].shape[:2]),
            "valid_depth": None
            if depth is None
            else round(float((depth > 0).mean()), 3),
            "calibrated": meta["intrinsics"] is not None,
            "posed": meta["T_base_camera"] is not None,
        }
    state = observation["robot_state"]
    return {
        "limits": client.meta["limits"],
        "health": state["health"],
        "tcp": {arm: entry["tcp_xyz"] for arm, entry in state["arms"].items()},
        "cameras": cameras,
        "images": str(output_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-endpoint", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("r2_check"))
    args = parser.parse_args()
    report = check(args.env_endpoint, args.output_dir)
    print(json.dumps(report, indent=2))
    return 0 if report["health"]["healthy"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
