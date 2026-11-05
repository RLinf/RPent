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

"""Capture and segment one LIBERO view using existing env and SAM3 services."""

from __future__ import annotations

import argparse
from contextlib import closing
from pathlib import Path
from typing import Any

import numpy as np

from rpent.robots.components.sam3_client import Sam3Client
from rpent.session import EnvState
from rpent.tools.localization.localization import FrameSource, RpcFrameSource, capture
from rpent.tools.perception import Sam3Tools
from rpent.utils.logging import get_logger
from rpent.utils.rpc import make_rpc_client

logger = get_logger("libero_localization_demo")


def run_demo(
    source: FrameSource,
    perception: Sam3Tools,
    output_dir: Path,
    *,
    prompt: str,
    camera: str = "agentview",
    height: int = 1024,
    width: int = 1024,
) -> dict[str, Any]:
    """Record one aligned RGB-D view and localize its SAM3 object mask.

    Args:
        source: Borrowed capture source; the environment is never reset or stepped.
        perception: SAM3 helper backed by a caller-owned model client.
        output_dir: New directory for EnvState artifacts; must not already exist.
        prompt: Text description of the target object.
        camera: Environment camera name.
        height: Capture height in pixels.
        width: Capture width in pixels.

    Returns:
        Detection status, score and median world coordinates in metres, or a
        reason why localization was unavailable. Also saved as ``result.json``.

    Raises:
        RuntimeError: If artifact recording fails.
    """
    output_dir.mkdir(parents=True, exist_ok=False)
    frame = capture(source, camera, height=height, width=width)
    state = EnvState(output_dir)
    with state.record_step(state={"camera": frame.camera, "prompt": prompt}) as step:
        artifacts = {
            "camera.png": frame.rgb,
            "depth.npz": frame.depth_m,
            "world.npz": frame.world_xyz,
            "valid_mask.npz": frame.valid_mask,
            "metadata.json": frame.describe(),
        }
        for name, value in artifacts.items():
            if state.save(name, value, step=step) is None:
                raise RuntimeError(f"could not save {name}")
        detection, overlay = perception.segment(
            state, step=step, image="camera.png", prompt=prompt
        )
        result: dict[str, Any] = {
            "camera": frame.camera,
            "found": detection.found,
            "score": detection.score,
            "world_xyz": None,
        }
        if detection.found and isinstance(detection.mask, np.ndarray):
            result.update(
                perception.mask_to_world(
                    state, step=step, world_map="world.npz", mask=detection.mask
                )
            )
            if state.save("object_mask.npz", detection.mask, step=step) is None:
                raise RuntimeError("could not save object_mask.npz")
        else:
            result["world_error"] = detection.reason or "SAM3 found no mask"
        if overlay is not None:
            if state.save("overlay.png", overlay, step=step) is None:
                raise RuntimeError("could not save overlay.png")
        if state.save("result.json", result, step=step) is None:
            raise RuntimeError("could not save result.json")
    return result


def main() -> None:
    """Run the LIBERO localization demo against two existing RPC services."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-endpoint", required=True)
    parser.add_argument("--sam3-endpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--camera", choices=("agentview", "robot0_eye_in_hand"), default="agentview"
    )
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    args = parser.parse_args()
    with (
        closing(make_rpc_client(args.env_endpoint)) as env_rpc,
        closing(make_rpc_client(args.sam3_endpoint)) as sam3_rpc,
    ):
        result = run_demo(
            RpcFrameSource(env_rpc),
            Sam3Tools(Sam3Client(sam3_rpc)),
            args.output_dir,
            prompt=args.prompt,
            camera=args.camera,
            height=args.height,
            width=args.width,
        )
    logger.info("Localization result: %s; artifacts: %s", result, args.output_dir)


if __name__ == "__main__":
    main()
