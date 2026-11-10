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

"""Convert official RAFT-Stereo flow and prepared images to millimetre RGB-D."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np

from rpent.utils.logging import get_logger

logger = get_logger(__name__)


def convert_depth(*, source: Path, flow_dir: Path, output: Path) -> None:
    """Convert the official demo's negative horizontal flow to millimetre PNGs."""
    import cv2

    metadata = json.loads((source / "stereo.json").read_text())
    names = metadata["frames"]
    if (
        not names
        or len(set(names)) != len(names)
        or any(Path(name).name != name or name in (".", "..") for name in names)
    ):
        raise ValueError("Invalid frame names in stereo.json")
    flows = {p.stem: p for p in flow_dir.glob("*.npy")}
    if set(names) != flows.keys():
        raise ValueError("Official RAFT output must contain one .npy per prepared pair")
    fx_baseline = float(metadata["fx_baseline"])
    if not np.isfinite(fx_baseline) or fx_baseline <= 0:
        raise ValueError("Invalid focal length times baseline")
    if output.resolve().is_relative_to(source.resolve()):
        raise ValueError("Output must be outside the input directory")
    output.mkdir(parents=True, exist_ok=False)
    for side in ("rgb", "depth"):
        (output / side).mkdir()
    # RTAB-Map sorts digit groups numerically, so fractions need equal width.
    precision = max(len(name.partition(".")[2]) for name in names)
    for name in names:
        seconds, _, fraction = name.partition(".")
        filename = f"{seconds}.{fraction.ljust(precision, '0')}.png"
        flow = np.load(flows[name], allow_pickle=False)
        if flow.shape != tuple(reversed(metadata["image_size"])) or not np.issubdtype(
            flow.dtype, np.floating
        ):
            raise ValueError(f"Expected floating-point HxW flow: {flows[name]}")
        disparity = -flow
        meters = np.zeros(disparity.shape, dtype=np.float32)
        np.divide(
            fx_baseline,
            disparity,
            out=meters,
            where=np.isfinite(disparity) & (disparity > 0),
        )
        valid = np.isfinite(meters) & (meters > 0) & (meters <= 65.535)
        depth = np.rint(np.where(valid, meters, 0) * 1000).astype(np.uint16)
        if not depth.any():
            raise ValueError(f"No representable metric depth for {name}")
        if not cv2.imwrite(str(output / "depth" / filename), depth):
            raise OSError(f"Cannot write depth for {name}")
        shutil.copyfile(source / "pairs" / name / "im0.png", output / "rgb" / filename)
    for name in ("camera.yaml", "local_transform.txt"):
        shutil.copyfile(source / name, output / name)
    logger.info("Converted %d RGB-D pairs into %s", len(names), output)


def main(argv: list[str] | None = None) -> None:
    """Convert official RAFT-Stereo flow files to aligned RGB-D images."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", dest="source", type=Path, required=True)
    parser.add_argument("--flow-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        convert_depth(**vars(args))
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Depth conversion failed: {exc}\n")


if __name__ == "__main__":
    main()
