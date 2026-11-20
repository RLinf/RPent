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

"""Fresh stereo depth for RTAB-Map obstacle segmentation on the Pi."""

from __future__ import annotations

import json
import math
import signal
import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]


@dataclass(frozen=True)
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Image dimensions must be positive")
        if not all(math.isfinite(v) for v in (self.fx, self.fy, self.cx, self.cy)):
            raise ValueError("Intrinsics must be finite")
        if min(self.fx, self.fy) <= 0:
            raise ValueError("Focal lengths must be positive; zero means uncalibrated")


class DepthProjector:
    """Cache pixel rays and preserve surface samples without interpolation."""

    def __init__(self, intrinsics: Intrinsics, stride: int = 3) -> None:
        if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
            raise ValueError("Pixel stride must be a positive integer")
        self.intrinsics = intrinsics
        self.stride = stride
        u, v = np.meshgrid(
            np.arange(0, intrinsics.width, stride, dtype=np.float32),
            np.arange(0, intrinsics.height, stride, dtype=np.float32),
        )
        self.rays = np.stack(
            (
                (u - intrinsics.cx) / intrinsics.fx,
                (v - intrinsics.cy) / intrinsics.fy,
                np.ones_like(u),
            ),
            axis=-1,
        )

    def project(self, depth: FloatArray, minimum: float, maximum: float) -> FloatArray:
        """Return finite XYZ in optical axes: right, down, forward."""
        if not 0 < minimum < maximum or not math.isfinite(maximum):
            raise ValueError("Depth limits must be finite, positive and increasing")
        if depth.shape != (self.intrinsics.height, self.intrinsics.width):
            raise ValueError("Depth dimensions must match intrinsics")
        sampled = depth[:: self.stride, :: self.stride]
        valid = np.isfinite(sampled) & (sampled >= minimum) & (sampled <= maximum)
        return np.ascontiguousarray(
            self.rays[valid] * sampled[valid, None], dtype=np.float32
        )


class LocalStereo:
    def __init__(self, cal, cfg):

        import cv2

        from robots.lekiwi.stereo import resized_k

        self.cal, self.cfg = cal, cfg
        self.size = (cfg["width"], cfg["height"])
        k = resized_k(cal.rectifications[0][1][:3, :3], cal.size, self.size)
        self.fx_baseline = (
            abs(cal.rectifications[1][1][0, 3]) * self.size[0] / cal.size[0]
        )
        self.projector = DepthProjector(
            Intrinsics(*self.size, k[0, 0], k[1, 1], k[0, 2], k[1, 2]),
            cfg["pixel_stride"],
        )
        self.matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=cfg["num_disparities"],
            blockSize=cfg["block_size"],
            P1=8 * cfg["block_size"] ** 2,
            P2=32 * cfg["block_size"] ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=50,
            speckleRange=2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

    def compute(self, raw):

        import cv2

        if raw.shape != (self.cal.size[1], self.cal.size[0] * 2, 3):
            raise ValueError("Unexpected stereo frame size")
        rect = [
            cv2.resize(
                cv2.remap(im, *maps, cv2.INTER_LINEAR),
                self.size,
                interpolation=cv2.INTER_AREA,
            )
            for im, maps in zip(np.split(raw, 2, axis=1), self.cal.maps)
        ]
        gray = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in rect]
        disparity = self.matcher.compute(*gray).astype(np.float32) / 16
        d = np.full(disparity.shape, np.nan, np.float32)
        valid = disparity > 1
        d[valid] = self.fx_baseline / disparity[valid]
        valid &= (d >= self.cfg["minimum_depth_m"]) & (d <= self.cfg["maximum_depth_m"])
        d[~valid] = np.nan
        points = self.projector.project(
            d, self.cfg["minimum_depth_m"], self.cfg["maximum_depth_m"]
        )
        # TF describes the physical optical frame, before rectification.
        raw_optical = points @ self.cal.rectifications[0][0]
        return np.ascontiguousarray(raw_optical, dtype=np.float32), {
            "depth_valid_fraction": float(valid.mean()),
            "point_count": len(points),
        }


def run_stereo() -> None:
    """Publish valid depth rays; RTAB-Map owns ground/obstacle segmentation."""
    import cv2
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image, PointCloud2
    from sensor_msgs_py.point_cloud2 import create_cloud_xyz32
    from std_msgs.msg import Header

    from robots.lekiwi.navigation.bridge import image_bgr
    from robots.lekiwi.navigation.config import stereo_config
    from robots.lekiwi.stereo import StereoCalibration

    config = stereo_config()
    cv2.setNumThreads(2)
    cv2.ocl.setUseOpenCL(False)
    engine = LocalStereo(StereoCalibration(), config)
    rclpy.init(args=["--ros-args", "--disable-external-lib-logs"])
    node = rclpy.create_node("lekiwi_stereo")
    points_pub = node.create_publisher(
        PointCloud2, "/stereo/navigation/points", qos_profile_sensor_data
    )
    latest = []
    previous = 0
    ending = False

    def stop(*_):
        nonlocal ending
        ending = True

    def receive(message):
        stamp = message.header.stamp.sec * 10**9 + message.header.stamp.nanosec
        if (
            stamp > previous
            and 0 <= (time.time_ns() - stamp) / 1e9 <= config["maximum_age_seconds"]
        ):
            latest[:] = [(message, stamp)]

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    node.create_subscription(
        Image, "/camera/image_raw", receive, qos_profile_sensor_data
    )
    try:
        while not ending and rclpy.ok():
            while not latest and not ending and rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.01)
            if ending or not rclpy.ok():
                break
            message, stamp = latest.pop()
            previous = stamp
            points, status = engine.compute(image_bgr(message))
            age = (time.time_ns() - stamp) / 1e9
            healthy = (
                0 <= age <= config["maximum_age_seconds"]
                and status["depth_valid_fraction"] >= config["minimum_depth_coverage"]
                and len(points) >= 100
            )
            if healthy:
                points_pub.publish(
                    create_cloud_xyz32(
                        Header(
                            stamp=message.header.stamp,
                            frame_id="zed_left_optical_frame",
                        ),
                        points,
                    )
                )
            print(
                json.dumps(
                    {"healthy": healthy, "age_seconds": age, **status}, allow_nan=False
                ),
                flush=True,
            )
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main(argv=None) -> None:
    import argparse
    from pathlib import Path

    from robots.lekiwi.runtime_config import set_robot_config_path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-config", type=Path)
    args = parser.parse_args(argv)
    set_robot_config_path(args.robot_config)
    run_stereo()


if __name__ == "__main__":
    main()
