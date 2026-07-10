"""Direct pyrealsense2 scene camera for the SO101 env (depth aligned to color).

LeRobot's RealSense wrapper does not align depth to color and does not expose
intrinsics, both of which we need for pixel -> 3D backprojection. So the scene
camera is managed here directly via ``pyrealsense2``: a single pipeline streams
color + depth, ``rs.align`` registers depth into the color frame, and the color
intrinsics + depth scale are read from the active profile.

The arm (hand) camera stays under LeRobot (color only); only the fixed scene
camera needs depth.
"""
from __future__ import annotations

import numpy as np


class SceneCameraD405:
    """Color + depth (aligned to color) from an Intel RealSense (e.g. D405)."""

    def __init__(
        self,
        serial: str,
        *,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        warmup_frames: int = 15,
    ) -> None:
        import pyrealsense2 as rs

        self._rs = rs
        self._serial = str(serial)
        self._width = int(width)
        self._height = int(height)

        self._pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(self._serial)
        cfg.enable_stream(rs.stream.color, self._width, self._height, rs.format.rgb8, int(fps))
        cfg.enable_stream(rs.stream.depth, self._width, self._height, rs.format.z16, int(fps))
        self._profile = self._pipeline.start(cfg)

        # Align depth into the color frame so depth[row, col] matches the
        # color pixel (row, col).
        self._align = rs.align(rs.stream.color)

        depth_sensor = self._profile.get_device().first_depth_sensor()
        self._depth_scale = float(depth_sensor.get_depth_scale())  # meters / unit

        color_stream = self._profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_stream.get_intrinsics()
        self._K = np.array(
            [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

        for _ in range(max(0, warmup_frames)):
            self._pipeline.wait_for_frames()

    # -- capture -----------------------------------------------------------

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(color_rgb_uint8 [H,W,3], depth_m_float32 [H,W])``.

        Depth is metric (meters), aligned to the color frame; invalid/no-return
        pixels are ``0.0``.
        """
        frames = self._pipeline.wait_for_frames()
        frames = self._align.process(frames)
        color = frames.get_color_frame()
        depth = frames.get_depth_frame()
        if not color or not depth:
            raise RuntimeError("scene camera: incomplete frameset (no color/depth)")
        color_img = np.ascontiguousarray(np.asanyarray(color.get_data()), dtype=np.uint8)
        depth_raw = np.asanyarray(depth.get_data())  # uint16, depth units
        depth_m = (depth_raw.astype(np.float32) * self._depth_scale)
        return color_img, np.ascontiguousarray(depth_m)

    # -- metadata ----------------------------------------------------------

    @property
    def K(self) -> np.ndarray:
        return self._K

    @property
    def depth_scale(self) -> float:
        return self._depth_scale

    @property
    def size(self) -> tuple[int, int]:
        """``(height, width)``."""
        return (self._height, self._width)

    def meta(self) -> dict:
        """JSON-able camera metadata (intrinsics, size, depth scale, serial)."""
        return {
            "serial": self._serial,
            "K": self._K.tolist(),
            "width": self._width,
            "height": self._height,
            "depth_scale": self._depth_scale,
        }

    def close(self) -> None:
        try:
            self._pipeline.stop()
        except Exception:
            pass
