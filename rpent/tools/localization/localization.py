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

"""Pixel-aligned world coordinates for simulator RGB-D cameras.

Capture records follow one canonical frame convention:

- ``rgb``: uint8 ``(H, W, 3)``. Row 0 is the image top and column 0 the left.
  Pixel ``(u, v)`` is array element ``[v, u]``.
- ``depth_m``: float32 ``(H, W)`` distance along the optical axis (camera Z)
  in metres; NaN where no surface was rendered.
- ``intrinsic_K``: OpenCV pinhole matrix for this image grid. Integer
  coordinates are pixel centres; simulators report the principal point in
  pixel-edge coordinates, so normalizers subtract 0.5 pixel.
- ``T_world_from_cv``: 4x4 transform from OpenCV camera axes (x right, y down,
  z forward) to the environment's own world frame.
- ``world_xyz``: float32 ``(H, W, 3)``; NaN at invalid pixels.
- ``valid_mask``: bool ``(H, W)``; finite positive depth and a finite world
  point. It describes geometry, not object segmentation.

Native renders are normalized once, at the owner of the simulator.
Capture sources never reset, step, close or save the borrowed environment.
The module imports only NumPy and the standard library.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TypeAlias

import numpy as np

#: One camera name, sized request, or mapping accepted by camera_requests.
CameraRequest: TypeAlias = (
    str
    | tuple[str, int | np.integer[Any] | None, int | np.integer[Any] | None]
    | Mapping[str, Any]
)
#: One camera request or an ordered sequence of requests.
CameraRequests: TypeAlias = CameraRequest | Sequence[CameraRequest]

#: MuJoCo draws its skybox as a closed cylinder centred on the camera, axis
#: along world z, radius and half-height ``0.70 * far`` (``skyboxdst`` in
#: render_gl3.c), with ``vis.quality.numslices`` flat side faces.
MUJOCO_SKYBOX_SCALE = 0.70

# Depth at or beyond this fraction of the skybox's nearest possible depth is
# background; the slack covers depth sample offsets and buffer quantization.
_SKYBOX_MARGIN = 0.98
FRAME_FORMAT_VERSION = 1

_CANONICAL_FIELDS = ("camera", "rgb", "depth_m", "intrinsic_K", "T_world_from_cv")


def _intrinsic(intrinsic_K: Any) -> np.ndarray:
    matrix = np.array(intrinsic_K, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("intrinsic_K must be a finite 3x3 matrix")
    if (
        matrix[0, 0] <= 0
        or matrix[1, 1] <= 0
        or matrix[1, 0] != 0
        or not np.allclose(matrix[2], [0.0, 0.0, 1.0], rtol=0, atol=1e-9)
    ):
        raise ValueError("intrinsic_K must be an upper-triangular pinhole matrix")
    return matrix


def _rigid(transform: Any) -> np.ndarray:
    matrix = np.array(transform, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("T_world_from_cv must be a finite 4x4 matrix")
    rotation = matrix[:3, :3]
    if not (
        np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], rtol=0, atol=1e-9)
        and np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-5)
        and np.isclose(np.linalg.det(rotation), 1.0, rtol=0, atol=1e-5)
    ):
        raise ValueError("T_world_from_cv must be a rigid transform (proper rotation)")
    return matrix


def _depth_image(depth: Any) -> np.ndarray:
    array = np.asarray(depth)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2 or min(array.shape) == 0:
        raise ValueError(f"depth must have nonempty shape (H, W), got {array.shape}")
    return array.astype(np.float64)


def _plain(value: Any) -> Any:
    """Return JSON-compatible built-in values for metadata."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def world_from_depth(
    depth_m: Any, intrinsic_K: Any, T_world_from_cv: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project canonical depth into the environment's world frame.

    Args:
        depth_m: Optical-axis depth in metres, shape ``(H, W)``, row 0 at the
            image top. Non-finite and non-positive values are invalid.
        intrinsic_K: OpenCV pinhole matrix for this ``(H, W)`` grid, with
            integer coordinates at pixel centres.
        T_world_from_cv: Rigid transform from OpenCV camera axes to world.

    Returns:
        ``(world_xyz, valid_mask)``: float32 ``(H, W, 3)`` with three NaNs at
        invalid pixels, and the matching boolean ``(H, W)`` mask.
    """
    depth = _depth_image(depth_m)
    intrinsic = _intrinsic(intrinsic_K)
    transform = _rigid(T_world_from_cv)
    valid = np.isfinite(depth) & (depth > 0)
    z = np.where(valid, depth, np.nan)
    rows, cols = np.indices(depth.shape, dtype=np.float64)
    fx, skew, cx = intrinsic[0]
    fy, cy = intrinsic[1, 1], intrinsic[1, 2]
    y = (rows - cy) / fy * z
    x = (cols - cx - skew * (rows - cy) / fy) / fx * z
    world = np.empty((*depth.shape, 3), dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        for axis in range(3):
            world[..., axis] = (
                transform[axis, 0] * x
                + transform[axis, 1] * y
                + transform[axis, 2] * z
                + transform[axis, 3]
            )
    valid &= np.isfinite(world).all(axis=-1)
    world[~valid] = np.nan
    return world, valid


def project_to_image(
    points_world: Any, intrinsic_K: Any, T_world_from_cv: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Project world points into canonical pixel coordinates.

    Args:
        points_world: World points with shape ``(..., 3)``.
        intrinsic_K: OpenCV pinhole matrix of the target image.
        T_world_from_cv: Rigid transform from OpenCV camera axes to world.

    Returns:
        ``(uv, depth)``: float64 ``(..., 2)`` pixel coordinates ``(u, v)``
        (column, row; integers are pixel centres) and ``(...)`` camera Z.
        Points behind the camera keep their negative depth.

    Raises:
        ValueError: If points do not have a trailing dimension of 3.
    """
    points = np.asarray(points_world, dtype=np.float64)
    if points.ndim < 1 or points.shape[-1] != 3:
        raise ValueError("points_world must have shape (..., 3)")
    intrinsic = _intrinsic(intrinsic_K)
    transform = _rigid(T_world_from_cv)
    camera = (points - transform[:3, 3]) @ transform[:3, :3]
    depth = camera[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        homogeneous = camera @ intrinsic.T
        uv = homogeneous[..., :2] / homogeneous[..., 2:3]
    return uv, depth


@dataclass(frozen=True, eq=False)
class LocalizationFrame:
    """One canonical camera frame and its pixel-aligned world table.

    Attributes:
        camera: Camera name used by the environment.
        rgb: uint8 ``(H, W, 3)`` image, row 0 at the top.
        depth_m: float32 ``(H, W)`` optical-axis depth, NaN where invalid.
        intrinsic_K: OpenCV pinhole matrix with pixel-centre coordinates.
        T_world_from_cv: Rigid OpenCV-camera-to-world transform.
        world_xyz: float32 ``(H, W, 3)`` world points, NaN where invalid.
        valid_mask: Geometric validity, shape ``(H, W)``.
        metadata: JSON-compatible capture details and native calibration.
    """

    camera: str
    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsic_K: np.ndarray
    T_world_from_cv: np.ndarray
    world_xyz: np.ndarray
    valid_mask: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_canonical(cls, record: Mapping[str, Any]) -> LocalizationFrame:
        """Build a frame from canonical inputs and compute its world table.

        Args:
            record: Mapping with ``camera``, ``rgb``, ``depth_m``,
                ``intrinsic_K``, ``T_world_from_cv`` and optional
                ``metadata``, already in the canonical convention.

        Returns:
            The validated frame.

        Raises:
            ValueError: If fields are missing or RGB and depth disagree.
        """
        missing = [name for name in _CANONICAL_FIELDS if name not in record]
        if missing:
            raise ValueError(f"canonical frame is missing {missing}")
        rgb = np.asarray(record["rgb"])
        # The stored float32 depth is the source of truth for the world table,
        # so a frame rebuilt from its canonical inputs is bit-identical.
        with np.errstate(over="ignore"):
            depth = _depth_image(record["depth_m"]).astype(np.float32)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError("rgb must be uint8 with shape (H, W, 3)")
        if rgb.shape[:2] != depth.shape:
            raise ValueError(
                f"rgb {rgb.shape[:2]} and depth {depth.shape} use different grids"
            )
        intrinsic = _intrinsic(record["intrinsic_K"])
        transform = _rigid(record["T_world_from_cv"])
        world, valid = world_from_depth(depth, intrinsic, transform)
        return cls(
            camera=str(record["camera"]),
            rgb=np.ascontiguousarray(rgb),
            depth_m=np.where(valid, depth, np.float32(np.nan)),
            intrinsic_K=intrinsic,
            T_world_from_cv=transform,
            world_xyz=world,
            valid_mask=valid,
            metadata=_plain(dict(record.get("metadata") or {})),
        )

    @property
    def shape(self) -> tuple[int, int]:
        """Return the image grid as ``(H, W)``."""
        return self.valid_mask.shape

    def describe(self) -> dict[str, Any]:
        """Return JSON-compatible calibration and capture details.

        Returns:
            Camera, grid, canonical calibration, pixel convention, valid pixel
            count and the frame metadata (including native calibration).
        """
        return _plain(
            {
                **self.metadata,
                "camera": self.camera,
                "height": self.shape[0],
                "width": self.shape[1],
                "intrinsic_K": self.intrinsic_K,
                "T_world_from_cv": self.T_world_from_cv,
                "pixel_convention": (
                    "pixel (u, v) is array element [v, u]; row 0 is the image top; "
                    "integer coordinates are pixel centres"
                ),
                "depth": "optical-axis Z in metres",
                "valid_pixels": int(self.valid_mask.sum()),
            }
        )

    def canonical(self) -> dict[str, Any]:
        """Return the canonical inputs, e.g. for RPC transport.

        Returns:
            Mapping accepted by :meth:`from_canonical`; derived arrays are
            omitted because they are recomputed deterministically.
        """
        return {
            "camera": self.camera,
            "rgb": self.rgb,
            "depth_m": self.depth_m,
            "intrinsic_K": self.intrinsic_K,
            "T_world_from_cv": self.T_world_from_cv,
            "metadata": self.metadata,
        }

    def points(self, mask: Any) -> np.ndarray:
        """Return world points of ``mask`` pixels that have valid geometry.

        Args:
            mask: Boolean object mask on this frame's exact ``(H, W)`` grid,
                e.g. a SAM3 mask computed on :attr:`rgb`. Masks from resized,
                cropped or flipped images must be mapped back first.

        Returns:
            float32 ``(N, 3)`` world points in row-major pixel order.

        Raises:
            ValueError: If the mask grid differs from the frame grid.
        """
        selected = np.asarray(mask, dtype=bool)
        if selected.shape != self.shape:
            raise ValueError(
                f"mask shape {selected.shape} differs from frame grid {self.shape}; "
                "map it back to this RGB image explicitly"
            )
        return self.world_xyz[selected & self.valid_mask]


def save_frame(path: str | Path, frame: LocalizationFrame) -> Path:
    """Write one frame to a new compressed ``.npz`` file.

    Args:
        path: Destination; an existing file is never overwritten.
        frame: Frame to store, including its derived world table and mask.

    Returns:
        The written path.
    """
    destination = Path(path)
    with destination.open("xb") as file:
        np.savez_compressed(
            file,
            format_version=np.array(FRAME_FORMAT_VERSION),
            camera=np.array(frame.camera),
            rgb=frame.rgb,
            depth_m=frame.depth_m,
            intrinsic_K=frame.intrinsic_K,
            T_world_from_cv=frame.T_world_from_cv,
            world_xyz=frame.world_xyz,
            valid_mask=frame.valid_mask,
            metadata=np.array(json.dumps(frame.metadata, sort_keys=True)),
        )
    return destination


def load_frame(path: str | Path) -> LocalizationFrame:
    """Read a frame written by :func:`save_frame` and check its consistency.

    Args:
        path: ``.npz`` file written by :func:`save_frame`.

    Returns:
        The stored frame, with the stored world table and mask.

    Raises:
        ValueError: If the format is unknown or the stored world table does
            not match the stored depth and calibration.
    """
    with np.load(Path(path), allow_pickle=False) as archive:
        if int(archive["format_version"]) != FRAME_FORMAT_VERSION:
            raise ValueError(f"unsupported frame format in {path}")
        stored = {name: archive[name] for name in archive.files}
    frame = LocalizationFrame.from_canonical(
        {
            "camera": str(stored["camera"]),
            "rgb": stored["rgb"],
            "depth_m": stored["depth_m"],
            "intrinsic_K": stored["intrinsic_K"],
            "T_world_from_cv": stored["T_world_from_cv"],
            "metadata": json.loads(str(stored["metadata"])),
        }
    )
    if not np.array_equal(frame.valid_mask, stored["valid_mask"]) or not np.allclose(
        frame.world_xyz, stored["world_xyz"], rtol=0, atol=1e-6, equal_nan=True
    ):
        raise ValueError(f"stored world table in {path} does not match its depth")
    return LocalizationFrame(
        camera=frame.camera,
        rgb=frame.rgb,
        depth_m=frame.depth_m,
        intrinsic_K=frame.intrinsic_K,
        T_world_from_cv=frame.T_world_from_cv,
        world_xyz=stored["world_xyz"],
        valid_mask=stored["valid_mask"],
        metadata=frame.metadata,
    )


# ---------------------------------------------------------------------------
# Native simulator conventions
# ---------------------------------------------------------------------------


def depth_from_opengl_buffer(buffer: Any, near: float, far: float) -> np.ndarray:
    """Decode a perspective OpenGL depth buffer into optical-axis metres.

    Args:
        buffer: Window depth in ``[0, 1)``, as returned by MuJoCo's
            ``mjr_readPixels``. The clear value 1 (nothing drawn), values
            outside the range and non-finite values become NaN.
        near: Near clipping distance in metres.
        far: Far clipping distance in metres.

    Returns:
        float32 depth with the input orientation.

    Raises:
        ValueError: If the clipping planes are invalid.
    """
    near, far = float(near), float(far)
    if not np.isfinite([near, far]).all() or not 0 < near < far:
        raise ValueError("depth clipping planes must satisfy 0 < near < far")
    window = _depth_image(buffer)
    usable = np.isfinite(window) & (window >= 0) & (window < 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        metric = near / (1.0 - np.where(usable, window, np.nan) * (1.0 - near / far))
    return metric.astype(np.float32)


def mujoco_skybox_depth(
    intrinsic_K: Any,
    T_world_from_cv: Any,
    shape: tuple[int, int],
    far: float,
    *,
    haze: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Return the optical-axis depth of MuJoCo's skybox pass per pixel.

    The skybox is centred on the camera with its axis along world z, so the
    depth depends on each pixel ray's direction and the far plane only. Over
    an infinite plane MuJoCo also draws haze, in the plane's frame a frustum
    from radius ``(1 - haze) * R`` at the plane to ``R`` at the camera height,
    where ``R = MUJOCO_SKYBOX_SCALE * far``. Below the plane its cone is
    continued; that part lies behind the opaque plane, and it keeps the
    surface continuous where the haze meets the plane.

    Args:
        intrinsic_K: Pixel-centre (canonical) intrinsics of the top-down image.
        T_world_from_cv: OpenCV-camera-to-world transform of that image.
        shape: Image grid ``(H, W)``.
        far: Far clipping distance in metres.
        haze: ``origin``, ``rotation`` (plane axes as columns, normal last)
            and ``haze`` (``vis.map.haze``) of the hazed plane, if any.

    Returns:
        float64 ``(H, W)`` depth where each pixel-centre ray first meets the
        skybox cylinder or the haze cone. The drawn faces are inscribed in
        these round surfaces, so the rendered depth lies between
        ``cos(pi / numslices)`` times this value and this value.
    """
    intrinsic = _intrinsic(intrinsic_K)
    transform = _rigid(T_world_from_cv)
    rows, cols = np.indices(tuple(shape), dtype=np.float64)
    fx, skew, cx = intrinsic[0]
    fy, cy = intrinsic[1, 1], intrinsic[1, 2]
    y = (rows - cy) / fy
    x = (cols - cx - skew * y) / fx
    rays = np.stack([x, y, np.ones_like(x)], axis=-1) @ transform[:3, :3].T
    radius = MUJOCO_SKYBOX_SCALE * float(far)
    with np.errstate(divide="ignore"):
        side = radius / np.hypot(rays[..., 0], rays[..., 1])
        cap = radius / np.abs(rays[..., 2])
    depth = np.minimum(side, cap)
    if haze is None:
        return depth
    axes = np.asarray(haze["rotation"], dtype=np.float64).reshape(3, 3)
    origin = np.asarray(haze["origin"], dtype=np.float64)
    elevation = float((transform[:3, 3] - origin) @ axes[:, 2]) / radius
    if elevation <= 0:  # MuJoCo draws no haze from below the plane
        return depth
    local = rays @ axes
    down = -local[..., 2]
    slope = float(haze["haze"]) / elevation
    with np.errstate(divide="ignore", invalid="ignore"):
        along = radius / (np.hypot(local[..., 0], local[..., 1]) + slope * down)
    return np.where(down > 0, np.minimum(depth, along), depth)


def normalize_robosuite_camera(
    camera: str,
    rgb: Any,
    depth_buffer: Any,
    *,
    intrinsic_K: Any,
    extrinsic_cam2world: Any,
    near: float,
    far: float,
    skybox: bool = True,
    skybox_slices: int = 28,
    haze_plane: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert one robosuite/MuJoCo render into canonical inputs.

    The inputs follow ``MjSim.render(..., depth=True)`` and
    ``robosuite.utils.camera_utils``: rows bottom-up, an OpenGL depth buffer,
    ``K`` with its principal point in pixel-edge coordinates of the top-down
    image, and a camera pose already expressed in OpenCV axes.

    Pixels where nothing was drawn are invalid. The skybox (and its haze over
    an infinite plane) is drawn on the surface described by
    :func:`mujoco_skybox_depth` and every visible surface lies in front of
    it, so depth at or beyond 98% of that surface's nearest possible depth
    within the pixel is background. Unlike a fixed depth limit this holds for
    any field of view or camera tilt and keeps real geometry at any distance
    in front of the sky.

    Args:
        camera: MuJoCo camera name.
        rgb: Rendered RGB, bottom row first.
        depth_buffer: Rendered depth buffer, bottom row first.
        intrinsic_K: robosuite intrinsic matrix for this resolution.
        extrinsic_cam2world: robosuite camera-to-world matrix (OpenCV axes).
        near: Near clipping distance in metres.
        far: Far clipping distance in metres.
        skybox: Whether a skybox texture is present and skybox rendering is
            enabled. Otherwise no skybox depth limit applies.
        skybox_slices: ``model.vis.quality.numslices``, the number of side
            faces of the skybox (MuJoCo default 28).
        haze_plane: The plane MuJoCo draws haze over, as accepted by
            :func:`mujoco_skybox_depth`; ``None`` when there is none.
        metadata: Extra JSON-compatible fields to record.

    Returns:
        Canonical inputs accepted by :meth:`LocalizationFrame.from_canonical`.

    Raises:
        ValueError: If ``skybox_slices`` is below 3.
    """
    slices = int(skybox_slices)
    if slices < 3:
        raise ValueError("skybox_slices must be at least 3")
    intrinsic = _intrinsic(intrinsic_K)
    transform = _rigid(extrinsic_cam2world)
    image = np.asarray(rgb)
    depth = np.ascontiguousarray(
        depth_from_opengl_buffer(depth_buffer, near, far)[::-1]
    )
    canonical_K = intrinsic.copy()
    canonical_K[0, 2] -= 0.5
    canonical_K[1, 2] -= 0.5
    radius = MUJOCO_SKYBOX_SCALE * float(far)
    fraction = _SKYBOX_MARGIN * float(np.cos(np.pi / slices))
    background = np.zeros(depth.shape, dtype=bool)
    if skybox:
        # The depth sample lies somewhere inside the pixel: bound the surface by
        # its nearest point over the pixel's four corners.
        corner_K = canonical_K.copy()
        corner_K[:2, 2] += 0.5
        height, width = depth.shape
        corners = mujoco_skybox_depth(
            corner_K, transform, (height + 1, width + 1), far, haze=haze_plane
        )
        surface = np.minimum(
            np.minimum(corners[:-1, :-1], corners[:-1, 1:]),
            np.minimum(corners[1:, :-1], corners[1:, 1:]),
        )
        with np.errstate(invalid="ignore"):
            background = depth >= fraction * surface
        depth[background] = np.nan
        rule = "depth >= fraction * nearest skybox depth over the pixel"
    else:
        rule = "no skybox rendered: only pixels with nothing drawn"
    return {
        "camera": camera,
        "rgb": np.ascontiguousarray(image[::-1]),
        "depth_m": depth,
        "intrinsic_K": canonical_K,
        "T_world_from_cv": transform,
        "metadata": _plain(
            {
                **dict(metadata or {}),
                "source": "robosuite",
                "height": int(depth.shape[0]),
                "width": int(depth.shape[1]),
                "conversions": {
                    "rows": "flipped once from OpenGL bottom-up to top-down",
                    "depth": "OpenGL depth buffer decoded with near/far; pixels "
                    "with nothing drawn or showing the skybox are invalid",
                    "principal_point": "shifted by -0.5 px to pixel-centre coordinates",
                    "camera_axes": "robosuite pose already uses OpenCV axes",
                },
                "background": {
                    "rule": rule,
                    "skybox_radius_m": radius if skybox else None,
                    "skybox_slices": slices,
                    "haze_plane": bool(skybox and haze_plane is not None),
                    "fraction": fraction if skybox else None,
                    "skybox_pixels": int(background.sum()),
                },
                "native": {
                    "intrinsic_K": intrinsic,
                    "extrinsic_cam2world": np.asarray(extrinsic_cam2world),
                    "depth_near": near,
                    "depth_far": far,
                },
            }
        ),
    }


# ---------------------------------------------------------------------------
# Capture sources
# ---------------------------------------------------------------------------


def camera_requests(cameras: CameraRequests) -> list[dict[str, Any]]:
    """Normalize camera requests to ``{"camera", "height", "width"}`` dicts.

    Args:
        cameras: One camera or a sequence (list or tuple) of cameras. Each item
            is a name, a ``(name, height, width)`` tuple, or a mapping with those
            keys. ``None`` sizes select the environment default or native size.

    Returns:
        Requests in the given order.

    Raises:
        ValueError: If a request is malformed.
    """
    sized_request = (
        isinstance(cameras, tuple)
        and len(cameras) == 3
        and isinstance(cameras[0], str)
        and all(
            size is None or isinstance(size, (int, np.integer)) for size in cameras[1:]
        )
    )
    items = (
        [cameras] if isinstance(cameras, (str, Mapping)) or sized_request else cameras
    )
    if not isinstance(items, Sequence):
        raise ValueError("cameras must be a camera request or an ordered sequence")
    requests = []
    for item in items:
        if isinstance(item, str):
            name, height, width = item, None, None
        elif isinstance(item, Mapping):
            name, height, width = (
                item.get("camera"),
                item.get("height"),
                item.get("width"),
            )
        elif isinstance(item, tuple) and len(item) == 3:
            name, height, width = item
        else:
            raise ValueError(f"invalid camera request: {item!r}")
        if not isinstance(name, str) or not name:
            raise ValueError(f"camera name must be a nonempty string: {item!r}")
        if (height is None) != (width is None):
            raise ValueError(f"set both height and width or neither: {item!r}")
        for size in (height, width):
            if size is not None and (
                isinstance(size, bool) or not isinstance(size, (int, np.integer))
            ):
                raise ValueError(f"image sizes must be integers: {item!r}")
            if size is not None and size <= 0:
                raise ValueError(f"image sizes must be positive: {item!r}")
        requests.append(
            {
                "camera": name,
                "height": None if height is None else int(height),
                "width": None if width is None else int(width),
            }
        )
    if not requests:
        raise ValueError("at least one camera is required")
    return requests


class FrameSource(Protocol):
    """Anything that captures canonical frames from one environment instant."""

    def capture_frames(self, cameras: CameraRequests) -> list[LocalizationFrame]:
        """Capture all requested cameras from one unchanged environment state.

        Args:
            cameras: Requests accepted by :func:`camera_requests`.

        Returns:
            Canonical frames in request order, without advancing the environment.
        """
        ...


def capture(
    source: FrameSource,
    camera: str,
    *,
    height: int | None = None,
    width: int | None = None,
) -> LocalizationFrame:
    """Capture one camera through any :class:`FrameSource`.

    Args:
        source: RPent env client or :class:`RpcFrameSource` implementing
            the :class:`FrameSource` contract.
        camera: Environment camera name.
        height: Requested height; ``None`` uses the default or native size.
        width: Requested width; set together with ``height``.

    Returns:
        The captured frame.
    """
    return source.capture_frames(
        [{"camera": camera, "height": height, "width": width}]
    )[0]


def capture_many(
    source: FrameSource, cameras: CameraRequests
) -> list[LocalizationFrame]:
    """Capture several cameras from the same environment state.

    Args:
        source: Any :class:`FrameSource`.
        cameras: Requests accepted by :func:`camera_requests`.

    Returns:
        Frames in request order.
    """
    return source.capture_frames(camera_requests(cameras))


class RpcFrameSource:
    """Capture frames from a running RPent env server without an EnvClient.

    Constructing an EnvClient resets its environment; this source only calls
    ``env.capture_localization`` and ``env.get_env_meta``, which do not change
    the environment state.
    """

    def __init__(self, endpoint_or_client: Any, *, timeout_s: float = 120.0) -> None:
        """Attach to a server.

        Args:
            endpoint_or_client: ``[http|socket]://host:port`` or an RPC client
                exposing ``call(method, kwargs=..., timeout_s=...)``.
            timeout_s: Timeout for each capture call.
        """
        if isinstance(endpoint_or_client, str):
            from rpent.utils.rpc import make_rpc_client

            endpoint_or_client = make_rpc_client(endpoint_or_client)
        self._client = endpoint_or_client
        self._timeout_s = float(timeout_s)

    def env_meta(self) -> dict[str, Any]:
        """Return the identity the server was launched with.

        Returns:
            Environment launch metadata reported by ``env.get_env_meta``.
        """
        return self._client.call("env.get_env_meta", timeout_s=self._timeout_s)

    def capture_frames(self, cameras: CameraRequests) -> list[LocalizationFrame]:
        """Capture requested cameras in one server call.

        Args:
            cameras: Requests accepted by :func:`camera_requests`.

        Returns:
            Canonical frames in request order, without resetting or stepping.
        """
        return frames_from_rpc(self._client, cameras, timeout_s=self._timeout_s)


def frames_from_rpc(
    client: Any, cameras: CameraRequests, *, timeout_s: float = 120.0
) -> list[LocalizationFrame]:
    """Call ``env.capture_localization`` and build frames from its records.

    Args:
        client: RPC client exposing ``call``.
        cameras: Requests accepted by :func:`camera_requests`.
        timeout_s: Call timeout in seconds.

    Returns:
        Frames in request order.

    Raises:
        ValueError: If the server returns a different number of frames.
    """
    requests = camera_requests(cameras)
    records = client.call(
        "env.capture_localization",
        kwargs={"cameras": requests},
        timeout_s=timeout_s,
    )
    if not isinstance(records, list) or len(records) != len(requests):
        raise ValueError("env.capture_localization returned an unexpected payload")
    return [LocalizationFrame.from_canonical(record) for record in records]
