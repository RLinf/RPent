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

"""Geometry, simulator conventions and localization capture contracts.

Synthetic renders below follow MuJoCo's rasterizer model: a
pixel (j, i) is sampled at its centre (j + 0.5, i + 0.5) in window
coordinates whose origin is a pixel corner. MuJoCo/OpenGL windows start at the
bottom row.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from rpent.tools.localization import localization as loc

NEAR, FAR = 0.05, 50.0


def _rotation(yaw: float, pitch: float) -> np.ndarray:
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    return rz @ rx


def _plane_hit(origin, directions, normal, offset):
    """Ray parameters where origin + t * direction meets normal . x = offset."""
    return (offset - origin @ normal) / (directions @ normal)


def _prism_depth(directions, far, slices=28):
    """Depth at which rays leave MuJoCo's skybox.

    MuJoCo draws a closed prism with ``slices`` side faces inscribed in a
    cylinder of radius and half-height 0.70 * far, centred on the camera with
    its axis along world z. ``directions`` are world ray directions whose
    optical-axis component is 1, so the exit parameter is the depth.
    """
    radius = 0.70 * far
    directions = np.asarray(directions, dtype=np.float64)
    middle = 2 * np.pi * (np.arange(slices) + 0.5) / slices
    facing = directions[..., :2] @ np.stack([np.cos(middle), np.sin(middle)])
    with np.errstate(divide="ignore"):
        side = np.where(facing > 0, radius * np.cos(np.pi / slices) / facing, np.inf)
        cap = radius / np.abs(directions[..., 2])
    return np.minimum(side.min(axis=-1), cap)


def _opencv_pose(forward, right, position=(0.0, 0.0, 0.0)):
    """Camera-to-world transform in OpenCV axes from world forward/right."""
    forward = np.asarray(forward, dtype=np.float64)
    forward /= np.linalg.norm(forward)
    right = np.asarray(right, dtype=np.float64)
    right = right - forward * (right @ forward)
    right /= np.linalg.norm(right)
    pose = np.eye(4)
    pose[:3, :3] = np.stack([right, np.cross(forward, right), forward], axis=1)
    pose[:3, 3] = position
    return pose


def _robosuite_K(fovy, height, width):
    focal = 0.5 * height / np.tan(np.deg2rad(fovy) / 2)
    return np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1.0]])


def _window(depth, near, far):
    """OpenGL depth-buffer value of metric depth (inverse of the decoder)."""
    return ((1 / near - 1 / depth) / (1 / near - 1 / far)).astype(np.float32)


def _top_down_rays(K, pose, shape):
    """World directions of pixel-centre rays of a top-down image."""
    rows, cols = np.indices(shape, dtype=np.float64)
    local = np.stack(
        [
            (cols + 0.5 - K[0, 2]) / K[0, 0],
            (rows + 0.5 - K[1, 2]) / K[1, 1],
            np.ones(shape),
        ],
        axis=-1,
    )
    return local @ pose[:3, :3].T


class _MujocoScene:
    """OpenGL-style render of the plane z = 0.1 seen by a MuJoCo camera."""

    def __init__(self, height=6, width=8, fovy=60.0):
        self.height, self.width, self.fovy = height, width, fovy
        self.position = np.array([0.3, -0.2, 1.4])
        # MuJoCo camera frame: x right, y up, looking along -z.
        self.xmat = _rotation(0.4, np.deg2rad(35))
        self.steps = 0

    def render(self, *, width, height, camera_name, depth):
        assert depth and camera_name == "front"
        focal = 0.5 * height / np.tan(np.deg2rad(self.fovy) / 2)
        rows, cols = np.indices((height, width), dtype=np.float64)  # row 0 = bottom
        local = np.stack(
            [
                (cols + 0.5 - width / 2) / focal,
                (rows + 0.5 - height / 2) / focal,
                -np.ones_like(rows),
            ],
            axis=-1,
        )
        directions = local @ self.xmat.T
        t = _plane_hit(self.position, directions, np.array([0, 0, 1.0]), 0.1)
        z = t  # local z component is -1, so t is the optical-axis depth
        z[0, 1] = _prism_depth(directions[0, 1], FAR)  # MuJoCo skybox
        window = (1 / NEAR - 1 / z) / (1 / NEAR - 1 / FAR)
        window[0, 0] = 1.0  # nothing drawn
        rgb = np.zeros((height, width, 3), np.uint8)
        rgb[..., 0] = rows.astype(np.uint8)  # encodes the OpenGL row
        return rgb, window.astype(np.float32)


def _render_frame(scene):
    rgb, buffer = scene.render(
        width=scene.width, height=scene.height, camera_name="front", depth=True
    )
    pose = np.eye(4)
    pose[:3, :3] = scene.xmat @ np.diag([1.0, -1.0, -1.0])
    pose[:3, 3] = scene.position
    return loc.LocalizationFrame.from_canonical(
        loc.normalize_robosuite_camera(
            "front",
            rgb,
            buffer,
            intrinsic_K=_robosuite_K(scene.fovy, scene.height, scene.width),
            extrinsic_cam2world=pose,
            near=NEAR,
            far=FAR,
            metadata={"environment": "test"},
        )
    )


def _frame(depth, K=None, T=None, rgb=None):
    depth = np.asarray(depth, dtype=np.float64)
    return loc.LocalizationFrame.from_canonical(
        {
            "camera": "cam",
            "rgb": np.zeros((*depth.shape, 3), np.uint8) if rgb is None else rgb,
            "depth_m": depth,
            "intrinsic_K": np.array([[2.0, 0, 1], [0, 2.0, 0.5], [0, 0, 1]])
            if K is None
            else K,
            "T_world_from_cv": np.eye(4) if T is None else T,
        }
    )


def test_pixel_centres_skew_and_rigid_pose_back_project_analytically():
    K = np.array([[2.0, 0.5, 1.0], [0, 4.0, 1.0], [0, 0, 1]])
    T = np.eye(4)
    T[:3, :3] = _rotation(np.pi / 2, 0)
    T[:3, 3] = [10, 20, 30]
    world, valid = loc.world_from_depth(np.full((3, 3), 2.0), K, T)
    assert world.dtype == np.float32 and valid.dtype == np.bool_ and valid.all()
    # The principal point maps to the optical axis.
    np.testing.assert_allclose(world[1, 1], [10, 20, 32])
    # u = 2, v = 2: y = (2 - 1) / 4 * 2 = 0.5, x = (2 - 1 - 0.5 * 0.25) / 2 * 2.
    camera = np.array([0.875, 0.5, 2.0])
    np.testing.assert_allclose(world[2, 2], T[:3, :3] @ camera + T[:3, 3], atol=1e-5)
    uv, depth = loc.project_to_image(world, K, T)
    rows, cols = np.indices((3, 3))
    np.testing.assert_allclose(uv[..., 0], cols, atol=1e-5)
    np.testing.assert_allclose(uv[..., 1], rows, atol=1e-5)
    np.testing.assert_allclose(depth, 2.0, atol=1e-5)


def test_invalid_depth_yields_nan_world_and_false_mask():
    depth = np.array([[np.nan, np.inf, -1.0, 0.0], [1.0, 2.0, 1e39, 3.0]])
    frame = _frame(depth.astype(np.float64), K=np.diag([1.0, 1.0, 1.0]))
    expected = np.array([[False, False, False, False], [True, True, False, True]])
    np.testing.assert_array_equal(frame.valid_mask, expected)
    assert np.isnan(frame.world_xyz[~expected]).all()
    assert np.isnan(frame.depth_m[~expected]).all()
    assert np.isfinite(frame.world_xyz[expected]).all()


@pytest.mark.parametrize(
    "field,value",
    [
        ("intrinsic_K", np.diag([0.0, 1.0, 1.0])),
        ("intrinsic_K", np.array([[1.0, 0, 0], [1.0, 1, 0], [0, 0, 1]])),
        ("T_world_from_cv", np.diag([1.0, 1.0, -1.0, 1.0])),
        ("T_world_from_cv", np.diag([2.0, 1.0, 1.0, 1.0])),
        ("T_world_from_cv", np.eye(3)),
        ("rgb", np.zeros((2, 3, 3), np.float32)),
        ("rgb", np.zeros((3, 3, 3), np.uint8)),
    ],
)
def test_bad_canonical_inputs_are_rejected(field, value):
    record = _frame(np.ones((2, 3))).canonical()
    record[field] = value
    with pytest.raises(ValueError):
        loc.LocalizationFrame.from_canonical(record)


def test_opengl_depth_buffer_decoding():
    window = np.array([[0.0, 0.5, 1.0, -0.1, np.nan]])
    depth = loc.depth_from_opengl_buffer(window, NEAR, FAR)
    np.testing.assert_allclose(depth[0, 0], NEAR)
    expected = NEAR / (1 - 0.5 * (1 - NEAR / FAR))
    np.testing.assert_allclose(depth[0, 1], expected, rtol=1e-6)
    assert np.isnan(depth[0, 2:]).all()
    for near, far in ((0, 1), (2, 1), (1, np.inf)):
        with pytest.raises(ValueError):
            loc.depth_from_opengl_buffer(window, near, far)


def test_mujoco_capture_flips_once_and_lands_on_the_rendered_surface():
    scene = _MujocoScene()
    frame = _render_frame(scene)
    H, W = scene.height, scene.width
    # Top-down row v shows OpenGL row H - 1 - v, encoded in the red channel.
    np.testing.assert_array_equal(frame.rgb[:, 0, 0], np.arange(H)[::-1])
    # OpenGL pixels (0, 0) (nothing drawn) and (0, 1) (skybox) are bottom-left.
    assert not frame.valid_mask[H - 1, :2].any()
    assert frame.valid_mask.sum() == H * W - 2
    assert frame.metadata["background"]["skybox_pixels"] == 1
    # Every valid point lies on the plane the synthetic renderer sampled.
    np.testing.assert_allclose(frame.world_xyz[frame.valid_mask][:, 2], 0.1, atol=1e-5)
    K = frame.intrinsic_K
    np.testing.assert_allclose(K[:2, 2], [W / 2 - 0.5, H / 2 - 0.5])
    native = frame.metadata["native"]
    assert native["intrinsic_K"][0][2] == W / 2
    np.testing.assert_allclose(frame.T_world_from_cv[:3, 3], scene.position)
    assert frame.metadata["environment"] == "test"
    assert frame.metadata["source"] == "robosuite"


def test_mujoco_skybox_depth_follows_the_camera_centred_cylinder():
    far = 10.0
    radius = 0.70 * far
    K = np.array([[2.0, 0, 1.0], [0, 2.0, 1.0], [0, 0, 1]])
    level = loc.mujoco_skybox_depth(
        K, _opencv_pose([1, 0, 0], [0, -1, 0], [5.0, -3.0, 2.0]), (3, 3), far
    )
    assert level[1, 1] == pytest.approx(radius)  # straight at the side
    assert level[1, 2] == pytest.approx(radius / np.hypot(1.0, 0.5))
    assert level[0, 1] == pytest.approx(radius)  # tilting up keeps the side distance
    # Identity OpenCV pose looks along world +z: every ray ends on the flat cap.
    np.testing.assert_allclose(
        loc.mujoco_skybox_depth(K, np.eye(4), (3, 3), far), radius
    )


@pytest.mark.parametrize(
    "fovy,pitch,below_half_far",
    [(45.0, 0.0, False), (100.0, 0.0, True), (75.0, -37.5, True), (45.0, 89.0, False)],
)
def test_skybox_is_background_for_wide_and_tilted_cameras(fovy, pitch, below_half_far):
    H = W = 32
    angle = np.deg2rad(pitch)
    pose = _opencv_pose(
        [np.cos(angle), 0.0, np.sin(angle)], [0, -1, 0], [0.3, -0.2, 1.4]
    )
    K = _robosuite_K(fovy, H, W)
    sky = _prism_depth(_top_down_rays(K, pose, (H, W)), FAR)
    # A fixed far / 2 limit keeps the image edges of wide or tilted cameras.
    assert (sky < 0.5 * FAR).any() == below_half_far
    record = loc.normalize_robosuite_camera(
        "probe",
        np.zeros((H, W, 3), np.uint8),
        _window(sky, NEAR, FAR)[::-1],
        intrinsic_K=K,
        extrinsic_cam2world=pose,
        near=NEAR,
        far=FAR,
    )
    assert np.isnan(record["depth_m"]).all()
    assert record["metadata"]["background"]["skybox_pixels"] == H * W


def test_geometry_in_front_of_the_skybox_stays_valid_at_any_distance():
    near, far, H, W = 0.01, 2.0, 16, 16
    K = _robosuite_K(45.0, H, W)
    pose = _opencv_pose([1, 0, 0], [0, -1, 0])
    wall = np.full((H, W), 1.2)  # head-on plane past far / 2, before the skybox
    window = _window(wall, near, far)
    window[0, 0] = 1.0  # nothing drawn at the bottom-left
    record = loc.normalize_robosuite_camera(
        "probe",
        np.zeros((H, W, 3), np.uint8),
        window,
        intrinsic_K=K,
        extrinsic_cam2world=pose,
        near=near,
        far=far,
    )
    depth = record["depth_m"]
    assert np.isnan(depth[H - 1, 0]) and np.isfinite(depth).sum() == H * W - 1
    np.testing.assert_allclose(depth[np.isfinite(depth)], 1.2, rtol=1e-4)
    # Past the skybox distance (1.4 m straight ahead) only a skybox can be drawn,
    # unless the model has none.
    past = _window(np.full((H, W), 1.39), near, far)
    common = {"intrinsic_K": K, "extrinsic_cam2world": pose, "near": near, "far": far}
    rgb = np.zeros((H, W, 3), np.uint8)
    assert np.isnan(
        loc.normalize_robosuite_camera("probe", rgb, past, **common)["depth_m"][8, 8]
    )
    without = loc.normalize_robosuite_camera("probe", rgb, past, skybox=False, **common)
    assert np.isfinite(without["depth_m"]).all()
    assert without["metadata"]["background"]["skybox_pixels"] == 0
    with pytest.raises(ValueError, match="skybox_slices"):
        loc.normalize_robosuite_camera("probe", rgb, past, skybox_slices=2, **common)


def test_fewer_skybox_slices_bring_the_side_faces_closer():
    H = W = 16
    K = _robosuite_K(90.0, H, W)
    pose = _opencv_pose([1, 0, 0], [0, -1, 0])
    sky = _prism_depth(_top_down_rays(K, pose, (H, W)), FAR, slices=8)
    common = {"intrinsic_K": K, "extrinsic_cam2world": pose, "near": NEAR, "far": FAR}
    rgb, window = np.zeros((H, W, 3), np.uint8), _window(sky, NEAR, FAR)[::-1]
    assumed = loc.normalize_robosuite_camera("probe", rgb, window, **common)
    assert np.isfinite(assumed["depth_m"]).any()  # the default 28 faces is wrong here
    actual = loc.normalize_robosuite_camera(
        "probe", rgb, window, skybox_slices=8, **common
    )
    assert np.isnan(actual["depth_m"]).all()


def _haze_depth(rays, camera, far, haze=0.3, floor=-1.0, steps=60):
    """Depth where rays meet MuJoCo's haze over the horizontal plane z = floor.

    In the plane frame the haze rises from radius (1 - haze) * R at the plane
    to R at the camera height (R = 0.70 * far). Solved by bisection on
    horizontal distance minus haze radius, up to where the ray meets the plane.
    """
    radius, height = 0.70 * far, camera[2] - floor
    down = -rays[..., 2]

    def gap(s):
        level = 1.0 - s * down / height  # 1 at the camera, 0 at the plane
        return s * np.hypot(rays[..., 0], rays[..., 1]) - radius * (
            1.0 - haze * (1.0 - level)
        )

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        to_floor = np.where(down > 0, height / down, np.inf)
        hits = (down > 0) & (gap(to_floor) > 0)
    low, high = np.zeros(rays.shape[:-1]), np.where(hits, to_floor, 0.0)
    for _ in range(steps):
        middle = 0.5 * (low + high)
        inside = gap(middle) < 0
        low, high = np.where(inside, middle, low), np.where(inside, high, middle)
    return np.where(hits, 0.5 * (low + high), np.inf)


def test_haze_over_an_infinite_plane_is_background_and_the_plane_stays_valid():
    H = W = 64
    camera = np.zeros(3)
    K = _robosuite_K(60.0, H, W)
    pose = _opencv_pose(
        [np.cos(np.deg2rad(-10)), 0.0, np.sin(np.deg2rad(-10))], [0, -1, 0], camera
    )
    rays = _top_down_rays(K, pose, (H, W))
    with np.errstate(divide="ignore"):
        floor = np.where(rays[..., 2] < 0, -1.0 / rays[..., 2], np.inf)
    haze = _haze_depth(rays, camera, FAR)
    sky = _prism_depth(rays, FAR)
    rendered = np.minimum(np.minimum(floor, haze), sky)  # draw order: floor, sky, haze
    assert (haze < sky).any() and (haze < floor).any()
    # Horizontal distance of each floor hit; the haze starts at (1 - 0.3) * R.
    reach = floor * np.hypot(rays[..., 0], rays[..., 1])
    plane = {"origin": [0, 0, -1.0], "rotation": np.eye(3), "haze": 0.3}
    common = {"intrinsic_K": K, "extrinsic_cam2world": pose, "near": NEAR, "far": FAR}
    rgb, window = np.zeros((H, W, 3), np.uint8), _window(rendered, NEAR, FAR)[::-1]
    record = loc.normalize_robosuite_camera(
        "probe", rgb, window, haze_plane=plane, **common
    )
    valid = np.isfinite(record["depth_m"])
    assert not valid[rendered < floor].any()  # haze and sky
    assert valid[reach < 0.8 * 0.7 * 0.70 * FAR].all()  # the plane before the haze
    assert record["metadata"]["background"]["haze_plane"] is True
    without = loc.normalize_robosuite_camera("probe", rgb, window, **common)
    assert np.isfinite(without["depth_m"])[haze < np.minimum(floor, sky)].any()


def test_canonical_round_trip_save_and_load_are_exact(tmp_path):
    scene = _MujocoScene()
    frame = _render_frame(scene)
    rebuilt = loc.LocalizationFrame.from_canonical(frame.canonical())
    np.testing.assert_array_equal(rebuilt.world_xyz, frame.world_xyz)
    path = loc.save_frame(tmp_path / "front.npz", frame)
    with pytest.raises(FileExistsError):
        loc.save_frame(path, frame)
    loaded = loc.load_frame(path)
    for name in (
        "rgb",
        "depth_m",
        "intrinsic_K",
        "T_world_from_cv",
        "world_xyz",
        "valid_mask",
    ):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(frame, name))
    assert loaded.metadata == frame.metadata and loaded.camera == "front"


def test_load_rejects_a_world_table_that_does_not_match_depth(tmp_path):
    frame = _frame(np.ones((2, 3)))
    tampered = loc.LocalizationFrame(
        **{**vars(frame), "world_xyz": frame.world_xyz + np.float32(0.01)}
    )
    path = loc.save_frame(tmp_path / "bad.npz", tampered)
    with pytest.raises(ValueError, match="does not match"):
        loc.load_frame(path)


def test_points_require_the_exact_rgb_grid_and_skip_invalid_pixels():
    frame = _frame(np.array([[1.0, np.nan, 2.0], [3.0, 4.0, 5.0]]))
    mask = np.array([[True, True, False], [False, False, True]])
    points = frame.points(mask)
    assert points.shape == (2, 3)
    np.testing.assert_array_equal(points, frame.world_xyz[[0, 1], [0, 2]])
    with pytest.raises(ValueError, match="map it back"):
        frame.points(np.ones((3, 2), bool))


def test_camera_requests_normalize_and_validate():
    assert loc.camera_requests("a") == [{"camera": "a", "height": None, "width": None}]
    expected = [
        {"camera": "a", "height": 4, "width": 5},
        {"camera": "b", "height": None, "width": None},
    ]
    assert loc.camera_requests([("a", 4, 5), {"camera": "b"}]) == expected
    assert loc.camera_requests((("a", 4, 5), {"camera": "b"})) == expected
    assert loc.camera_requests(("a", 4, 5)) == expected[:1]
    assert loc.camera_requests(("a", np.int64(4), np.int32(5))) == expected[:1]
    assert loc.camera_requests((("a", 4, 5),)) == expected[:1]
    assert loc.camera_requests(("b", None, None)) == expected[1:]
    assert loc.camera_requests(("a", "b", "c")) == [
        {"camera": name, "height": None, "width": None} for name in "abc"
    ]
    assert loc.camera_requests(("c", ("a", 4, 5), {"camera": "b"})) == [
        {"camera": "c", "height": None, "width": None},
        *expected,
    ]
    for bad in (
        None,
        3,
        {},
        [],
        (),
        [("a", 4, None)],
        [("a", 0, 4)],
        [("a", 4.5, 4)],
        ("a", 4, None),
        ("a", 0, 4),
        ("a", 4.5, 4),
        ("a", True, 4),
        [3],
        [""],
    ):
        with pytest.raises(ValueError):
            loc.camera_requests(bad)


class _RpcClient:
    def __init__(self, records):
        self.records = records
        self.calls = []
        self.requests = []

    def call(self, method, args=(), kwargs=None, *, timeout_s=None):
        self.calls.append(method)
        if method == "env.get_env_meta":
            return {"task": "offline"}
        assert method == "env.capture_localization"
        self.requests.append(kwargs["cameras"])
        return self.records[: len(kwargs["cameras"])]


def test_rpc_source_only_calls_read_only_methods():
    record = _frame(np.ones((2, 3))).canonical()
    client = _RpcClient([record] * 3)
    source = loc.RpcFrameSource(client)
    frames = loc.capture_many(source, ("a", "b", "c"))
    assert [frame.shape for frame in frames] == [(2, 3)] * 3
    assert [request["camera"] for request in client.requests[0]] == ["a", "b", "c"]
    assert source.env_meta() == {"task": "offline"}
    assert client.calls == ["env.capture_localization", "env.get_env_meta"]
    client.records = [record]
    with pytest.raises(ValueError, match="unexpected"):
        source.capture_frames(["a", "b"])


def test_import_uses_only_numpy_and_the_standard_library():
    root = Path(__file__).resolve().parents[5]
    code = (
        f"import sys; sys.path.insert(0, {str(root)!r})\n"
        "import rpent.tools.localization.localization\n"
        "heavy = ['torch', 'rlinf', 'mujoco', 'robosuite', 'sapien', 'pydantic',\n"
        "         'pydantic_ai', 'rpent.tools.base', 'rpent.tools.toolkit',\n"
        "         'rpent.robots', 'rpent.utils', 'robots']\n"
        "loaded = [name for name in heavy if name in sys.modules]\n"
        "assert not loaded, loaded"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
