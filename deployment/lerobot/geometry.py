"""Pure-numpy geometry helpers for SO101 scene-camera localization.

No hardware / lerobot imports — safe to unit-test offline. All transforms use
the convention ``T_a_b`` = pose of frame ``b`` in frame ``a`` so that
``p_a = T_a_b @ [p_b; 1]``. The world frame is the arm ``base_link``.
"""
from __future__ import annotations

import numpy as np


def backproject_pixel(K, col: float, row: float, depth_m: float) -> np.ndarray:
    """Backproject a pixel + metric depth into the camera frame (meters).

    Args:
        K: 3x3 pinhole intrinsics (of the stream the pixel was taken from;
            for our driver, depth is aligned to color so the color ``K``
            applies to both).
        col: pixel x (column, u).
        row: pixel y (row, v).
        depth_m: metric depth at ``(row, col)`` in meters.

    Returns:
        ``(3,)`` point ``[x, y, z]`` in the camera frame.
    """
    K = np.asarray(K, dtype=np.float64)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    z = float(depth_m)
    x = (float(col) - cx) * z / fx
    y = (float(row) - cy) * z / fy
    return np.array([x, y, z], dtype=np.float64)


def transform_points(T, pts) -> np.ndarray:
    """Apply a 4x4 homogeneous transform to a point or array of points.

    Args:
        T: 4x4 transform.
        pts: ``(3,)`` or ``(N, 3)`` points.

    Returns:
        Transformed points, same leading shape as ``pts``.
    """
    T = np.asarray(T, dtype=np.float64)
    pts = np.asarray(pts, dtype=np.float64)
    single = pts.ndim == 1
    p = np.atleast_2d(pts)
    ph = np.concatenate([p, np.ones((p.shape[0], 1))], axis=1)  # (N, 4)
    out = (ph @ T.T)[:, :3]
    return out[0] if single else out


def invert_transform(T) -> np.ndarray:
    """Invert a 4x4 rigid transform (R, t) -> (R^T, -R^T t)."""
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def kabsch_umeyama(src, dst) -> tuple[np.ndarray, float]:
    """Best-fit rigid transform mapping ``src`` -> ``dst`` (no scaling).

    Solves for ``T`` minimizing ``sum_i || T @ src_i - dst_i ||^2`` using the
    SVD (Kabsch/Umeyama) with a reflection guard.

    Args:
        src: ``(N, 3)`` source points (e.g. camera-frame).
        dst: ``(N, 3)`` destination points (e.g. base-frame).

    Returns:
        ``(T_4x4, rmse_meters)``.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError("src and dst must both be (N, 3) with matching N")
    if src.shape[0] < 3:
        raise ValueError("need at least 3 correspondences (4+ recommended)")

    c_src = src.mean(axis=0)
    c_dst = dst.mean(axis=0)
    s = src - c_src
    d = dst - c_dst

    H = s.T @ d
    U, _, Vt = np.linalg.svd(H)
    # Reflection guard: ensure a proper rotation (det = +1).
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ D @ U.T
    t = c_dst - R @ c_src

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    resid = transform_points(T, src) - dst
    rmse = float(np.sqrt((resid ** 2).sum(axis=1).mean()))
    return T, rmse


def ransac_kabsch(
    src,
    dst,
    *,
    thresh_m: float = 0.015,
    iters: int = 300,
    min_inliers: int = 4,
    seed: int = 0,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Robust rigid fit ``src`` -> ``dst`` with RANSAC over :func:`kabsch_umeyama`.

    Drops outlier correspondences (e.g. a mis-detected tip). Samples 4 points,
    fits, counts inliers within ``thresh_m``, keeps the best consensus, then
    refits on all inliers.

    Returns ``(T_4x4, inlier_rmse_m, inlier_mask)``. Falls back to a plain fit
    on all points if no good consensus is found.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = src.shape[0]
    if n < 4:
        T, rmse = kabsch_umeyama(src, dst)
        return T, rmse, np.ones(n, dtype=bool)

    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    best_mask = None
    best_count = 0
    for _ in range(iters):
        sample = rng.choice(idx, size=4, replace=False)
        try:
            T, _ = kabsch_umeyama(src[sample], dst[sample])
        except Exception:
            continue
        resid = np.linalg.norm(transform_points(T, src) - dst, axis=1)
        mask = resid < thresh_m
        count = int(mask.sum())
        if count > best_count:
            best_count = count
            best_mask = mask

    if best_mask is None or best_count < min_inliers:
        T, rmse = kabsch_umeyama(src, dst)
        return T, rmse, np.ones(n, dtype=bool)

    T, rmse = kabsch_umeyama(src[best_mask], dst[best_mask])
    return T, rmse, best_mask

def rotation_to_quat(R) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a quaternion ``[w, x, y, z]``."""
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z], dtype=np.float64)


def sample_depth_patch(depth_m, col: int, row: int, radius: int = 2) -> float:
    """Median of the valid (>0, finite) depths in a small patch around a pixel.

    Robustifies a single-pixel depth read (sensor noise / dropouts). Returns
    ``nan`` if no valid depth is found in the patch.
    """
    depth_m = np.asarray(depth_m, dtype=np.float64)
    h, w = depth_m.shape[:2]
    r0, r1 = max(0, row - radius), min(h, row + radius + 1)
    c0, c1 = max(0, col - radius), min(w, col + radius + 1)
    patch = depth_m[r0:r1, c0:c1].reshape(-1)
    valid = patch[np.isfinite(patch) & (patch > 0)]
    if valid.size == 0:
        return float("nan")
    return float(np.median(valid))


def detect_tip_pixel_by_motion(
    rgb_open,
    rgb_closed,
    depth_m,
    K,
    *,
    diff_thresh: int = 18,
    min_area: int = 40,
    max_area: int = 40000,
) -> dict | None:
    """Locate the gripper in the scene image via gripper-toggle motion.

    Given two scene frames that differ only by the gripper opening (arm held
    still), the changed pixels are the gripper fingers. Returns the centroid of
    the largest valid motion blob, the median depth over it, and the
    backprojected camera-frame point — or ``None`` if no usable blob is found.

    ``cv2`` is imported lazily so the rest of this module stays import-light.
    """
    import cv2

    a = cv2.cvtColor(np.asarray(rgb_open, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
    b = cv2.cvtColor(np.asarray(rgb_closed, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
    diff = cv2.GaussianBlur(cv2.absdiff(a, b), (5, 5), 0)
    _, mask = cv2.threshold(diff, int(diff_thresh), 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return None

    depth_m = np.asarray(depth_m, dtype=np.float64)
    # Largest component first (skip background label 0).
    for comp in np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1] + 1:
        area = int(stats[comp, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        col, row = float(centroids[comp][0]), float(centroids[comp][1])
        dvals = depth_m[labels == comp]
        dvals = dvals[np.isfinite(dvals) & (dvals > 0)]
        if dvals.size == 0:
            continue
        z = float(np.median(dvals))
        p_cam = backproject_pixel(K, col, row, z)
        return {
            "pixel": [row, col],
            "depth_m": z,
            "area": area,
            "xyz_cam": p_cam.tolist(),
        }
    return None
