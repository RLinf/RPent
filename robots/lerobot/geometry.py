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


def robust_surface_centroid(
    depth: np.ndarray,
    K,
    T_base_cam,
    row: int,
    col: int,
    *,
    radius: int = 6,
    band: float = 0.015,
) -> dict:
    """Backproject a pixel neighborhood and return its dominant-surface median."""
    row, col = int(row), int(col)
    radius = max(0, int(radius))
    height, width = depth.shape[:2]
    if not (0 <= row < height and 0 <= col < width):
        return {"error": f"pixel ({row},{col}) out of bounds; image is {height}x{width}"}

    row_start, row_end = max(0, row - radius), min(height, row + radius + 1)
    col_start, col_end = max(0, col - radius), min(width, col + radius + 1)
    rows, cols = np.mgrid[row_start:row_end, col_start:col_end]
    depths = np.asarray(
        depth[row_start:row_end, col_start:col_end], dtype=np.float64
    )
    valid = np.isfinite(depths) & (depths > 0)
    if not np.any(valid):
        return {"error": f"no valid depth near ({row},{col}); pick another pixel"}

    median_depth = float(np.median(depths[valid]))
    surface = valid & (np.abs(depths - median_depth) <= band)
    if not np.any(surface):
        return {"error": f"no dominant surface depth near ({row},{col})"}

    surface_depths = depths[surface]
    surface_rows = rows[surface]
    surface_cols = cols[surface]
    intrinsics = np.asarray(K, dtype=np.float64)
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    camera_points = np.column_stack(
        (
            (surface_cols - cx) * surface_depths / fx,
            (surface_rows - cy) * surface_depths / fy,
            surface_depths,
        )
    )
    camera_point = np.median(camera_points, axis=0)
    out = {
        "pixel": [row, col],
        "radius": radius,
        "n_points": int(camera_points.shape[0]),
        "depth_m": round(median_depth, 4),
        "xyz_cam": [round(float(value), 4) for value in camera_point],
    }
    output_points = camera_points
    if T_base_cam is not None:
        output_points = transform_points(T_base_cam, camera_points)
        out["xyz"] = [
            round(float(value), 4)
            for value in np.median(output_points, axis=0)
        ]
    out["xy_spread_m"] = round(
        float(np.hypot(*output_points[:, :2].std(axis=0))), 4
    )
    return out


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


def solve_extrinsic_with_offset(
    cam_pts,
    tip_origins,
    tip_rotations,
    *,
    thresh_m: float = 0.015,
    offset_iters: int = 15,
    outer_iters: int = 3,
    seed: int = 0,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Joint fit of ``T_base_cam`` AND a constant gripper-local tip offset.

    The markerless routine detects the moving jaw's motion-blob centroid in the
    camera, but matches it against the FK tip-frame ORIGIN in the base frame.
    Those are not the same physical point: the centroid is displaced from the
    tip-frame origin by an (approximately) constant vector ``d`` expressed in
    the gripper LOCAL frame. In a free-orientation grid that displacement points
    a different way in the base frame at every pose, so a single rigid
    ``T_base_cam`` cannot absorb it -- it lands in the residual and is the main
    driver of the ~1 cm fit RMSE.

    This models the displacement explicitly. For pose ``i`` with tip-frame
    origin ``o_i`` and rotation ``R_i`` (base frame, from FK) and detected
    camera point ``c_i``::

        T_base_cam @ c_i  ==  o_i + R_i @ d

    It alternates two convex steps: (1) with ``d`` fixed, Kabsch-fit ``T`` to the
    corrected targets ``b_i = o_i + R_i @ d``; (2) with ``T`` fixed, solve the
    linear least squares ``R_i @ d = T @ c_i - o_i`` (closed form
    ``d = mean_i R_i^T (T c_i - o_i)`` since each ``R_i`` is orthonormal).
    Gross outliers (bad detections) are rejected with RANSAC. ``d`` is only
    identifiable when the tip ROTATIONS vary across poses; with (near-)constant
    orientation the displacement is indistinguishable from ``T``'s translation
    and the solver returns ``d ~= 0`` (reducing to the plain rigid fit).

    Args:
        cam_pts: ``(N, 3)`` detected points in the camera frame.
        tip_origins: ``(N, 3)`` FK tip-frame origins in the base frame.
        tip_rotations: ``(N, 3, 3)`` FK tip-frame rotations in the base frame.
        thresh_m: RANSAC inlier threshold (m) on the offset-corrected fit.
        offset_iters: max inner alternations per outer round.
        outer_iters: rounds of (refine offset -> re-select inliers).
        seed: RANSAC RNG seed.

    Returns:
        ``(T_4x4, inlier_rmse_m, inlier_mask, d_local)``.
    """
    cam = np.asarray(cam_pts, dtype=np.float64)
    o = np.asarray(tip_origins, dtype=np.float64)
    Rs = np.asarray(tip_rotations, dtype=np.float64)
    n = cam.shape[0]
    if n < 4 or Rs.shape != (n, 3, 3):
        T, rmse = kabsch_umeyama(cam, o)
        return T, rmse, np.ones(n, dtype=bool), np.zeros(3)

    # Stage 1: a loose RANSAC (tolerating the still-unknown offset) drops gross
    # outliers -- mis-detected tips -- before we estimate the offset.
    loose = max(thresh_m, 0.07)
    T, rmse, mask = ransac_kabsch(cam, o, thresh_m=loose, seed=seed)
    if int(mask.sum()) < 4:
        mask = np.ones(n, dtype=bool)

    d = np.zeros(3)
    for _ in range(outer_iters):
        # Alternate: solve the offset from residuals, refit T to the corrected
        # targets, until the offset stops moving.
        for _ in range(offset_iters):
            r = transform_points(T, cam[mask]) - o[mask]  # (M, 3)
            # mean_i R_i^T r_i  (R_i^T r_i via 'mba,mb->ma' since R_i is (i,j)=(row,col))
            d_new = np.einsum("mba,mb->ma", Rs[mask], r).mean(axis=0)
            b = o + np.einsum("nij,j->ni", Rs, d_new)  # o_i + R_i d
            T, _ = kabsch_umeyama(cam[mask], b[mask])
            if np.linalg.norm(d_new - d) < 1e-6:
                d = d_new
                break
            d = d_new
        # Re-select inliers with the tight threshold on the corrected targets.
        b = o + np.einsum("nij,j->ni", Rs, d)
        T, rmse, mask = ransac_kabsch(cam, b, thresh_m=thresh_m, seed=seed)
        if int(mask.sum()) < 4:
            mask = np.ones(n, dtype=bool)
            T, rmse = kabsch_umeyama(cam, b)
    return T, rmse, mask, d


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
        # Depth AT the centroid (small patch), so the returned (col, row, z) all
        # describe the SAME point. The blob spans a depth gradient under the
        # oblique scene view, so a whole-blob median would pair the centroid
        # pixel with some other pixel's depth and bias the back-projection.
        z = sample_depth_patch(depth_m, int(round(col)), int(round(row)), radius=3)
        if not np.isfinite(z):
            # Centroid fell on a depth dropout -- fall back to the blob median.
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
