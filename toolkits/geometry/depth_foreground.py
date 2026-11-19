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

"""Recover an object's connected RGB-D foreground against a locally measured support plane."""

import re

import numpy as np
from scipy.ndimage import binary_erosion, label, maximum_filter, minimum_filter


def local_depth_range(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Return the measured 3x3 depth range at each valid pixel."""
    if valid.all():
        return maximum_filter(depth, 3) - minimum_filter(depth, 3)
    high = maximum_filter(np.where(valid, depth, -np.inf), 3)
    low = minimum_filter(np.where(valid, depth, np.inf), 3)
    spread = high - low
    spread[~valid] = np.nan
    return spread


def described_lightness(query):
    """Use an explicit light/dark appearance only when a nearby component competes."""
    phrase = re.split(r"\bwith\b|,|\bexcluding\b", query.lower(), maxsplit=1)[0]
    if re.search(r"\blight[- ]colou?red\b|\bpale\b", phrase):
        return 1
    if re.search(r"\bdark[- ]colou?red\b", phrase):
        return -1
    colors = set(
        re.findall(
            r"\b(black|white|gr[ae]y|red|green|blue|yellow|orange|pink|brown|purple)\b",
            phrase,
        )
    )
    return 1 if colors == {"white"} else -1 if colors == {"black"} else 0


def foreground(world, valid, up, cam, rc, *, rgb=None, lightness=0):
    h, w = valid.shape
    r, c = rc
    if up is None or cam is None:
        return None, {"reason": "calibration_or_up_unavailable"}
    # Estimate the support height using smooth, nearly horizontal neighbors.
    half = max(40, int(0.28 * min(w, h)))
    roi = np.s_[
        max(0, r - half) : min(h, r + half), max(0, c - half) : min(w, c + half)
    ]
    patch = world[roi][::5, ::5]
    if min(patch.shape[:2]) < 2:
        return None, {"reason": "insufficient_support_pixels"}
    height = np.where(valid[roi][::5, ::5], patch @ up, np.nan)
    dr, dc = np.gradient(height)
    flats = valid[roi][::5, ::5] & (abs(dc) + abs(dr) < 0.003)
    if flats.sum() < 40:
        return None, {"reason": "horizontal_support_not_visible"}
    anchor = world[r, c] @ up if valid[r, c] else np.nan
    if np.isfinite(anchor):
        flats &= height <= anchor + 0.004
    zz = height[flats]
    if len(zz) < 25:
        return None, {"reason": "no_plane_below_anchor"}
    bins = np.floor(zz / 0.002).astype(np.int64)
    keys, counts = np.unique(bins, return_counts=True)
    i = counts.argmax()
    z0 = np.median(zz[abs(zz - (keys[i] + 0.5) * 0.002) < 0.0025])
    # Use camera-range discontinuities to break projected contact with other objects.
    optical = (world - cam[:3, 3]) @ cam[:3, 2]
    spread = local_depth_range(optical, valid)
    raised = valid & ((world @ up) > z0 + 0.002) & (spread < 0.012)
    if valid[r, c]:
        raised &= abs(optical - optical[r, c]) < 0.46
    cc, n = label(raised)
    if not n:
        return None, {"reason": "no_raised_component"}
    rr, cols = np.ogrid[:h, :w]
    radius = max(7, min(w, h) * 0.065)
    sl = np.s_[
        max(0, r - int(radius)) : min(h, r + int(radius) + 1),
        max(0, c - int(radius)) : min(w, c + int(radius) + 1),
    ]
    ids = cc[sl]
    dist = (rr[sl[0], :] - r) ** 2 + (cols[:, sl[1]] - c) ** 2
    areas = np.bincount(cc.ravel())
    cost = np.full(n + 1, np.inf)
    np.minimum.at(cost, ids.ravel(), dist.ravel())
    cost[0] = np.inf
    cost[(areas < max(22, (min(w, h) / 100) ** 2)) | (areas > h * w * 0.48)] = np.inf
    k = cost.argmin()
    if not np.isfinite(cost[k]) or cost[k] > radius**2:
        return None, {"reason": "no_object_close_to_grounded_pixel"}
    neighborhood = max(8, 0.025 * min(w, h))
    peers = (
        (cost <= neighborhood**2) & (areas >= 0.2 * areas[k]) & (areas <= 5 * areas[k])
    )
    original = k
    if rgb is not None and lightness and np.count_nonzero(peers) >= 2:
        ranked = []
        for candidate in np.flatnonzero(peers):
            luminosity = (
                rgb[cc == candidate] @ np.array([0.2126, 0.7152, 0.0722])
            ) * lightness
            ranked.append((np.quantile(luminosity, [0.25, 0.5, 0.75]), candidate))
        ranked.sort(key=lambda pair: pair[0][1], reverse=True)
        if np.min(ranked[0][0] - ranked[1][0]) > 0.07 * 255:
            k = ranked[0][1]
    return cc == k, {
        "support_height_m": float(z0),
        "anchor_distance_px": float(np.sqrt(cost[k])),
        "n_pixels": int(areas[k]),
        "nearby_components": int(np.count_nonzero(peers)),
        "selected_by_relative_lightness": bool(k != original),
    }


def anchored_plane(world, valid, up, cam, rc):
    """Recover a connected face around a material pixel from metric plane continuity."""
    if cam is None or up is None:
        return None
    r, c = rc
    h, w = valid.shape
    if not valid[r, c]:
        return None
    size = max(4, round(min(w, h) * 0.012))
    roi = np.s_[
        max(0, r - size) : min(h, r + size + 1), max(0, c - size) : min(w, c + size + 1)
    ]
    anchor = world[r, c]
    pts = world[roi][valid[roi]]
    pts = pts[np.linalg.norm(pts - anchor, axis=1) < 0.018]
    if len(pts) < 18:
        return None
    _, s, basis = np.linalg.svd(pts - np.mean(pts, axis=0), full_matrices=False)
    n = basis[-1]
    if s[-1] > s[-2] * 0.23 or abs(n @ up) < 0.65:
        return None
    optical = (world - cam[:3, 3]) @ cam[:3, 2]
    jump = local_depth_range(optical, valid)
    aligned = binary_erosion(
        valid & (abs((world - anchor) @ n) < 0.0021) & (jump < 0.01)
    )
    cc, k = label(aligned)
    idx = cc[r, c]
    if idx == 0:
        return None
    mask = cc == idx
    if not 30 <= mask.sum() <= h * w * 0.31:
        return None
    return mask


def nearby_support_height(world, valid, mask, up, top):
    """Measure a neighboring horizontal plane below the target's top."""
    yy, xx = np.nonzero(mask)
    if len(yy) < 20 or up is None:
        return None
    pad = max(22, round(0.42 * max(np.ptp(yy), np.ptp(xx))))
    region = np.s_[
        max(0, yy.min() - pad) : yy.max() + pad + 1,
        max(0, xx.min() - pad) : xx.max() + pad + 1,
    ]
    z = np.where(valid[region][::4, ::4], world[region][::4, ::4] @ up, np.nan)
    if min(z.shape) < 2:
        return None
    dr, dc = np.gradient(z)
    flat = (
        valid[region][::4, ::4]
        & ~mask[region][::4, ::4]
        & (abs(dr) + abs(dc) < 0.003)
        & (z < top - 0.006)
    )
    samples = z[flat]
    if len(samples) < 25:
        return None
    bins, support = np.unique(np.floor(samples / 0.002).astype(int), return_counts=True)
    if support.max() < max(20, len(samples) * 0.15):
        return None
    plane = (bins[support.argmax()] + 0.5) * 0.002
    return float(np.median(samples[abs(samples - plane) < 0.0025]))
