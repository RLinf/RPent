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

"""Use calibrated alternative observations when a single view lacks requested evidence."""

from __future__ import annotations

import time

import numpy as np
from scipy.spatial import cKDTree

from rpent.tools import ToolResult
from toolkits.observation import GeometryObservation


def _missing(result):
    return result.get("status") in {"partial", "need_observation", "warning"}


def _accepted(field):
    return field.get("value") is not None and field.get("fit_accepted") is True


def surfaces_overlap(a, b):
    """Check overlap of independently identified measured surfaces in the shared frame."""
    if a is None or b is None or not len(a) or not len(b):
        return False
    span = min(np.linalg.norm(np.ptp(a, axis=0)), np.linalg.norm(np.ptp(b, axis=0)))
    tol = np.clip(0.045 * span, 0.0025, 0.012)
    nearest = cKDTree(a).query(b)[0]
    reverse = cKDTree(b).query(a)[0]
    return (
        min(np.count_nonzero(nearest < tol), np.count_nonzero(reverse < tol)) >= 5
        and max(np.mean(nearest < tol), np.mean(reverse < tol)) >= 0.09
    )


def _definite_twins(data):
    reasons = [data.get("reason")] + [
        v.get("reason") for v in data.get("identity_evidence", {}).values()
    ]
    return bool(
        set(reasons)
        & {
            "multiple_matching_colored_boxes",
            "multiple_matching_packages",
            "another_compatible_open_container",
            "another_visually_compatible_instance",
        }
    )


def _project_hint(points, observation):
    """Find a source surface also measured at its projection in this camera."""
    pose = observation.camera_to_world
    optical = (points - pose[:3, 3]) @ pose[:3, :3]
    front = np.isfinite(optical).all(axis=1) & (optical[:, 2] > 0)
    points, optical = points[front], optical[front]
    pixels = optical @ observation.intrinsics.T
    uv = pixels[:, :2] / pixels[:, 2, None]
    h, w = observation.valid_mask.shape
    inside = ((uv >= 0) & (uv < [w, h])).all(axis=1)
    if not inside.any():
        return None, "source_region_outside_view"
    xy = np.minimum(np.rint(uv[inside]).astype(int), [w - 1, h - 1])
    rc = xy[:, ::-1]
    index = tuple(rc.T)
    # Require agreement with the actual surface, including depth occlusion.
    # Allow up to 1 cm at typical tabletop pixel footprints.
    gaps = np.linalg.norm(observation.world_xyz[index] - points[inside], axis=1)
    visible = observation.valid_mask[index] & (gaps <= 0.01)
    if not visible.any():
        return None, "source_region_not_visible_at_measured_depth"
    rc = rc[visible]
    center = np.median(rc, axis=0)
    pixel = rc[np.argmin(np.linalg.norm(rc - center, axis=1))]
    return pixel.tolist(), None


def _merge_fields(primary, candidate):
    fields = primary.get("geometry", {})
    merged = False
    for name, other in candidate.get("geometry", {}).items():
        current = fields.get(name)
        if (
            current is not None
            and other.get("value") is not None
            and (
                current.get("value") is None
                or (_accepted(other) and not _accepted(current))
            )
        ):
            fields[name] = {
                **other,
                "source_camera": other.get("source_camera", candidate["camera"]),
            }
            merged = True
    if merged:
        warnings = [
            {"quantity": k, "reason": w}
            for k, f in fields.items()
            for w in f.get("warnings", ())
        ]
        missing = any(f.get("value") is None for f in fields.values())
        primary.update(
            found=True,
            warnings=warnings,
            evidence_status="inferred",
            status="partial"
            if missing
            else "warning"
            if warnings or not all(map(_accepted, fields.values()))
            else "usable",
        )
        if not missing:
            primary.pop("reason", None)
            primary.pop("missing_geometry", None)
    return merged


def observe_with_alternatives(
    provider,
    measure,
    query,
    camera,
    step,
    measurement,
    fields,
    cancel,
):
    """Resolve the step once and verify the named instance across recorded cameras."""
    start = time.perf_counter()
    cancel()
    try:
        primary = provider.get_observation(camera=camera, step=step)
        if (
            not isinstance(primary, GeometryObservation)
            or primary.camera != camera
            or (step >= 0 and primary.step != step)
        ):
            raise ValueError("Provider returned a different camera or step.")
    except (ValueError, LookupError, OSError, RuntimeError) as exc:
        return ToolResult(
            data={
                "agent": "GeometryTool",
                "status": "error",
                "code": "observation_unavailable",
                "camera": camera,
                "step": step,
                "world_xyz": None,
                "error": str(exc),
            }
        )
    evidence = {}
    initial = measure(
        query,
        camera,
        primary.step,
        measurement,
        fields,
        cancel,
        observation=primary,
        evidence=evidence,
    )
    best = initial
    first = initial.data
    if first.get("status") == "error":
        return initial
    calls = dict(first.get("calls", {}))
    attempts = [{"camera": camera, "step": primary.step, "status": first["status"]}]
    if first.get("reason"):
        attempts[0]["reason"] = first["reason"]
    skipped = []
    used = []
    poses = [primary.camera_to_world]
    ambiguous = _definite_twins(first)
    source_points = evidence.get("points")
    # An unverified model guess is not a constraint on a different camera.
    if source_points is None and ambiguous and "anchor_xyz" in evidence:
        source_points = np.asarray([evidence["anchor_xyz"]])
    extra = getattr(provider, "additional_observations", None)
    if _missing(first) and callable(extra):
        for obs in _available_views(extra, primary, skipped):
            cancel()
            if not isinstance(obs, GeometryObservation):
                skipped.append({"reason": "invalid_observation"})
                continue
            reason = None
            if obs.step != primary.step:
                reason = "different_pinned_step"
            elif (
                obs.camera_to_world is None
                or obs.intrinsics is None
                or obs.world_frame_id != primary.world_frame_id
            ):
                reason = "incompatible_calibration_or_world_frame"
            elif obs.camera == camera or any(
                p is not None and np.allclose(obs.camera_to_world, p, rtol=0, atol=1e-5)
                for p in poses
            ):
                reason = "same_optical_view"
            if reason:
                skipped.append({"camera": obs.camera, "reason": reason})
                continue
            poses.append(obs.camera_to_world)
            hint = None
            if source_points is not None:
                hint, reason = _project_hint(source_points, obs)
            if ambiguous and hint is None and "points" not in evidence:
                # A look-alike elsewhere cannot resolve the ambiguous source instance.
                skipped.append(
                    {
                        "camera": obs.camera,
                        "reason": reason or "source_hint_has_no_valid_3d",
                    }
                )
                continue
            other = {}
            if "points" in evidence:
                other["source_points"] = evidence["points"]
            result = measure(
                query,
                obs.camera,
                primary.step,
                measurement,
                fields,
                cancel,
                observation=obs,
                evidence=other,
                direction_camera=primary.camera_to_world,
                projected_rc=hint,
            )
            candidate = result.data
            attempt = {
                "camera": obs.camera,
                "step": obs.step,
                "status": candidate["status"],
            }
            if candidate.get("reason"):
                attempt["reason"] = candidate["reason"]
            attempts.append(attempt)
            for key, count in candidate.get("calls", {}).items():
                calls[key] = calls.get(key, 0) + count
            if candidate["status"] == "error":
                attempt["reason"] = candidate.get("code", "view_measurement_failed")
                continue
            # Neither an unverified hint nor a matching category resolves known twins.
            if _definite_twins(candidate):
                ambiguous = True
                attempt["reason"] = "same_description_refers_to_multiple_instances"
                if not candidate.get("found"):
                    best, evidence, used = result, {}, []
                continue
            if not candidate.get("found"):
                continue
            if ambiguous:
                # Require an independently verified relation at the projected hint.
                if hint is None or not any(
                    e.get("accepted") and e.get("relation_verified")
                    for e in candidate.get("identity_evidence", {}).values()
                ):
                    attempt["reason"] = "same_description_refers_to_multiple_instances"
                    continue
            if (
                best.data.get("found") or "points" in evidence
            ) and not surfaces_overlap(evidence.get("points"), other.get("points")):
                attempt["reason"] = "same_target_not_confirmed_across_views"
                continue
            ambiguous = False
            if not best.data.get("found"):
                best, evidence = result, other
                used = [obs.camera]
                if other.get("points") is not None:
                    source_points = other["points"]
            elif _merge_fields(best.data, candidate):
                used.append(obs.camera)
            if not _missing(best.data):
                break
    if ambiguous and best.data.get("found"):
        # Contradictory identity invalidates even fields accepted in the first view.
        prior = best.data
        best = ToolResult(
            data={
                k: prior[k]
                for k in (
                    "agent",
                    "query",
                    "camera",
                    "step",
                    "measurement",
                    "world_frame_id",
                    "units",
                    "model",
                )
                if k in prior
            }
        )
        best.data.update(
            status="need_observation",
            found=False,
            world_xyz=None,
            evidence_status="unknown",
            reason="same_description_refers_to_multiple_instances",
        )
        used = []
    data = best.data
    data["view_evidence"] = {
        "first_camera": camera,
        "first_status": attempts[0]["status"],
        "attempts": attempts,
        "skipped": skipped,
        "used_additional": used,
    }
    if _missing(data):
        quantities = data.get("geometry", {})
        missing = [
            k
            for k in (fields or quantities)
            if quantities.get(k, {}).get("value") is None
        ]
        if missing:
            data["missing_geometry"] = missing
        if data["status"] == "need_observation":
            data["status"] = "insufficient_evidence"
            data.setdefault("reason", "requested_geometry_not_observable")
    data["calls"] = calls
    data["elapsed_ms"] = round((time.perf_counter() - start) * 1000, 2)
    return best


def _available_views(extra, primary, skipped):
    """An unavailable auxiliary sensor must preserve the evidence in the pinned view."""
    try:
        yield from extra(primary)
    except (OSError, LookupError, ValueError, RuntimeError) as exc:
        skipped.append(
            {"reason": "additional_observation_unavailable", "detail": str(exc)}
        )
