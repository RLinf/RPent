# ruff: noqa: CPY001
from __future__ import annotations

import json
import signal
import subprocess
from pathlib import Path
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
import pytest

from robots.libero import tools

TASK = "turn on the stove and put the moka pot on it"
OTHER_TASK = "pick up the black bowl"


def _raw_mask() -> np.ndarray:
    """Two large cores joined by a bridge thinner than the guard erosion."""
    mask = np.zeros((128, 128), dtype=bool)
    mask[20:60, 10:50] = True
    mask[20:60, 70:110] = True
    mask[38:42, 50:70] = True
    return mask


def _world_map(*, invalidate_target: bool = False) -> np.ndarray:
    world = np.full((128, 128, 3), np.nan, dtype=np.float32)
    world[20:60, 10:50] = [1.0, 2.0, 3.0]
    world[20:60, 70:110] = [8.0, 9.0, 10.0]
    world[38:42, 50:70] = [4.0, 5.0, 6.0]
    if invalidate_target:
        world[20:60, 10:50] = np.nan
    return world


def _response(
    decision: str,
    *,
    reason_code: str | None = None,
    confidence: float = 0.99,
    target_point: list[int] | None = None,
    reject_polygons: list[list[list[int]]] | None = None,
) -> dict:
    if reason_code is None:
        reason_code = {
            "KEEP": "target_only",
            "WRONG": "wrong_target",
            "SUBTRACT": "mixed_target_and_distractor",
        }[decision]
    return {
        "image_width": 128,
        "image_height": 128,
        "decision": decision,
        "reason_code": reason_code,
        "confidence": confidence,
        "target_point": target_point,
        "reject_polygons": reject_polygons or [],
        "notes": "test response",
    }


def _valid_subtract_response() -> dict:
    # Remove the right core and bridge while preserving all 1,600 target pixels.
    return _response(
        "SUBTRACT",
        target_point=[30, 40],
        reject_polygons=[[[50, 15], [115, 15], [115, 65], [50, 65]]],
    )


class _FakeSam3Client:
    def __init__(self, *, found: bool = True):
        self.found = found
        self.calls: list[tuple[tuple, dict]] = []

    def segment(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return SimpleNamespace(
            found=self.found,
            mask=_raw_mask() if self.found else None,
            reason=None if self.found else "not found",
            score=0.91234 if self.found else None,
            box=[10, 20, 109, 59] if self.found else None,
            mask_shape=(128, 128) if self.found else None,
        )


def _patch_segment_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    world: np.ndarray | None = None,
    task_language: str = TASK,
) -> None:
    image_path = tmp_path / "image.png"
    world_path = tmp_path / "world.npy"
    imageio.imwrite(image_path, np.zeros((128, 128, 3), dtype=np.uint8))
    np.save(world_path, _world_map() if world is None else world)
    monkeypatch.setattr(tools, "_latest_step", lambda: 0)
    monkeypatch.setattr(
        tools, "_load_step", lambda step: {"task_language": task_language}
    )
    monkeypatch.setattr(
        tools,
        "_select_segment_artifacts",
        lambda step, camera: (image_path, world_path, [(image_path, world_path)]),
    )
    monkeypatch.setattr(tools, "get_output_dir", lambda: tmp_path)


def _primitive(client: _FakeSam3Client, task: str = TASK) -> tools.LiberoPrimitives:
    primitive = tools.LiberoPrimitives(object(), object(), client, lambda: None)
    primitive._last_obs = {"task_descriptions": task}
    return primitive


def _enable_review(
    monkeypatch: pytest.MonkeyPatch,
    *,
    mode: str = "apply",
    tasks: list[str] | None = None,
) -> None:
    monkeypatch.setenv("RPENT_LIBERO_SAM3_LLM_REVIEW_MODE", mode)
    monkeypatch.setenv(
        "RPENT_LIBERO_SAM3_LLM_REVIEW_TASKS", json.dumps(tasks or [TASK])
    )


def _patch_codex(
    monkeypatch: pytest.MonkeyPatch,
    response: dict,
    *,
    returncode: int = 0,
) -> list[list[str]]:
    calls: list[list[str]] = []

    def fake_run(command, check_cancelled):
        calls.append(command)
        check_cancelled()
        assert command[0] == "/custom/codex"
        assert command[command.index("-s") + 1] == "read-only"
        assert command[command.index("-m") + 1] == "gpt-5.5"
        if returncode == 0:
            output_path = Path(command[command.index("-o") + 1])
            output_path.write_text(json.dumps(response), encoding="utf-8")
        return subprocess.CompletedProcess(
            command, returncode, stdout="", stderr="codex failed"
        )

    monkeypatch.setenv("CODEX_BIN", "/custom/codex")
    monkeypatch.setattr(tools, "_run_sam3_llm_codex", fake_run)
    return calls


def _segment_blob(result: dict) -> dict:
    return json.loads(Path(result["segment_path"]).read_text(encoding="utf-8"))


def _review_metadata(blob: dict) -> dict:
    for key in (
        "sam3_llm_review",
        "llm_subtractive_refinement",
        "llm_review",
        "refinement",
    ):
        value = blob.get(key)
        if isinstance(value, dict):
            return value
    raise AssertionError(f"segment artifact has no review metadata: {blob.keys()}")


def _assert_raw_result(
    result: dict, *, expected_world: list[float] | None = None
) -> dict:
    assert result["found"] is True
    assert "error" not in result
    assert result["score"] == pytest.approx(0.912, abs=1e-6)
    assert result["box"] == [10, 20, 109, 59]
    expected_world = expected_world or [4.0, 5.0, 6.0]
    assert result["world_xyz"] == expected_world
    blob = _segment_blob(result)
    assert blob["score"] == pytest.approx(0.912, abs=1e-6)
    assert blob["box"] == [10, 20, 109, 59]
    assert blob["world_xyz"] == expected_world
    return blob


@pytest.mark.parametrize(
    ("mode", "task", "use_point"),
    [
        ("off", TASK, False),
        ("apply", OTHER_TASK, False),
        ("apply", TASK, True),
    ],
)
def test_review_is_not_called_when_disabled_unlisted_or_point_mode(
    monkeypatch, tmp_path, mode, task, use_point
) -> None:
    _patch_segment_artifacts(monkeypatch, tmp_path, task_language=task)
    _enable_review(monkeypatch, mode=mode, tasks=[TASK])

    def unexpected(*args, **kwargs):
        raise AssertionError("Codex must not run for an ineligible segment call")

    monkeypatch.setattr(tools, "_run_sam3_llm_codex", unexpected)
    primitive = _primitive(_FakeSam3Client(), task)

    if use_point:
        result = primitive.segment(point=[30, 40])
    else:
        result = primitive.segment(prompt="the silver octagonal moka pot")

    blob = _assert_raw_result(result)
    assert not any("review" in key or "refinement" in key for key in blob)


def test_not_found_does_not_call_review(monkeypatch, tmp_path) -> None:
    _patch_segment_artifacts(monkeypatch, tmp_path)
    _enable_review(monkeypatch)
    monkeypatch.setattr(
        tools,
        "_run_sam3_llm_codex",
        lambda *args, **kwargs: pytest.fail("Codex must not run without a SAM mask"),
    )

    result = _primitive(_FakeSam3Client(found=False)).segment(prompt="the moka pot")

    assert result["found"] is False
    assert result["world_xyz"] is None
    assert "not found" in result["error"]
    assert Path(result["segment_path"]).exists()


@pytest.mark.parametrize(
    "response",
    [
        _response("KEEP"),
        _response("WRONG"),
    ],
    ids=["keep", "wrong"],
)
def test_keep_and_wrong_decisions_leave_raw_result_unchanged(
    monkeypatch, tmp_path, response
) -> None:
    _patch_segment_artifacts(monkeypatch, tmp_path)
    _enable_review(monkeypatch)
    _patch_codex(monkeypatch, response)

    result = _primitive(_FakeSam3Client()).segment(prompt="the moka pot")

    blob = _assert_raw_result(result)
    review = _review_metadata(blob)
    assert review["decision"] == response["decision"]
    assert review.get("chosen_source", "raw") == "raw"


@pytest.mark.parametrize("decision", ["KEEP", "WRONG", "SUBTRACT"])
def test_low_confidence_review_is_advisory_raw(monkeypatch, tmp_path, decision) -> None:
    _patch_segment_artifacts(monkeypatch, tmp_path)
    _enable_review(monkeypatch)
    response = (
        _valid_subtract_response() if decision == "SUBTRACT" else _response(decision)
    )
    response["confidence"] = 0.89
    _patch_codex(monkeypatch, response)

    result = _primitive(_FakeSam3Client()).segment(prompt="the moka pot")

    review = _review_metadata(_assert_raw_result(result))
    assert review["status"] == "raw_fallback"
    assert review["fallback_reason"] == "review_confidence_below_threshold"


def test_codex_error_falls_back_to_raw_and_still_writes_artifact(
    monkeypatch, tmp_path
) -> None:
    _patch_segment_artifacts(monkeypatch, tmp_path)
    _enable_review(monkeypatch)
    _patch_codex(monkeypatch, _response("KEEP"), returncode=7)

    result = _primitive(_FakeSam3Client()).segment(prompt="the moka pot")

    blob = _assert_raw_result(result)
    review = _review_metadata(blob)
    assert review.get("chosen_source", "raw") == "raw"
    assert review.get("status") not in (None, "applied")


def test_invalid_task_whitelist_config_keeps_official_raw_path(
    monkeypatch, tmp_path
) -> None:
    _patch_segment_artifacts(monkeypatch, tmp_path)
    monkeypatch.setenv("RPENT_LIBERO_SAM3_LLM_REVIEW_MODE", "apply")
    monkeypatch.setenv("RPENT_LIBERO_SAM3_LLM_REVIEW_TASKS", "not-json")
    monkeypatch.setattr(
        tools,
        "_run_sam3_llm_codex",
        lambda *args, **kwargs: pytest.fail("invalid config must not run Codex"),
    )

    result = _primitive(_FakeSam3Client()).segment(prompt="the moka pot")

    blob = _assert_raw_result(result)
    assert "sam3_llm_review" not in blob


@pytest.mark.parametrize("payload", ["not-json", {"image_width": 127}])
def test_malformed_or_wrong_size_response_falls_back_and_seals_artifact(
    monkeypatch, tmp_path, payload
) -> None:
    _patch_segment_artifacts(monkeypatch, tmp_path)
    _enable_review(monkeypatch)

    def fake_run(command, check_cancelled):
        check_cancelled()
        response_path = Path(command[command.index("-o") + 1])
        if isinstance(payload, str):
            response_path.write_text(payload, encoding="utf-8")
        else:
            response_path.write_text(json.dumps(payload), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(tools, "_run_sam3_llm_codex", fake_run)

    result = _primitive(_FakeSam3Client()).segment(prompt="the moka pot")

    review = _review_metadata(_assert_raw_result(result))
    assert review["status"] == "raw_fallback"
    assert review["fallback_reason"].startswith("invalid_review_response:")


def test_shadow_mode_records_valid_subtraction_but_projects_raw_mask(
    monkeypatch, tmp_path
) -> None:
    _patch_segment_artifacts(monkeypatch, tmp_path)
    _enable_review(monkeypatch, mode="shadow")
    _patch_codex(monkeypatch, _valid_subtract_response())

    result = _primitive(_FakeSam3Client()).segment(prompt="the moka pot")

    blob = _assert_raw_result(result)
    review = _review_metadata(blob)
    assert review["decision"] == "SUBTRACT"
    assert review.get("chosen_source", "raw") == "raw"
    assert review.get("candidate_valid", True) is True
    assert review["effective_box"] == [10, 20, 109, 59]
    assert review["final_pixels"] == int(_raw_mask().sum())
    assert review["retained_fraction"] == 1.0
    assert review["candidate_box"] == [10, 20, 49, 59]
    assert review["candidate_pixels"] == 1600


def test_apply_uses_valid_component_subtraction_for_world_only(
    monkeypatch, tmp_path
) -> None:
    _patch_segment_artifacts(monkeypatch, tmp_path)
    _enable_review(monkeypatch)
    calls = _patch_codex(monkeypatch, _valid_subtract_response())

    result = _primitive(_FakeSam3Client()).segment(prompt="the moka pot")

    assert result["found"] is True
    assert result["score"] == pytest.approx(0.912, abs=1e-6)
    # Public score and box retain raw SAM semantics even when final-mask world
    # projection uses the safe candidate.
    assert result["box"] == [10, 20, 109, 59]
    assert result["world_xyz"] == [1.0, 2.0, 3.0]
    blob = _segment_blob(result)
    assert blob["score"] == pytest.approx(0.912, abs=1e-6)
    assert blob["box"] == [10, 20, 109, 59]
    assert blob["world_xyz"] == [1.0, 2.0, 3.0]
    review = _review_metadata(blob)
    assert review["decision"] == "SUBTRACT"
    assert review.get("chosen_source", "refined") in ("candidate", "refined")
    assert review["raw_box"] == [10, 20, 109, 59]
    assert review["raw_score"] == pytest.approx(0.91234)
    assert review["effective_box"] == review["candidate_box"]
    assert review["final_pixels"] == review["candidate_pixels"] == 1600
    assert review["retained_fraction"] == review["candidate_retained_fraction"]
    assert "generous margin" in calls[0][-1]
    assert "deletion is intersected with the red SAM3 mask" in calls[0][-1]


@pytest.mark.parametrize(
    ("response", "world"),
    [
        # No significant core is removed.
        (
            _response(
                "SUBTRACT",
                target_point=[30, 40],
                reject_polygons=[[[0, 0], [5, 0], [5, 5], [0, 5]]],
            ),
            _world_map(),
        ),
        # Partial overlap cuts directly through a significant target core.
        (
            _response(
                "SUBTRACT",
                target_point=[20, 40],
                reject_polygons=[[[30, 15], [55, 15], [55, 65], [30, 65]]],
            ),
            _world_map(),
        ),
        # target_point lies in the rejected core.
        (
            _response(
                "SUBTRACT",
                target_point=[90, 40],
                reject_polygons=[[[49, 15], [115, 15], [115, 65], [49, 65]]],
            ),
            _world_map(),
        ),
        # A geometrically valid candidate has no valid world projection.
        (_valid_subtract_response(), _world_map(invalidate_target=True)),
        # Retained fraction is above 0.60.
        (
            _response(
                "SUBTRACT",
                target_point=[30, 40],
                reject_polygons=[[[88, 15], [115, 15], [115, 65], [88, 65]]],
            ),
            _world_map(),
        ),
        # Retained fraction is below 0.20 / final candidate is too small.
        (
            _response(
                "SUBTRACT",
                target_point=[12, 22],
                reject_polygons=[[[15, 0], [127, 0], [127, 127], [15, 127]]],
            ),
            _world_map(),
        ),
        # Candidate is fragmented into multiple similarly-sized components.
        (
            _response(
                "SUBTRACT",
                target_point=[20, 40],
                reject_polygons=[[[35, 15], [85, 15], [85, 65], [35, 65]]],
            ),
            _world_map(),
        ),
    ],
    ids=[
        "no-core-removed",
        "partial-core-cut",
        "target-point-rejected",
        "invalid-world",
        "retained-too-high",
        "retained-too-low-or-tiny",
        "fragmented-candidate",
    ],
)
def test_guard_rejection_falls_back_to_raw(
    monkeypatch, tmp_path, response, world
) -> None:
    _patch_segment_artifacts(monkeypatch, tmp_path, world=world)
    _enable_review(monkeypatch)
    _patch_codex(monkeypatch, response)

    result = _primitive(_FakeSam3Client()).segment(prompt="the moka pot")

    expected_world = tools._mask_to_world(_raw_mask(), world)["world_xyz"]
    blob = _assert_raw_result(result, expected_world=expected_world)
    review = _review_metadata(blob)
    assert review.get("chosen_source", "raw") == "raw"
    assert review.get("status") != "applied"
    assert review["effective_box"] == [10, 20, 109, 59]
    assert review["final_pixels"] == int(_raw_mask().sum())
    assert review["retained_fraction"] == 1.0


def test_guard_rejects_target_boundary_erosion() -> None:
    raw = _raw_mask()
    candidate = np.zeros_like(raw)
    candidate[23:57, 13:47] = True

    accepted, diagnostic = tools._guard_sam3_subtraction(
        raw, candidate, [30, 40], _world_map()
    )

    assert accepted is False
    assert diagnostic["guard_reason"] == "target_boundary_not_preserved"
    assert diagnostic["protected_target_coverage"] < 0.99


def test_guard_rejects_candidate_world_inconsistent_with_target_core() -> None:
    raw = np.zeros((256, 256), dtype=bool)
    raw[10:40, 10:40] = True
    raw[10:240, 38:41] = True
    for row in range(10, 238, 12):
        raw[row : row + 3, 38:150] = True
    raw[50:250, 160:250] = True
    candidate = raw.copy()
    candidate[50:250, 160:250] = False
    world = np.full((256, 256, 3), np.nan, dtype=np.float32)
    world[raw] = [8.0, 9.0, 10.0]
    world[10:40, 10:40] = [1.0, 2.0, 3.0]
    world[50:250, 160:250] = [20.0, 21.0, 22.0]

    accepted, diagnostic = tools._guard_sam3_subtraction(
        raw, candidate, [20, 20], world
    )

    assert accepted is False
    assert diagnostic["guard_reason"] == "candidate_target_world_inconsistent"
    assert diagnostic["candidate_target_xy_drift_m"] > 0.005


@pytest.mark.parametrize(("z_drift", "expected"), [(0.010, True), (0.010001, False)])
def test_guard_z_drift_boundary(monkeypatch, z_drift, expected) -> None:
    raw = _raw_mask()
    candidate = np.zeros_like(raw)
    candidate[20:60, 10:50] = True
    calls = 0

    def fake_mask_to_world(mask, world_map):
        nonlocal calls
        calls += 1
        return {"world_xyz": [0.0, 0.0, 0.0 if calls == 1 else z_drift]}

    monkeypatch.setattr(tools, "_mask_to_world", fake_mask_to_world)

    accepted, diagnostic = tools._guard_sam3_subtraction(
        raw, candidate, [30, 40], np.zeros((128, 128, 3))
    )

    assert accepted is expected
    if expected:
        assert diagnostic["guard_reason"] == "accepted"
    else:
        assert diagnostic["guard_reason"] == "candidate_target_world_inconsistent"


def test_review_artifacts_are_auditable_and_raw_overlay_is_preserved(
    monkeypatch, tmp_path
) -> None:
    _patch_segment_artifacts(monkeypatch, tmp_path)
    _enable_review(monkeypatch)
    _patch_codex(monkeypatch, _valid_subtract_response())

    result = _primitive(_FakeSam3Client()).segment(prompt="the moka pot")

    blob = _segment_blob(result)
    review = _review_metadata(blob)
    artifact_values = [
        Path(value)
        for key, value in review.items()
        if key.endswith("_path") and isinstance(value, str)
    ]
    assert artifact_values
    assert all(path.exists() for path in artifact_values)
    assert any("raw" in path.name and "sam3" in path.name for path in artifact_values)
    assert any("response" in path.name for path in artifact_values)
    assert result["overlay_path"] != next(
        str(path)
        for path in artifact_values
        if "raw" in path.name and "sam3" in path.name
    )


def test_codex_timeout_falls_back_to_raw_and_records_failure(
    monkeypatch, tmp_path
) -> None:
    _patch_segment_artifacts(monkeypatch, tmp_path)
    _enable_review(monkeypatch)
    monkeypatch.setenv("CODEX_BIN", "/custom/codex")

    def timeout(command, check_cancelled):
        check_cancelled()
        raise subprocess.TimeoutExpired(command, 120)

    monkeypatch.setattr(tools, "_run_sam3_llm_codex", timeout)

    result = _primitive(_FakeSam3Client()).segment(prompt="the moka pot")

    blob = _assert_raw_result(result)
    review = _review_metadata(blob)
    assert review.get("chosen_source", "raw") == "raw"
    assert "timeout" in json.dumps(review).lower()


def test_codex_runner_kills_process_group_on_timeout(monkeypatch) -> None:
    killed: list[tuple[int, signal.Signals]] = []

    class FakeProcess:
        pid = 321
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self):
            self.returncode = -signal.SIGKILL

    process = FakeProcess()
    monkeypatch.setattr(tools.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(tools.os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    monkeypatch.setattr(tools, "_SAM3_LLM_REVIEW_TIMEOUT_S", 0.0)

    with pytest.raises(subprocess.TimeoutExpired):
        tools._run_sam3_llm_codex(["codex"], lambda: None)

    assert killed == [(321, signal.SIGKILL)]
    assert process.returncode == -signal.SIGKILL


def test_cancellation_kills_codex_and_seals_standard_segment_artifact(
    monkeypatch, tmp_path
) -> None:
    _patch_segment_artifacts(monkeypatch, tmp_path)
    _enable_review(monkeypatch)
    killed: list[tuple[int, signal.Signals]] = []
    checks = 0

    class FakeProcess:
        pid = 654
        returncode = None

        def poll(self):
            return self.returncode

        def communicate(self, timeout=None):
            raise AssertionError("cancellation should be checked before communicate")

        def wait(self):
            self.returncode = -signal.SIGKILL

    def check_cancelled() -> None:
        nonlocal checks
        checks += 1
        if checks >= 2:
            raise RuntimeError("test cancellation")

    monkeypatch.setattr(
        tools.subprocess, "Popen", lambda *args, **kwargs: FakeProcess()
    )
    monkeypatch.setattr(tools.os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    primitive = tools.LiberoPrimitives(
        object(), object(), _FakeSam3Client(), check_cancelled
    )

    with pytest.raises(RuntimeError, match="test cancellation"):
        primitive.segment(prompt="the moka pot")

    assert killed == [(654, signal.SIGKILL)]
    artifacts = list((tmp_path / "segments").glob("segment_*.json"))
    assert len(artifacts) == 1
    blob = json.loads(artifacts[0].read_text(encoding="utf-8"))
    assert blob["world_xyz"] == [4.0, 5.0, 6.0]
    assert blob["sam3_llm_review"]["fallback_reason"] == "review_cancelled"
