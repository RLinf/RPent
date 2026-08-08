from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
import pytest

from robots.libero import tools


def _sam_mask() -> np.ndarray:
    mask = np.zeros((48, 48), dtype=bool)
    mask[4:20, 4:20] = True
    mask[28:44, 28:44] = True
    return mask


def _response(*, width: int = 48, polygon: list | None = None) -> dict:
    return {
        "image_width": width,
        "image_height": 48,
        "polygon": polygon
        or [[3, 3], [12, 3], [20, 3], [20, 12], [20, 20], [12, 20], [3, 20], [3, 12]],
        "confidence": 0.9,
        "notes": "target only",
    }


def _patch_codex(monkeypatch, response: dict, *, returncode: int = 0) -> None:
    def fake_run(command, **kwargs):
        assert command[0] == "/custom/codex"
        assert command[command.index("-i") + 1].endswith("_sam3_raw.png")
        assert command[command.index("-s") + 1] == "read-only"
        assert command[command.index("-m") + 1] == "gpt-5.5"
        if returncode == 0:
            with open(command[command.index("-o") + 1], "w", encoding="utf-8") as f:
                json.dump(response, f)
        return subprocess.CompletedProcess(
            command, returncode, stdout="", stderr="boom"
        )

    monkeypatch.setenv("CODEX_BIN", "/custom/codex")
    monkeypatch.setattr(tools.subprocess, "run", fake_run)


def _refine(monkeypatch, tmp_path, response: dict):
    overlay_path = tmp_path / "segment_overlay_00_00.png"
    imageio.imwrite(overlay_path, np.zeros((48, 48, 3), dtype=np.uint8))
    _patch_codex(monkeypatch, response)
    return tools._refine_moka_mask_with_codex(
        _sam_mask(), overlay_path, tmp_path / "segment_00_00.json"
    )


def test_refinement_can_only_subtract_sam3_pixels(monkeypatch, tmp_path) -> None:
    refined, metadata = _refine(monkeypatch, tmp_path, _response())
    sam_mask = _sam_mask()

    assert not np.any(refined & ~sam_mask)
    assert refined[4:20, 4:20].all()
    assert not refined[28:44, 28:44].any()
    assert metadata["sam_pixels"] == int(sam_mask.sum())
    assert metadata["refined_box"] == [4, 4, 19, 19]
    assert (tmp_path / "llm_subtractive_00_00_mask.png").exists()


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (_response(width=47), "dimensions do not match"),
        (_response(polygon=[[21, 21]] * 8), "retained too few pixels"),
    ],
)
def test_refinement_rejects_invalid_llm_masks(
    monkeypatch, tmp_path, response, message
) -> None:
    with pytest.raises(ValueError, match=message):
        _refine(monkeypatch, tmp_path, response)


def test_refinement_fails_closed_when_codex_fails(monkeypatch, tmp_path) -> None:
    overlay_path = tmp_path / "overlay.png"
    imageio.imwrite(overlay_path, np.zeros((48, 48, 3), dtype=np.uint8))
    _patch_codex(monkeypatch, _response(), returncode=1)

    with pytest.raises(RuntimeError, match="codex exit=1: boom"):
        tools._refine_moka_mask_with_codex(
            _sam_mask(), overlay_path, tmp_path / "segment_00_00.json"
        )


class _FakeSam3Client:
    def segment(self, *args, **kwargs):
        return SimpleNamespace(
            found=True,
            mask=_sam_mask(),
            reason=None,
            score=0.9,
            box=[3, 3, 44, 44],
            mask_shape=(48, 48),
        )


def _patch_segment_artifacts(monkeypatch, tmp_path, world: np.ndarray) -> None:
    image_path = tmp_path / "image.png"
    world_path = tmp_path / "world.npy"
    image_path.write_bytes(b"image")
    np.save(world_path, world)
    monkeypatch.setattr(tools, "_latest_step", lambda: 0)
    monkeypatch.setattr(
        tools,
        "_select_segment_artifacts",
        lambda step, camera: (image_path, world_path, [(image_path, world_path)]),
    )
    monkeypatch.setattr(tools, "get_output_dir", lambda: tmp_path)


@pytest.mark.parametrize(
    ("prompt", "blocked"),
    [("the silver octagonal moka pot", True), ("the stove burner", False)],
)
def test_overlay_failure_only_blocks_matching_strict_prompt(
    monkeypatch, tmp_path, prompt, blocked
) -> None:
    _patch_segment_artifacts(monkeypatch, tmp_path, np.zeros((48, 48, 3)))
    monkeypatch.setenv("RPENT_LIBERO_MOKA_LLM_SUBTRACTIVE", "1")
    monkeypatch.setattr(tools, "_write_segment_overlay", lambda *args: False)
    primitives = tools.LiberoPrimitives(
        object(), object(), _FakeSam3Client(), lambda: None
    )

    result = primitives.segment(prompt=prompt)

    assert ("error" in result) is blocked
    if blocked:
        assert "needs a SAM3 overlay" in result["error"]


def test_segment_projects_the_refined_mask(monkeypatch, tmp_path) -> None:
    world = np.zeros((48, 48, 3), dtype=np.float32)
    world[4:20, 4:20] = [1.0, 2.0, 3.0]
    world[28:44, 28:44] = [8.0, 9.0, 10.0]
    _patch_segment_artifacts(monkeypatch, tmp_path, world)
    monkeypatch.setenv("RPENT_LIBERO_MOKA_LLM_SUBTRACTIVE", "1")
    monkeypatch.setattr(tools, "_write_segment_overlay", lambda *args: True)
    refined = np.zeros((48, 48), dtype=bool)
    refined[4:20, 4:20] = True
    metadata = {"mode": "strict_subtractive", "refined_box": [4, 4, 19, 19]}
    monkeypatch.setattr(
        tools, "_refine_moka_mask_with_codex", lambda *args: (refined, metadata)
    )
    primitives = tools.LiberoPrimitives(
        object(), object(), _FakeSam3Client(), lambda: None
    )

    result = primitives.segment(prompt="the silver octagonal moka pot")

    assert result["world_xyz"] == [1.0, 2.0, 3.0]
    assert result["box"] == [4, 4, 19, 19]
    blob = json.loads((tmp_path / "segments/segment_00_00.json").read_text())
    assert blob["llm_subtractive_refinement"]["sam3_box"] == [3, 3, 44, 44]
