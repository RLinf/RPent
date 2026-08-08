from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robots.libero import sam3_review
from robots.libero import tools as libero_tools
from rpent.utils.sam3_client import Sam3Result


def _raw_mask() -> np.ndarray:
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:60, 10:55] = True
    mask[10:60, 65:95] = True
    return mask


def _overlay(tmp_path: Path) -> Path:
    path = tmp_path / "overlay.png"
    path.write_bytes(b"fake image; model calls are monkeypatched")
    return path


def _reviewer(*, no_images: bool = False) -> sam3_review.Sam3Reviewer:
    return sam3_review.build_sam3_reviewer(
        planner_type="api",
        model="provider:eval-vision-model",
        base_url="https://eval.example/v1",
        no_images=no_images,
    )


def _reply(decision: str, **extra: object) -> dict[str, object]:
    return {"decision": decision, "confidence": 0.95, **extra}


@pytest.mark.parametrize(
    ("backend", "model", "base_url", "env", "expected_model", "expected_url"),
    [
        (
            "api",
            "provider:vision",
            "https://api.eval/v1",
            {},
            "provider:vision",
            "https://api.eval/v1",
        ),
        (
            "codex",
            "explicit-codex",
            "https://ignored",
            {"CODEX_MODEL": "env-codex", "CODEX_BASE_URL": "https://codex.eval"},
            "explicit-codex",
            "https://codex.eval",
        ),
        (
            "codex",
            None,
            None,
            {"CODEX_MODEL": "env-codex", "CODEX_BASE_URL": "https://codex.eval"},
            "env-codex",
            "https://codex.eval",
        ),
        (
            "claude_code",
            None,
            None,
            {"ANTHROPIC_BASE_URL": "https://claude.eval"},
            "sonnet",
            "https://claude.eval",
        ),
    ],
)
def test_builder_resolves_same_eval_configuration(
    monkeypatch, backend, model, base_url, env, expected_model, expected_url
) -> None:
    from rpent.planner import base as planner_base

    resolver, calls = planner_base.resolve_planner_model, []
    monkeypatch.setattr(
        planner_base,
        "resolve_planner_model",
        lambda kind, value: calls.append((kind, value)) or resolver(kind, value),
    )
    for name in ("CODEX_MODEL", "CODEX_BASE_URL", "ANTHROPIC_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    reviewer = sam3_review.build_sam3_reviewer(backend, model, base_url, False)

    assert reviewer.model == expected_model
    assert reviewer.base_url == expected_url
    assert calls == [(backend, model)]


def test_no_images_never_calls_model(monkeypatch, tmp_path: Path) -> None:
    reviewer = _reviewer(no_images=True)
    monkeypatch.setattr(
        sam3_review, "_call_api", lambda *_: pytest.fail("model must not be called")
    )

    assert reviewer is None


def test_image_rejection_is_cached(monkeypatch, tmp_path: Path) -> None:
    reviewer = _reviewer()
    calls = 0

    def reject_image(*_):
        nonlocal calls
        calls += 1
        raise RuntimeError("model does not support image input")

    monkeypatch.setattr(reviewer, "_call", reject_image)
    raw = _raw_mask()
    path = _overlay(tmp_path)

    first = reviewer.review(raw, path, "cup", "task", lambda: None)
    second = reviewer.review(raw, path, "cup", "task", lambda: None)

    assert calls == 1
    assert np.array_equal(first.mask, raw)
    assert np.array_equal(second.mask, raw)
    assert first.metadata["reason"] == "image_unsupported"
    assert second.metadata["reason"] == "image_unsupported_cached"


@pytest.mark.parametrize("decision", ["KEEP", "WRONG"])
def test_keep_and_wrong_return_raw(monkeypatch, tmp_path: Path, decision: str) -> None:
    reviewer = _reviewer()
    monkeypatch.setattr(reviewer, "_call", lambda *_: _reply(decision))
    raw = _raw_mask()

    result = reviewer.review(raw, _overlay(tmp_path), "cup", "task", lambda: None)

    assert np.array_equal(result.mask, raw)
    assert result.metadata["decision"] == decision
    assert result.metadata["status"] == "kept_raw"
    assert result.metadata["applied"] is False


def test_valid_subtract_is_strict_subset_and_preserves_core(
    monkeypatch, tmp_path: Path
) -> None:
    reviewer = _reviewer()
    response = _reply(
        "SUBTRACT",
        target_point=[30, 30],
        reject_polygons=[[[66, 9], [96, 9], [96, 61], [66, 61]]],
    )
    monkeypatch.setattr(reviewer, "_call", lambda *_: response)
    raw = _raw_mask()

    result = reviewer.review(raw, _overlay(tmp_path), "cup", "task", lambda: None)

    assert result.metadata["status"] == "applied"
    assert result.metadata["protected_core_coverage"] >= 0.90
    assert result.mask.sum() < raw.sum()
    assert np.all(~result.mask | raw)
    assert result.mask[30, 30]
    target_component = np.zeros_like(raw)
    target_component[10:60, 10:55] = True
    assert np.array_equal(result.projection_mask, target_component)
    assert result.mask[:, 65].sum() == 50
    assert not result.projection_mask[:, 65].any()


def test_fragmented_candidate_returns_raw() -> None:
    raw = _raw_mask()
    answer = sam3_review._Answer(
        decision="SUBTRACT",
        confidence=0.95,
        target_point=[30, 30],
        reject_polygons=[[[74, 9], [96, 9], [96, 61], [74, 61]]],
    )

    result = sam3_review._apply(raw, answer, "api")

    assert np.array_equal(result.mask, raw)
    assert result.metadata["reason"] == "candidate_fragmented"


def test_subtract_below_protected_coverage_returns_raw(
    monkeypatch, tmp_path: Path
) -> None:
    reviewer = _reviewer()
    response = _reply(
        "SUBTRACT",
        target_point=[40, 35],
        reject_polygons=[
            [[9, 9], [25, 9], [25, 61], [9, 61]],
            [[64, 9], [96, 9], [96, 61], [64, 61]],
        ],
    )
    monkeypatch.setattr(reviewer, "_call", lambda *_: response)
    raw = _raw_mask()

    result = reviewer.review(raw, _overlay(tmp_path), "cup", "task", lambda: None)

    assert np.array_equal(result.mask, raw)
    assert result.metadata["reason"] == "protected_core_cut"
    assert result.metadata["protected_core_coverage"] < 0.90


@pytest.mark.parametrize(
    "outcome",
    [
        _reply("SUBTRACT", target_point=[30, 30], reject_polygons=[]),
        TimeoutError("review timed out"),
    ],
)
def test_invalid_or_timeout_returns_raw(
    monkeypatch, tmp_path: Path, outcome: object
) -> None:
    reviewer = _reviewer()

    def fake_call(*_):
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(reviewer, "_call", fake_call)
    raw = _raw_mask()

    result = reviewer.review(raw, _overlay(tmp_path), "cup", "task", lambda: None)

    assert np.array_equal(result.mask, raw)
    assert result.metadata["applied"] is False
    assert result.metadata["status"] == "raw_fallback"


def test_codex_uses_one_awaited_turn_with_one_image(
    monkeypatch, tmp_path: Path
) -> None:
    calls = {"turn": 0, "images": 0}

    class LocalImageInput:
        def __init__(self, path):
            self.path = path

    class Turn:
        async def run(self):
            return SimpleNamespace(
                error=None,
                final_response='{"decision":"KEEP","confidence":0.95}',
            )

    class Thread:
        async def turn(self, inputs, **_kwargs):
            calls["turn"] += 1
            calls["images"] += sum(isinstance(item, LocalImageInput) for item in inputs)
            calls["schema"] = _kwargs["output_schema"]
            return Turn()

    class AsyncCodex:
        def __init__(self, _config):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            pass

        async def thread_start(self, **_kwargs):
            return Thread()

    fake = SimpleNamespace(
        AsyncCodex=AsyncCodex,
        CodexConfig=lambda **kwargs: kwargs,
        ApprovalMode=SimpleNamespace(deny_all="deny"),
        Sandbox=SimpleNamespace(read_only="read-only"),
        TextInput=lambda text: text,
        LocalImageInput=LocalImageInput,
    )
    monkeypatch.setitem(sys.modules, "openai_codex", fake)

    result = sam3_review._call_codex("eval-model", None, _overlay(tmp_path), "prompt")

    schema = calls.pop("schema")
    assert result["decision"] == "KEEP"
    assert calls == {"turn": 1, "images": 1}
    assert schema["additionalProperties"] is False
    stack = [schema]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            assert "default" not in value
            if value.get("type") == "object":
                assert value.get("additionalProperties") is False
                assert set(value.get("required", [])) == set(
                    value.get("properties", {})
                )
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)


def test_segment_rolls_back_candidate_with_invalid_world_projection(
    monkeypatch, tmp_path: Path
) -> None:
    target = np.zeros((30, 30), dtype=bool)
    target[2:22, 2:22] = True
    extra = np.zeros_like(target)
    extra[25:28, 2:6] = True
    removed = np.zeros_like(target)
    removed[25:29, 10:15] = True
    raw, candidate = target | extra | removed, target | extra
    image, world_path = tmp_path / "image.png", tmp_path / "world.npy"
    image.write_bytes(b"unused")
    world = np.full((30, 30, 3), np.nan)
    world[extra] = [1.0, 2.0, 3.0]
    np.save(world_path, world)

    class Sam:
        def segment(self, *_args, **_kwargs):
            return Sam3Result(True, 0.9, [2, 2, 17, 17], raw, raw.shape)

    class Reviewer:
        def review(self, *_args):
            return sam3_review.ReviewResult(
                candidate, {"status": "applied", "applied": True}, target
            )

    monkeypatch.setattr(libero_tools, "get_output_dir", lambda: tmp_path)
    monkeypatch.setattr(
        libero_tools, "_select_segment_artifacts", lambda *_: (image, world_path, [])
    )
    monkeypatch.setattr(
        libero_tools, "_load_step", lambda *_: {"task_language": "task"}
    )
    monkeypatch.setattr(libero_tools, "_write_segment_overlay", lambda *_: True)
    primitives = libero_tools.LiberoPrimitives(
        None, None, Sam(), lambda: None, Reviewer()
    )

    result = primitives.segment(prompt="cup", step=0)

    assert result["world_xyz"] == [1.0, 2.0, 3.0]
    assert result["sam3_review"]["reason"] == "invalid_review_projection"
    assert result["sam3_review"]["final_pixels"] == int(raw.sum())
