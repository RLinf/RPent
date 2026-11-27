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

from __future__ import annotations

import base64
import binascii
import threading
from typing import Any
from unittest.mock import Mock

import pytest

from rpent.robots.components.sam3_client import Sam3Client
from rpent.robots.components.sam3_server import Sam3Facade, Sam3Result
from rpent.utils.rpc import RpcError, RpcFacade


class _StubSam3(Sam3Facade):
    """Sam3Facade with the CUDA checkpoint load and inference stubbed out."""

    def __init__(self, *, height: int = 8, width: int = 8) -> None:
        self._height = height
        self._width = width
        super().__init__("unused-checkpoint")

    def _load(self, checkpoint: str) -> None:
        self._torch = None
        self._model = None
        self._processor = None
        self._device = "cpu"
        self._lock = threading.Lock()
        self._image_digest = None
        self._image_state = None

    def _state_for_image(self, image_bytes: bytes) -> dict[str, Any]:
        return {"original_height": self._height, "original_width": self._width}

    def _segment_text(self, state: Any, prompt: str, min_score: float) -> Sam3Result:
        return Sam3Result(found=True, score=1.0, reason=prompt)

    def _segment_point(
        self, state: Any, row: int, col: int, min_score: float
    ) -> Sam3Result:
        return Sam3Result(found=True, score=1.0, reason=f"{row},{col}")


def _image_b64(payload: bytes = b"image-bytes") -> str:
    return base64.b64encode(payload).decode("ascii")


def test_sam3_segment_is_a_readonly_namespaced_rpc() -> None:
    facade = _StubSam3()
    assert facade._rpc["sam3.segment"] == facade.segment
    assert "sam3.segment" in facade._readonly_methods
    # The bare name is not a valid RPC method.
    assert "segment" not in facade._rpc


def test_sam3_dispatch_rejects_the_bare_method_name() -> None:
    with pytest.raises(ValueError, match="unknown RPC method"):
        _StubSam3()._dispatch("segment", (), {})


def test_sam3_dispatch_routes_the_registered_method() -> None:
    result = _StubSam3()._dispatch(
        "sam3.segment", (_image_b64(),), {"text_prompt": "the cup"}
    )
    assert result["found"] is True
    assert result["reason"] == "the cup"


@pytest.mark.parametrize("legacy_health", [False, True], ids=["current", "legacy"])
def test_sam3_checks_candidate_support_without_inference(
    make_server_and_client, monkeypatch, legacy_health
):
    facade = _StubSam3()
    if legacy_health:
        monkeypatch.setattr(_StubSam3, "_builtin_dispatch", RpcFacade._builtin_dispatch)
    candidates = Mock(wraps=facade.segment_candidates)
    facade._rpc["sam3.segment_candidates"] = candidates
    with make_server_and_client(facade, "http") as rpc:
        assert Sam3Client(rpc).supports_segment_candidates()
    assert candidates.call_count == (1 if legacy_health else 0)


def test_sam3_candidate_check_preserves_other_rpc_errors(make_server_and_client):
    facade = RpcFacade()
    facade._rpc["sam3.segment_candidates"] = Mock(side_effect=RuntimeError("busy"))
    with make_server_and_client(facade, "http") as rpc:
        with pytest.raises(RpcError, match="busy"):
            Sam3Client(rpc).supports_segment_candidates()


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({}, "provide exactly one of text_prompt or point"),
        ({"text_prompt": "   "}, "provide exactly one of text_prompt or point"),
        (
            {"text_prompt": "cup", "point": [0, 0]},
            "provide exactly one of text_prompt or point",
        ),
        ({"point": [0]}, r"point must be \[row, col\]"),
        ({"point": "0,0"}, r"point must be \[row, col\]"),
        ({"point": [0, 0], "min_score": "high"}, "min_score must be a number"),
        ({"point": [0, 0], "min_score": 1.5}, "min_score must be between 0 and 1"),
    ],
)
def test_sam3_segment_rejects_bad_arguments(
    kwargs: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _StubSam3().segment(_image_b64(), **kwargs)


@pytest.mark.parametrize("bad_image", [None, 123, b"bytes"])
def test_sam3_segment_rejects_a_non_string_image(bad_image: Any) -> None:
    with pytest.raises(ValueError, match="image_base64 must be a string"):
        _StubSam3().segment(bad_image, text_prompt="cup")


def test_sam3_segment_rejects_an_empty_image() -> None:
    with pytest.raises(ValueError, match="image_base64 is empty"):
        _StubSam3().segment("", text_prompt="cup")


def test_sam3_segment_rejects_invalid_base64() -> None:
    with pytest.raises(binascii.Error):
        _StubSam3().segment("!!!not-base64!!!", text_prompt="cup")


@pytest.mark.parametrize("point", [[4, 0], [0, 5], [-1, 0], [0, -1]])
def test_sam3_segment_rejects_a_point_outside_the_image(point: list[int]) -> None:
    facade = _StubSam3(height=4, width=5)
    with pytest.raises(ValueError, match="outside image shape"):
        facade._segment_bytes(
            b"image-bytes", text_prompt=None, point=point, min_score=0.2
        )


def test_sam3_segment_bytes_routes_text_and_point_prompts() -> None:
    facade = _StubSam3()
    text = facade._segment_bytes(
        b"image-bytes", text_prompt="the cup", point=None, min_score=0.2
    )
    point = facade._segment_bytes(
        b"image-bytes", text_prompt=None, point=[1, 2], min_score=0.2
    )
    assert text.reason == "the cup"
    assert point.reason == "1,2"


def test_sam3_result_to_dict_drops_none_fields() -> None:
    assert Sam3Result(found=False).to_dict() == {"found": False}
    assert Sam3Result(found=True, score=0.5, reason=None).to_dict() == {
        "found": True,
        "score": 0.5,
    }


def test_sam3_client_accepts_a_strided_image_buffer() -> None:
    rpc = Mock()
    rpc.call.return_value = {"found": False}
    image = memoryview(b"pxnxg")[::2]

    Sam3Client(rpc).segment(image, text_prompt="cup")

    assert rpc.call.call_args.kwargs["kwargs"]["image_base64"] == _image_b64(b"png")


def test_sam3_text_candidates_keep_each_qualified_nonempty_instance():
    from types import SimpleNamespace

    import numpy as np

    facade = _StubSam3(height=4, width=5)
    masks = np.zeros((5, 1, 4, 5), dtype=bool)
    masks[:4, 0, 1, :4] = np.eye(4, dtype=bool)
    output = {
        "masks": masks,
        "scores": np.array([0.2, 0.3, 0.9, 0.1, 0.8]),
        "boxes": None,
    }
    facade._processor = SimpleNamespace(set_text_prompt=lambda **_: output)
    facade._torch = None
    client = Sam3Client(
        SimpleNamespace(
            call=lambda method, kwargs, **_: facade._dispatch(method, (), kwargs)
        )
    )
    result = client.segment_candidates(b"image-bytes", text_prompt="cup", min_score=0.2)
    assert [item.score for item in result] == [0.9, 0.3]
    assert result[0].mask[1, 2] and result[1].mask[1, 1]
    assert not client.segment_candidates(
        b"image-bytes", text_prompt="cup", min_score=0.95
    )


def test_sam3_top_result_handles_one_score_and_no_mask():
    import numpy as np

    score = np.array(0.9)
    for masks, found in [(np.ones((4, 5)), True), (np.empty((0, 4, 5)), False)]:
        result = Sam3Facade._select_top(
            masks=masks, scores=score, boxes=None, min_score=0.2
        )
        assert result.found is found
