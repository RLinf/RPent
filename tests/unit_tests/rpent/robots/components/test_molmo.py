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
import io
import threading
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from rpent.robots.components.molmo_client import MolmoClient
from rpent.robots.components.molmo_server import (
    MolmoFacade,
    MolmoResult,
    _parse_point,
)


class _StubMolmo(MolmoFacade):
    """MolmoFacade with the CUDA checkpoint load and inference stubbed out."""

    def _load(self, checkpoint: str) -> None:
        self._torch = None
        self._model = None
        self._processor = None
        self._lock = threading.Lock()

    def _ground_bytes(
        self, image_bytes: bytes, query: str, *, point_mode="manipulation"
    ) -> MolmoResult:
        return MolmoResult(point_xy=[1.0, 2.0], answer=query, image_size=[4, 4])


class _InferenceMolmo(MolmoFacade):
    """Exercise real prompting and pixel conversion without a GPU checkpoint."""

    def __init__(self, answer):
        self.answer = answer
        super().__init__("unused-checkpoint")

    def _load(self, checkpoint):
        self._lock = threading.Lock()
        self._torch = SimpleNamespace(inference_mode=nullcontext)
        input_ids = SimpleNamespace(shape=(1, 2))
        input_ids.to = lambda device: input_ids

        def apply_chat_template(messages, **kwargs):
            self.messages = messages
            return {"input_ids": input_ids}

        def generate(**kwargs):
            import numpy as np

            self.generation_options = kwargs
            return np.array([[0, 0, 1]])

        self._model = SimpleNamespace(device="cpu", generate=generate)
        self._processor = SimpleNamespace(
            apply_chat_template=apply_chat_template,
            tokenizer=SimpleNamespace(decode=lambda *a, **kw: self.answer),
        )


class _LoopbackRpcClient:
    def __init__(self, facade):
        self.facade = facade
        self.calls = []

    def call(self, method, *, kwargs=None, timeout_s=None):
        self.calls.append((method, kwargs))
        return self.facade._dispatch(method, (), kwargs or {})


class _FakeRpcClient:
    """Records calls and replays a canned payload."""

    def __init__(self, payload: Any) -> None:
        self._payload = payload
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def call(self, method: str, *, kwargs: dict | None = None, timeout_s: Any = None):
        self.calls.append((method, kwargs or {}))
        return self._payload


def _image_b64(payload: bytes = b"image-bytes") -> str:
    return base64.b64encode(payload).decode("ascii")


def _png():
    buffer = io.BytesIO()
    Image.new("RGB", (800, 400)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_parse_molmo2_point() -> None:
    assert _parse_point('<points coords="1 1 399 756"/>') == (399.0, 756.0)
    assert _parse_point('<points coords="1 399 756"/>') == (399.0, 756.0)


def test_parse_first_of_multiple_points() -> None:
    assert _parse_point('<points coords="1 1 125 875; 2 500 600"/>') == (
        125.0,
        875.0,
    )
    assert _parse_point('<points coords="1 125 875; 2 500 600"/>') == (
        125.0,
        875.0,
    )


def test_reject_invalid_or_out_of_range_point() -> None:
    assert _parse_point("This isn't in the image.") is None
    assert _parse_point('<points coords="1 1 1001 200"/>') is None
    assert _parse_point('<points coords="1 1 399"/>') is None
    assert _parse_point('<points coords="1 399"/>') is None


def test_molmo_ground_is_a_readonly_namespaced_rpc() -> None:
    facade = _StubMolmo("unused-checkpoint")
    assert facade._rpc["molmo.ground"] == facade.ground
    assert "molmo.ground" in facade._readonly_methods
    # The bare name is not a valid RPC method.
    assert "ground" not in facade._rpc


def test_molmo_dispatch_rejects_the_bare_method_name() -> None:
    with pytest.raises(ValueError, match="unknown RPC method"):
        _StubMolmo("unused-checkpoint")._dispatch("ground", (), {})


def test_molmo_dispatch_routes_the_registered_method() -> None:
    result = _StubMolmo("unused-checkpoint")._dispatch(
        "molmo.ground", (_image_b64(),), {"query": "the cup"}
    )
    assert result == {"point_xy": [1.0, 2.0], "answer": "the cup", "image_size": [4, 4]}


@pytest.mark.parametrize("bad_query", [None, 123, "", "   "])
def test_molmo_ground_requires_a_nonempty_query(bad_query: Any) -> None:
    with pytest.raises(ValueError, match="ground requires a non-empty query"):
        _StubMolmo("unused-checkpoint").ground(_image_b64(), bad_query)


@pytest.mark.parametrize("bad_image", [None, 123, b"bytes"])
def test_molmo_ground_rejects_a_non_string_image(bad_image: Any) -> None:
    with pytest.raises(ValueError, match="image_base64 must be a string"):
        _StubMolmo("unused-checkpoint").ground(bad_image, "the cup")


def test_molmo_ground_rejects_an_empty_image() -> None:
    with pytest.raises(ValueError, match="image_base64 is empty"):
        _StubMolmo("unused-checkpoint").ground("", "the cup")


def test_molmo_ground_rejects_invalid_base64() -> None:
    with pytest.raises(binascii.Error):
        _StubMolmo("unused-checkpoint").ground("!!!not-base64!!!", "the cup")


def test_molmo_result_to_dict_drops_none_fields() -> None:
    assert MolmoResult().to_dict() == {}
    assert MolmoResult(point_xy=[1.0, 2.0], answer="x").to_dict() == {
        "point_xy": [1.0, 2.0],
        "answer": "x",
    }


def test_molmo_client_calls_the_namespaced_rpc_method() -> None:
    client = _FakeRpcClient({"point_xy": [10.0, 20.0], "image_size": [100, 200]})
    result = MolmoClient(client).ground(b"fake-png", "the cup")
    method, kwargs = client.calls[0]
    assert method == "molmo.ground"
    assert kwargs["query"] == "the cup"
    assert kwargs["image_base64"] == base64.b64encode(b"fake-png").decode("ascii")
    assert set(kwargs) == {"query", "image_base64"}
    assert result.found is True
    assert result.point_xy == (10.0, 20.0)
    assert result.image_size == (100, 200)
    assert result.answer is None


def test_molmo_client_reports_a_miss_without_a_point() -> None:
    client = _FakeRpcClient({"answer": "no object here"})
    result = MolmoClient(client).ground(b"fake-png", "the cup")
    assert result.found is False
    assert result.point_xy is None
    assert result.answer == "no object here"


def test_geometry_mode_preserves_material_part_query_and_original_resolution():
    facade = _InferenceMolmo('<points coords="1 1 125 750"/>')
    rpc = _LoopbackRpcClient(facade)
    query = "the inner upper rim of the blue-and-white cup left of the red box"
    result = MolmoClient(rpc, point_mode="geometry").ground(_png(), query)
    assert result.found and result.image_size == (800, 400)
    assert result.point_xy == (100, 300)  # col scales by width; row by height.
    assert rpc.calls[0][1]["point_mode"] == "geometry"
    prompt = facade.messages[0]["content"][0]["text"]
    assert prompt.endswith("Target and part: " + query)
    assert "not a grasp point or an empty geometric center" in prompt
    assert "inner upper edge" in prompt and "without point markup" in prompt
    assert "away from edges" not in prompt
    assert facade.generation_options["max_new_tokens"] == 96
    assert facade.generation_options["do_sample"] is False


@pytest.mark.parametrize(
    "answer,expected",
    [
        ('<points coords="1 000 000"/>', (0, 0)),
        ('<points coords="1 1 1000 1000"/>', (799, 399)),
        ("not found", None),
        ('<points coords="1 1 1001 500"/>', None),
        ('<points coords="1 125 750; 2 500 600"/>', None),
        ('<points coords="1 125 750"/><points coords="1 500 600"/>', None),
        ('<points coords="2 1 125 750"/>', None),
    ],
)
def test_geometry_mode_handles_border_and_unresolved_points(answer, expected):
    result = MolmoClient(
        _LoopbackRpcClient(_InferenceMolmo(answer)), point_mode="geometry"
    ).ground(_png(), "the sealed top lid of the blue-and-yellow soup can")
    assert result.found is (expected is not None)
    assert result.point_xy == expected
    assert result.image_size == (800, 400)
    assert result.answer == answer


def test_default_flash_prompt_generation_and_first_point_are_unchanged():
    facade = _InferenceMolmo('<points coords="1 125 750; 2 500 600"/>')
    rpc = _LoopbackRpcClient(facade)
    result = MolmoClient(rpc).ground(_png(), "the cup")
    assert result.point_xy == (100, 300)
    assert set(rpc.calls[0][1]) == {"query", "image_base64"}
    assert facade.messages[0]["content"][0]["text"] == (
        "Point to the cup in this robot camera image. Choose the final "
        "safe manipulation point yourself, on visible object surface and "
        "away from edges. Return one point only."
    )
    assert facade.generation_options["max_new_tokens"] == 48
    assert facade.generation_options["do_sample"] is False


def test_reject_unknown_mode_on_client_and_rpc():
    with pytest.raises(ValueError, match="point_mode"):
        MolmoClient(_FakeRpcClient({}), point_mode="other")
    with pytest.raises(ValueError, match="point_mode"):
        _StubMolmo("unused").ground(_image_b64(), "the cup", point_mode="other")


@pytest.mark.parametrize("point", [[float("nan"), 1], [1, "invalid"]])
def test_molmo_client_rejects_invalid_coordinates(point):
    with pytest.raises(RuntimeError, match="invalid Molmo point_xy"):
        MolmoClient(_FakeRpcClient({"point_xy": point})).ground(_png(), "the cup")
