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

import pytest

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

    def _ground_bytes(self, image_bytes: bytes, query: str) -> MolmoResult:
        return MolmoResult(point_xy=[1.0, 2.0], answer=query, image_size=[4, 4])


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


def test_parse_molmo2_point() -> None:
    assert _parse_point('<points coords="1 125 875"/>') == (125.0, 875.0)


def test_parse_first_of_multiple_points() -> None:
    assert _parse_point('<points coords="1 125 875; 2 500 600"/>') == (
        125.0,
        875.0,
    )


def test_reject_invalid_or_out_of_range_point() -> None:
    assert _parse_point("This isn't in the image.") is None
    assert _parse_point('<points coords="1 1001 200"/>') is None


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
