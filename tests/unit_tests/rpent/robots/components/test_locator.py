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
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from rpent.robots.components.locator_client import LocatorClient
from rpent.robots.components.locator_server import (
    LocatorFacade,
    Prediction,
    _build_argparser,
    _build_backend,
    _parse_point,
)
from rpent.utils.rpc import RpcError, make_rpc_client
from rpent.utils.rpc.http_rpc import HttpRpcServer
from rpent.utils.rpc.socket_rpc import SocketRpcServer


def _image_b64():
    buffer = io.BytesIO()
    Image.new("RGB", (200, 100)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ('<points coords="1 125 875"/>', (125, 875)),
        ('<points coords="1 125 875; 2 500 600"/>', (125, 875)),
        ('<point coords="1 0 1000"/>', (0, 1000)),
        ("This isn't in the image.", None),
        ('<points coords="1 1001 200"/>', None),
    ],
)
def test_molmo_point_markup(answer, expected):
    assert _parse_point(answer) == expected


def test_client_server_roundtrip_preserves_prompt_and_original_pixels():
    backend = Mock()
    backend.predict.return_value = Prediction((125, 875), "native answer")
    facade = LocatorFacade(backend)
    rpc = Mock()
    rpc.call.side_effect = lambda method, kwargs, **_: facade._dispatch(
        method, (), kwargs
    )
    point = LocatorClient(rpc).locate(
        np.zeros((100, 200, 3), dtype=np.uint8), " the cup "
    )
    assert point == (25.0, 87.5)
    image, prompt = backend.predict.call_args.args
    assert image.size == (200, 100)
    assert image.mode == "RGB"
    assert prompt == (
        "Point to the cup in this robot camera image. Choose the final "
        "safe manipulation point yourself, on visible object surface and "
        "away from edges. Return one point only."
    )
    assert rpc.call.call_args.kwargs["timeout_s"] == 180.0


@pytest.mark.parametrize(
    "transport, server_type", [("http", HttpRpcServer), ("socket", SocketRpcServer)]
)
def test_locator_over_real_rpc_transports(transport, server_type):
    backend = Mock(predict=Mock(return_value=Prediction((125, 875), "answer")))
    facade = LocatorFacade(backend)
    with server_type(("127.0.0.1", 0), facade._dispatch) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        rpc = make_rpc_client(f"{transport}://127.0.0.1:{server.server_address[1]}")
        client = LocatorClient(rpc)
        try:
            image = np.zeros((100, 200, 3), dtype=np.uint8)
            assert client.locate(image, "cup") == (25, 87.5)
            backend.predict.return_value = Prediction(None, "absent")
            assert client.locate(image, "cup") is None
            backend.predict.side_effect = RuntimeError("model unavailable")
            with pytest.raises(RpcError, match="model unavailable"):
                client.locate(image, "cup")
        finally:
            rpc.close()
            server.shutdown()
            thread.join(timeout=2)
        assert not thread.is_alive()


def test_locator_dispatch_and_miss():
    facade = LocatorFacade(
        SimpleNamespace(predict=lambda *_: Prediction(None, "absent"))
    )
    assert facade._dispatch("locator.locate", (_image_b64(), "cup"), {}) == {
        "point_xy": None,
        "answer": "absent",
        "image_size": [200, 100],
    }
    with pytest.raises(ValueError, match="unknown RPC method"):
        facade._dispatch("locate", (), {})
    assert (
        LocatorClient(Mock(call=Mock(return_value={"point_xy": None}))).locate(
            b"png", "cup"
        )
        is None
    )


@pytest.mark.parametrize("query", [None, 123, "", "   "])
def test_empty_query_is_rejected_by_client_and_server(query):
    with pytest.raises(ValueError, match="non-empty query"):
        LocatorClient(Mock()).locate(b"png", query)
    with pytest.raises(ValueError, match="non-empty query"):
        LocatorFacade(Mock()).locate(_image_b64(), query)


@pytest.mark.parametrize("image", [None, 123, b"bytes"])
def test_server_rejects_non_string_image(image):
    with pytest.raises(ValueError, match="must be a string"):
        LocatorFacade(Mock()).locate(image, "cup")


@pytest.mark.parametrize("image", ["", "!!!not-base64!!!"])
def test_server_rejects_empty_or_invalid_base64(image):
    with pytest.raises((ValueError, binascii.Error)):
        LocatorFacade(Mock()).locate(image, "cup")


def test_server_rejects_non_image_bytes_before_calling_model():
    backend = Mock()
    with pytest.raises(OSError):
        LocatorFacade(backend).locate(base64.b64encode(b"not an image").decode(), "cup")
    backend.predict.assert_not_called()


@pytest.mark.parametrize(
    "point",
    [
        (float("nan"), 20),
        (10, float("inf")),
        (True, 4),
        (-1, 500),
        (500, 1001),
        ("10", 20),
        (1,),
    ],
)
def test_server_rejects_invalid_model_coordinates(point):
    backend = Mock(predict=Mock(return_value=Prediction(point, "bad")))
    with pytest.raises(ValueError):
        LocatorFacade(backend).locate(_image_b64(), "cup")


@pytest.mark.parametrize(
    "payload", [None, {}, {"point_xy": [1]}, {"point_xy": [float("nan"), 2]}]
)
def test_client_rejects_malformed_response(payload):
    with pytest.raises((RuntimeError, ValueError)):
        LocatorClient(Mock(call=Mock(return_value=payload))).locate(b"png", "cup")


def test_model_error_is_not_a_target_miss():
    backend = Mock(predict=Mock(side_effect=TimeoutError("model timed out")))
    with pytest.raises(TimeoutError, match="timed out"):
        LocatorFacade(backend).locate(_image_b64(), "cup")


@pytest.mark.parametrize(
    "args",
    [
        ["--backend", "api"],
        ["--backend", "codex", "--base-url", "https://example.invalid"],
        ["--backend", "api", "--model", "x", "--cuda-device", "0"],
        ["--backend", "api", "--model", "x", "--reasoning-effort", "low"],
        ["--backend", "molmo", "--model", "x"],
    ],
)
def test_invalid_backend_configuration_is_rejected(args):
    with pytest.raises(ValueError):
        _build_backend(_build_argparser().parse_args(args))


def test_removed_server_timeout_flag_is_rejected():
    with pytest.raises(SystemExit):
        _build_argparser().parse_args(["--timeout", "120"])
