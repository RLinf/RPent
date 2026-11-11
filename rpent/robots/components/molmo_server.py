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
"""RPC server owning the local Molmo visual-grounding model.

Run manually with::

    MOLMO_CHECKPOINT_PATH=/path/to/molmo PYTHONPATH=/path/to/RPent \
        python -m rpent.robots.components.molmo_server \
        --transport http --host 127.0.0.1 --port 8115

Runs under the ``molmo`` extra's own interpreter, which does not have RPent
installed -- hence the explicit ``PYTHONPATH``.

The ``molmo.ground`` RPC selects an open-vocabulary point over HTTP or socket.
Its default manipulation mode serves Flash. Geometry mode selects a visible
material point on an exact object part using the same loaded Molmo2 model.
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
import os
import re
import threading
from dataclasses import dataclass
from typing import Any, Literal

from PIL import Image

from rpent.utils.logging import get_logger
from rpent.utils.rpc import RpcFacade

logger = get_logger("molmo_server")

#: Molmo2 writes ``point-id x y`` triples, optionally preceded by an image ID;
#: x and y are thousandths zero-padded to three digits.
_COORDS = re.compile(r"<(?:point|points)\b[^>]*\bcoords=[\"']([^\"']+)[\"']", re.I)
_POINT = re.compile(r"[0-9]+[ \t]+([0-9]{3,4})[ \t]+([0-9]{3,4})")
_SINGLE_POINT = re.compile(r"(?:1[ \t]+)?[0-9]+[ \t]+([0-9]{3,4})[ \t]+([0-9]{3,4})")


def _parse_point(
    answer: str, *, require_single: bool = False
) -> tuple[float, float] | None:
    """Read native Molmo2 coordinates, optionally requiring one point/image."""
    coords = list(_COORDS.finditer(answer))
    if not coords or (require_single and len(coords) != 1):
        return None
    point = (
        _SINGLE_POINT.fullmatch(coords[0].group(1).strip())
        if require_single
        else _POINT.search(coords[0].group(1))
    )
    if point is None:
        return None
    x, y = float(point.group(1)), float(point.group(2))
    if not (0 <= x <= 1000 and 0 <= y <= 1000):
        return None
    return x, y


@dataclass
class MolmoResult:
    """Wire response carrying at most one pixel."""

    point_xy: list[float] | None = None
    answer: str | None = None
    image_size: list[int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


class MolmoFacade(RpcFacade):
    """RPC server wrapping the local Molmo visual-grounding model."""

    def __init__(self, checkpoint: str) -> None:
        super().__init__()
        self._load(checkpoint)
        self._register_rpc()

    def _load(self, checkpoint: str) -> None:
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "local Molmo dependencies are missing; install RPent with "
                '`pip install -e ".[molmo]"`'
            ) from exc

        if not torch.cuda.is_available():
            raise RuntimeError("local Molmo requires a CUDA-capable GPU")
        processor = AutoProcessor.from_pretrained(
            checkpoint, trust_remote_code=True, local_files_only=True
        )
        model = (
            AutoModelForImageTextToText.from_pretrained(
                checkpoint,
                trust_remote_code=True,
                local_files_only=True,
                dtype=torch.bfloat16,
            )
            .to("cuda")
            .eval()
        )
        self._torch = torch
        self._model = model
        self._processor = processor
        self._lock = threading.Lock()

    def _register_rpc(self) -> None:
        self._rpc["molmo.ground"] = self.ground
        self._readonly_methods.add("molmo.ground")

    def _ground_bytes(
        self,
        image_bytes: bytes,
        query: str,
        *,
        point_mode: Literal["manipulation", "geometry"] = "manipulation",
    ) -> MolmoResult:
        """Run one grounding prompt against the loaded model."""
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        width, height = image.size
        if point_mode == "geometry":
            prompt = (
                "Point to one visible material point on the exact object and "
                "part described below. Match the object's appearance and spatial "
                "relations to distinguish it from similar objects. Select a "
                "pixel on the requested physical surface or boundary, not a "
                "grasp point or an empty geometric center. For a rim, select "
                "rim material, not the empty opening or interior floor; an inner "
                "rim means the inner upper edge. For a lid or top face, select "
                "that face, not the side wall or its label. A requested edge "
                "must stay on that edge. Do not substitute another part or "
                "object. If the target or requested part is absent, fully "
                "occluded, or cannot be uniquely identified, reply 'not found' "
                "without point markup. Otherwise return exactly one point in "
                "native Molmo2 point markup, with no explanation.\n"
                f"Target and part: {query}"
            )
        else:
            prompt = (
                f"Point to {query} in this robot camera image. Choose the final "
                "safe manipulation point yourself, on visible object surface and "
                "away from edges. Return one point only."
            )
        inputs = self._processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "image": image},
                    ],
                }
            ],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = {key: value.to(self._model.device) for key, value in inputs.items()}
        with self._lock, self._torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=96 if point_mode == "geometry" else 48,
                do_sample=False,
            )
        answer = self._processor.tokenizer.decode(
            generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=False
        )
        normalized = _parse_point(answer, require_single=point_mode == "geometry")
        point = None
        if normalized is not None:
            x, y = normalized
            point = [
                x / 1000 * width,
                y / 1000 * height,
            ]
            if point_mode == "geometry":
                # Molmo2's 1000 denotes the image edge, not a valid array index.
                point = [min(width - 1, point[0]), min(height - 1, point[1])]
        return MolmoResult(point_xy=point, answer=answer, image_size=[width, height])

    def ground(
        self,
        image_base64: str,
        query: str,
        *,
        point_mode: Literal["manipulation", "geometry"] = "manipulation",
    ) -> dict[str, Any]:
        if point_mode not in ("manipulation", "geometry"):
            raise ValueError("point_mode must be manipulation or geometry")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("ground requires a non-empty query")
        query = query.strip()
        if not isinstance(image_base64, str):
            raise ValueError("image_base64 must be a string")

        image_bytes = base64.b64decode(image_base64, validate=True)
        if not image_bytes:
            raise ValueError("image_base64 is empty")
        return self._ground_bytes(image_bytes, query, point_mode=point_mode).to_dict()


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RPent local Molmo server")
    parser.add_argument("--transport", choices=["socket", "http"], default="http")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8115)
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=None,
        help="GPU device exposed through CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument(
        "--parent-watch",
        action="store_true",
        help="watch parent process via stdin pipe and exit when it dies",
    )
    return parser


def main() -> None:
    """Load Molmo and serve until terminated."""
    args = _build_argparser().parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    if args.cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
    checkpoint = os.environ.get("MOLMO_CHECKPOINT_PATH")
    if not checkpoint:
        raise RuntimeError(
            "MOLMO_CHECKPOINT_PATH is not set; export the path to the Molmo "
            "weights before starting RPent"
        )
    facade = MolmoFacade(checkpoint)
    facade.serve(
        transport=args.transport,
        host=args.host,
        port=args.port,
        parent_watch=args.parent_watch,
    )


if __name__ == "__main__":
    main()
