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

Where SAM3 answers "which pixels are this phrase", Molmo answers "where would
you put the gripper" -- an open-vocabulary point on a named object, for phrases
no mask proposal names. The service exposes a ``ground`` RPC method over either
HTTP or socket transport.
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
import os
import re
import threading
from typing import Any

from PIL import Image
from pydantic import BaseModel, Field

from rpent.utils.logging import get_logger
from rpent.utils.rpc import RpcFacade

logger = get_logger("molmo_server")

#: Molmo2 writes one or more ``point-id x y`` triples in normalized thousandths.
_COORDS = re.compile(r"<(?:point|points)\b[^>]*\bcoords=[\"']([^\"']+)[\"']", re.I)
_POINT = re.compile(r"(?:^|[\t:;,])\s*\d+\s+([0-9]{1,4})\s+([0-9]{1,4})")


def _parse_point(answer: str) -> tuple[float, float] | None:
    """Return the first normalized Molmo2 point from generated markup."""
    coords = _COORDS.search(answer)
    if coords is None:
        return None
    point = _POINT.search(coords.group(1))
    if point is None:
        return None
    x, y = float(point.group(1)), float(point.group(2))
    if not (0 <= x <= 1000 and 0 <= y <= 1000):
        return None
    return x, y


class GroundRequest(BaseModel):
    """Wire request naming one object to point at."""

    image_base64: str
    query: str = Field(min_length=1)


class GroundResponse(BaseModel):
    """Wire response carrying at most one pixel."""

    point_xy: list[float] | None = None
    answer: str | None = None
    image_size: list[int] | None = None


class MolmoEngine:
    """Serialize Molmo inference behind one lock."""

    def __init__(self, model: Any, processor: Any) -> None:
        self._model = model
        self._processor = processor
        self._lock = threading.Lock()

    @classmethod
    def load(cls, checkpoint: str) -> "MolmoEngine":
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "local Molmo dependencies are missing; install RPent with "
                '`pip install -e ".[full]"`'
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
        return cls(model, processor)

    def ground(self, image_bytes: bytes, query: str) -> GroundResponse:
        import torch

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        width, height = image.size
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
        with self._lock, torch.inference_mode():
            generated = self._model.generate(
                **inputs, max_new_tokens=48, do_sample=False
            )
        answer = self._processor.tokenizer.decode(
            generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=False
        )
        normalized = _parse_point(answer)
        point = None
        if normalized is not None:
            x, y = normalized
            point = [
                x / 1000 * width,
                y / 1000 * height,
            ]
        return GroundResponse(point_xy=point, answer=answer, image_size=[width, height])


class MolmoFacade(RpcFacade):
    """Expose :class:`MolmoEngine` through the shared RPC transports."""

    def __init__(self, engine: MolmoEngine) -> None:
        super().__init__()
        self._engine = engine

    def _dispatch(self, method: str, args: tuple, kwargs: dict) -> Any:
        if method == "ground":
            return self.ground(*args, **kwargs)
        return super()._dispatch(method, args, kwargs)

    def ground(self, image_base64: str, query: str) -> dict[str, Any]:
        request = GroundRequest(image_base64=image_base64, query=query)
        image_bytes = base64.b64decode(request.image_base64, validate=True)
        if not image_bytes:
            raise ValueError("image_base64 is empty")
        return self._engine.ground(image_bytes, request.query).model_dump(
            exclude_none=True
        )


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
    facade = MolmoFacade(MolmoEngine.load(checkpoint))
    facade.serve(
        transport=args.transport,
        host=args.host,
        port=args.port,
        parent_watch=args.parent_watch,
    )


if __name__ == "__main__":
    main()
