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

"""Single-point locator RPC service with Molmo, API, and Codex backends.

Run ``python -m rpent.robots.components.locator_server --help`` for configuration.
Model credentials and optional dependencies belong to this server's environment.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import logging
import os
import re
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Any, Protocol

from PIL import Image

from rpent.robots.components.locator_client import validate_point
from rpent.utils.logging import get_logger
from rpent.utils.rpc import RpcFacade

logger = get_logger("locator_server")


def manipulation_prompt(query: str) -> str:
    """Build the manipulation prompt used by every locator backend."""
    return (
        f"Point to {query} in this robot camera image. Choose the final "
        "safe manipulation point yourself, on visible object surface and "
        "away from edges. Return one point only."
    )


JSON_POINT_INSTRUCTIONS = (
    '\nReturn only JSON: {"point": [x, y]}. Coordinates are normalized to '
    "0–1000 across the full input image, with the origin at the top left; "
    "x increases rightward and y downward. "
    'If the target cannot be located, return {"point": null}.'
)


def point_schema() -> dict[str, Any]:
    """Return the JSON output schema used by structured-output backends."""
    return {
        "type": "object",
        "properties": {
            "point": {
                "anyOf": [
                    {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    {"type": "null"},
                ]
            }
        },
        "required": ["point"],
        "additionalProperties": False,
    }


def parse_point_json(answer: str) -> tuple[float, float] | None:
    """Parse a JSON answer; malformed output is an error, not a target miss."""
    payload = json.loads(answer)
    if not isinstance(payload, dict) or set(payload) != {"point"}:
        raise ValueError('locator response must contain exactly one "point" field')
    return None if payload["point"] is None else validate_point(payload["point"])


@dataclass(frozen=True)
class Prediction:
    """A normalized point, or a miss, together with the model's answer."""

    point: tuple[float, float] | None
    answer: str


class LocatorBackend(Protocol):
    def predict(self, image: Image.Image, prompt: str) -> Prediction:
        """Locate one point in ``image`` using the supplied manipulation prompt."""
        ...


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


class MolmoBackend:
    """Molmo2 running in its own CUDA environment."""

    def __init__(self, checkpoint: str) -> None:
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

    def predict(self, image: Image.Image, prompt: str) -> Prediction:
        """Generate Molmo's native point markup without changing its prompt."""
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
                **inputs, max_new_tokens=48, do_sample=False
            )
        answer = self._processor.tokenizer.decode(
            generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=False
        )
        return Prediction(_parse_point(answer), answer)


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class ApiBackend:
    """One image request through RPent's existing API provider resolution."""

    def __init__(self, model: str, *, base_url: str | None = None) -> None:
        self._model = model
        self._base_url = base_url

    def predict(self, image: Image.Image, prompt: str) -> Prediction:
        """Request JSON coordinates with no conversation history or robot tools."""
        answer = asyncio.run(self._request(_png_bytes(image), prompt))
        return Prediction(parse_point_json(answer), answer)

    async def _request(self, image: bytes, prompt: str) -> str:
        from pydantic_ai import Agent, BinaryContent

        from rpent.planner.base import build_api_model

        model = build_api_model(self._model, self._base_url)
        async with Agent(model, output_type=str, retries=0) as agent:
            result = await agent.run(
                [
                    prompt + JSON_POINT_INSTRUCTIONS,
                    BinaryContent(data=image, media_type="image/png"),
                ],
            )
        return result.output


class CodexBackend:
    """Image localization using the server machine's existing Codex login."""

    def __init__(
        self,
        model: str | None = None,
        *,
        reasoning_effort: str = "none",
    ) -> None:
        self._model = model
        self._reasoning_effort = reasoning_effort

    def predict(self, image: Image.Image, prompt: str) -> Prediction:
        """Locate in an isolated Codex turn using the shared runner."""
        import openai_codex
        from openai_codex.generated.v2_all import ReasoningEffort, TurnStatus

        from rpent.planner.codex import build_codex_config, run_codex_turn

        data_url = "data:image/png;base64," + base64.b64encode(
            _png_bytes(image)
        ).decode("ascii")
        with tempfile.TemporaryDirectory(prefix="rpent-locator-") as cwd:
            result = run_codex_turn(
                build_codex_config(mcp_url=None, base_url=None, api_key=None, cwd=cwd),
                input=[
                    openai_codex.TextInput(prompt + JSON_POINT_INSTRUCTIONS),
                    openai_codex.ImageInput(data_url),
                ],
                model=self._model,
                thread_options={
                    "cwd": cwd,
                    "ephemeral": True,
                    "base_instructions": (
                        "Locate the requested point in the supplied image. "
                        "Respond directly with JSON. Do not use tools."
                    ),
                    "config": {"features.shell_tool": False, "web_search": "disabled"},
                },
                turn_options={
                    "effort": ReasoningEffort(self._reasoning_effort),
                    "output_schema": point_schema(),
                },
            )
        if result.status != TurnStatus.completed:
            raise RuntimeError(f"Codex locator turn {result.status}: {result.error}")
        if not result.final_response:
            raise RuntimeError("Codex locator returned no final response")
        return Prediction(
            parse_point_json(result.final_response), result.final_response
        )


class LocatorFacade(RpcFacade):
    """Decode images, run the selected backend, and return original-image pixels."""

    def __init__(self, backend: LocatorBackend) -> None:
        super().__init__()
        self._backend = backend
        self._rpc["locator.locate"] = self.locate
        self._readonly_methods.add("locator.locate")

    def locate(self, image_base64: str, query: str) -> dict[str, Any]:
        """Locate one target; backend failures propagate through the RPC envelope."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("locate requires a non-empty query")
        if not isinstance(image_base64, str):
            raise ValueError("image_base64 must be a string")
        image_bytes = base64.b64decode(image_base64, validate=True)
        if not image_bytes:
            raise ValueError("image_base64 is empty")
        with Image.open(io.BytesIO(image_bytes)) as source:
            image = source.convert("RGB")
        prompt = manipulation_prompt(query.strip())
        started = time.monotonic()
        prediction = self._backend.predict(image, prompt)
        point = None
        if prediction.point is not None:
            x, y = validate_point(prediction.point)
            if not (0 <= x <= 1000 and 0 <= y <= 1000):
                raise ValueError(f"locator coordinates outside 0–1000: {(x, y)}")
            # Preserve Molmo's normalized-thousandths convention, including
            # its boundary values, for all backends.
            point = [x / 1000 * image.width, y / 1000 * image.height]
        result = {
            "point_xy": point,
            "answer": prediction.answer,
            "image_size": list(image.size),
        }
        logger.info(
            "%s",
            json.dumps(
                {
                    "backend": type(self._backend).__name__,
                    "prompt": prompt,
                    **result,
                    "elapsed_s": round(time.monotonic() - started, 3),
                },
                ensure_ascii=False,
            ),
        )
        return result


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RPent point locator server")
    parser.add_argument("--backend", choices=["molmo", "api", "codex"], default="molmo")
    parser.add_argument("--checkpoint", default=os.environ.get("MOLMO_CHECKPOINT_PATH"))
    parser.add_argument("--model", help="Provider:model for API; model name for Codex.")
    parser.add_argument("--base-url", help="API provider URL override.")
    parser.add_argument(
        "--reasoning-effort", choices=["none", "low", "medium", "high", "xhigh"]
    )
    parser.add_argument("--transport", choices=["socket", "http"], default="http")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8115)
    parser.add_argument("--cuda-device", type=int, help="GPU exposed to Molmo.")
    parser.add_argument("--parent-watch", action="store_true")
    return parser


def _build_backend(args: argparse.Namespace) -> LocatorBackend:
    if args.base_url is not None and args.backend != "api":
        raise ValueError("--base-url requires --backend api")
    if args.reasoning_effort is not None and args.backend != "codex":
        raise ValueError("--reasoning-effort requires --backend codex")
    if args.cuda_device is not None and args.backend != "molmo":
        raise ValueError("--cuda-device requires --backend molmo")
    if args.backend == "molmo":
        if args.model is not None:
            raise ValueError("Molmo uses --checkpoint, not --model")
        if not args.checkpoint:
            raise ValueError("Molmo requires --checkpoint or MOLMO_CHECKPOINT_PATH")
        if args.cuda_device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
        return MolmoBackend(args.checkpoint)
    if args.backend == "api":
        if not args.model:
            raise ValueError("--backend api requires --model provider:model")
        return ApiBackend(args.model, base_url=args.base_url)
    return CodexBackend(
        args.model,
        reasoning_effort=args.reasoning_effort or "none",
    )


def main() -> None:
    """Start one selected backend and serve until terminated."""
    parser = _build_argparser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    try:
        backend = _build_backend(args)
    except ValueError as exc:
        parser.error(str(exc))
    logger.info(
        "backend=%s model=%s checkpoint=%s",
        args.backend,
        args.model or "default",
        args.checkpoint if args.backend == "molmo" else None,
    )
    LocatorFacade(backend).serve(
        transport=args.transport,
        host=args.host,
        port=args.port,
        parent_watch=args.parent_watch,
    )


if __name__ == "__main__":
    main()
