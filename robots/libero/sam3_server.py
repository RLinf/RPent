"""RPC server owning the local SAM 3.0 image segmentation model.

Run manually with::

    SAM3_CHECKPOINT_PATH=/path/to/sam3.pt \
        python -m robots.libero.sam3_server \
        --transport http --host 127.0.0.1 --port 8114

RPent normally starts this process automatically. The service exposes a
``segment`` RPC method over either HTTP or socket transport.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import logging
import os
import threading
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from rpent.utils.logging import get_logger
from rpent.utils.rpc import RpcFacade

logger = get_logger("sam3_server")


@dataclass
class Sam3Result:
    """Segmentation result with at most one compressed binary mask."""

    found: bool
    score: float | None = None
    box: list[float] | None = None
    mask_png_base64: str | None = None
    mask_shape: list[int] | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


def _encode_mask_png(mask: np.ndarray) -> str:
    image = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class Sam3Facade(RpcFacade):
    """RPC server wrapping the local SAM 3.0 image segmentation model."""

    def __init__(self, checkpoint: str) -> None:
        super().__init__()
        self._load(checkpoint)
        self._register_rpc()

    def _load(self, checkpoint: str) -> None:
        """Load the official SAM 3.0 model and interactive point head."""
        try:
            import torch
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model
        except ImportError as exc:
            raise RuntimeError(
                "local SAM3 dependencies are missing; install RPent with "
                '`pip install -e ".[sam3]"` (or `.[full]`)'
            ) from exc

        if not torch.cuda.is_available():
            raise RuntimeError("local SAM3 requires a CUDA-capable GPU")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.set_device(0)

        resolved = Path(checkpoint).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"SAM3 checkpoint not found: {resolved}")
        checkpoint_path = str(resolved)

        logger.info("loading SAM 3.0 checkpoint: %s", checkpoint_path)
        try:
            model = build_sam3_image_model(
                device="cuda",
                checkpoint_path=checkpoint_path,
                load_from_HF=False,
                enable_inst_interactivity=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"failed to load SAM 3.0 checkpoint: {checkpoint_path}"
            ) from exc
        self._torch = torch
        self._model = model
        self._processor = Sam3Processor(model, device="cuda", confidence_threshold=0.0)
        self._device = "cuda"
        self._lock = threading.Lock()
        self._image_digest: str | None = None
        self._image_state: dict[str, Any] | None = None

    def _register_rpc(self) -> None:
        self._rpc["segment"] = self.segment
        self._readonly_methods.add("segment")

    def _segment_bytes(
        self,
        image_bytes: bytes,
        *,
        text_prompt: str | None,
        point: list[int] | None,
        min_score: float,
    ) -> Sam3Result:
        """Run one prompt against the cached latest-image features."""
        with self._lock:
            state = self._state_for_image(image_bytes)
            height = int(state["original_height"])
            width = int(state["original_width"])
            if text_prompt is not None:
                return self._segment_text(state, text_prompt, min_score)
            assert point is not None
            row, col = point
            if row < 0 or col < 0 or row >= height or col >= width:
                raise ValueError(
                    f"point [row, col] {point} is outside image shape "
                    f"[{height}, {width}]"
                )
            return self._segment_point(state, row, col, min_score)

    def _inference_context(self):
        if self._torch is None or not self._device.startswith("cuda"):
            return nullcontext()
        return self._torch.autocast("cuda", dtype=self._torch.bfloat16)

    def _state_for_image(self, image_bytes: bytes) -> dict[str, Any]:
        digest = hashlib.sha256(image_bytes).hexdigest()
        if digest == self._image_digest and self._image_state is not None:
            return self._image_state
        try:
            with Image.open(io.BytesIO(image_bytes)) as source:
                image = source.convert("RGB")
        except Exception as exc:
            raise ValueError(f"invalid image data: {exc}") from exc
        with self._inference_context():
            state = self._processor.set_image(image)
        self._image_digest = digest
        self._image_state = state
        return state

    def _segment_text(
        self,
        state: dict[str, Any],
        prompt: str,
        min_score: float,
    ) -> Sam3Result:
        with self._inference_context():
            output = self._processor.set_text_prompt(prompt=prompt, state=state)
        return self._select_top(
            masks=output.get("masks"),
            scores=output.get("scores"),
            boxes=output.get("boxes"),
            min_score=min_score,
        )

    def _segment_point(
        self,
        state: dict[str, Any],
        row: int,
        col: int,
        min_score: float,
    ) -> Sam3Result:
        if getattr(self._model, "inst_interactive_predictor", None) is None:
            raise RuntimeError("SAM3 instance interactivity is not enabled")
        point_coords = np.asarray([[col, row]], dtype=np.float32)
        point_labels = np.asarray([1], dtype=np.int64)
        with self._inference_context():
            masks, scores, _ = self._model.predict_inst(
                state,
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )
        return self._select_top(
            masks=masks,
            scores=scores,
            boxes=None,
            min_score=min_score,
        )

    @staticmethod
    def _select_top(
        *,
        masks: Any,
        scores: Any,
        boxes: Any,
        min_score: float,
    ) -> Sam3Result:
        if masks is None or scores is None:
            return Sam3Result(found=False, reason="SAM3 returned no candidate")
        masks_array = (
            masks
            if isinstance(masks, np.ndarray)
            else masks.detach().float().cpu().numpy()
        )
        scores_array = (
            scores
            if isinstance(scores, np.ndarray)
            else scores.detach().float().cpu().numpy()
        ).reshape(-1)
        if masks_array.size == 0 or scores_array.size == 0:
            return Sam3Result(found=False, reason="SAM3 returned no candidate")
        if masks_array.ndim == 4 and masks_array.shape[1] == 1:
            masks_array = masks_array[:, 0]
        if masks_array.ndim == 2:
            masks_array = masks_array[None]
        if masks_array.ndim != 3 or masks_array.shape[0] != scores_array.shape[0]:
            raise RuntimeError(
                "unexpected SAM3 candidate shapes: "
                f"masks={masks_array.shape}, scores={scores_array.shape}"
            )

        index = int(np.argmax(scores_array))
        score = float(scores_array[index])
        box: list[float] | None = None
        if boxes is not None:
            boxes_array = (
                boxes
                if isinstance(boxes, np.ndarray)
                else boxes.detach().float().cpu().numpy()
            )
            if boxes_array.ndim >= 2 and index < boxes_array.shape[0]:
                box = [float(value) for value in boxes_array[index].reshape(-1)[:4]]
        if score < min_score:
            return Sam3Result(
                found=False,
                score=score,
                box=box,
                reason=f"top score {score:.3f} is below min_score {min_score:.3f}",
            )

        mask = np.asarray(masks_array[index]) > 0
        if mask.ndim != 2 or not mask.any():
            return Sam3Result(
                found=False,
                score=score,
                box=box,
                reason="SAM3 returned an empty mask",
            )
        return Sam3Result(
            found=True,
            score=score,
            box=box,
            mask_png_base64=_encode_mask_png(mask),
            mask_shape=[int(mask.shape[0]), int(mask.shape[1])],
        )

    def segment(
        self,
        image_base64: str,
        *,
        text_prompt: str | None = None,
        point: list[int] | None = None,
        min_score: float = 0.2,
    ) -> dict[str, Any]:
        has_text = isinstance(text_prompt, str) and bool(text_prompt.strip())
        has_point = point is not None
        if has_text == has_point:
            raise ValueError("provide exactly one of text_prompt or point")
        if has_text:
            text_prompt = text_prompt.strip()

        image_bytes = base64.b64decode(image_base64, validate=True)
        if not image_bytes:
            raise ValueError("image_base64 is empty")
        response = self._segment_bytes(
            image_bytes,
            text_prompt=text_prompt,
            point=point,
            min_score=min_score,
        )
        return response.to_dict()


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RPent local SAM 3.0 server")
    parser.add_argument("--transport", choices=["socket", "http"], default="http")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8114)
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=None,
        help="GPU device exposed through CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument("--parent-watch", action="store_true",
                        help="watch parent process via stdin pipe and exit when it dies")
    return parser


def main() -> None:
    """Load SAM3 and serve until terminated."""
    args = _build_argparser().parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    if args.cuda_device is not None:
        target = str(args.cuda_device)
        prev = os.environ.get("CUDA_VISIBLE_DEVICES")
        if prev is not None and prev != target:
            logging.warning(
                "CUDA_VISIBLE_DEVICES=%s is already set; overriding with --cuda-device=%s",
                prev, args.cuda_device,
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = target
    checkpoint = os.environ.get("SAM3_CHECKPOINT_PATH")
    if not checkpoint:
        raise RuntimeError(
            "SAM3_CHECKPOINT_PATH is not set; export the path to sam3.pt "
            "before starting RPent"
        )
    facade = Sam3Facade(checkpoint)
    facade.serve(
        transport=args.transport,
        host=args.host,
        port=args.port,
        parent_watch=args.parent_watch,
    )


if __name__ == "__main__":
    main()
