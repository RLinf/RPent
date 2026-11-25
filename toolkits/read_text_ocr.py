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

"""Read text in recorded camera images with an optional local OCR engine."""

import csv
import io
import shutil
import subprocess
from collections.abc import Callable
from typing import Annotated, Any, Literal

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from rpent.tools import ToolResult, tool
from toolkits.observation import ImageSelection, Index, ObservationProvider, Size

Layout = Literal["block", "line", "sparse"]
_PAGE_SEGMENTATION = {"block": "6", "line": "7", "sparse": "11"}
_MAX_EDGE = 2048
_MAX_WORDS = 64
_MAX_TEXT_CHARS = 1500
_LINE_FIELDS = ("page_num", "block_num", "par_num", "line_num")


class TextRegion(BaseModel):
    """A rectangle specified on a full-frame camera image or its preview."""

    model_config = ConfigDict(extra="forbid")
    image_shape: Annotated[list[Size], Field(min_length=2, max_length=2)]
    box_rc: Annotated[list[Index], Field(min_length=4, max_length=4)]


@tool(readonly=True, exclude=("observations", "recognize"))
def read_text_ocr(
    camera: Annotated[str, Field(min_length=1, max_length=128)],
    step: Annotated[int, Field(ge=-1)] = -1,
    region: TextRegion | None = None,
    layout: Layout = "block",
    upscale: Annotated[int, Field(strict=True, ge=1, le=4)] = 2,
    min_confidence: Annotated[FiniteFloat, Field(ge=0, le=1)] = 0.5,
    *,
    observations: ObservationProvider,
    recognize: Callable[[bytes, Layout], str] | None = None,
) -> ToolResult:
    """Read printed English text and digits in a recorded RGB image.

    Returns text, per-word confidence on a 0–1 scale, and half-open word boxes
    [top,left,bottom,right] in the native full-frame image. Inspect the image
    if the transcription is incomplete or ambiguous.

    Args:
        camera: Name of the recorded camera.
        step: Recorded step, or -1 for the latest record.
        region: Optional box_rc on a full-frame image of the given image_shape.
        layout: Block for a compact label, line for a single row of text,
            or sparse for scattered labels.
        upscale: Integer image enlargement, 2 by default. Keep each processed
            edge at most 2048 pixels by selecting a smaller region if needed.
        min_confidence: Minimum confidence for a returned word, 0.5 by default.
        observations: Provider for recorded camera observations.
        recognize: Optional backend that returns Tesseract-compatible TSV.
    """
    try:
        view = observations.get_observation(camera=camera, step=step)
        if view.camera != camera or (step >= 0 and view.step != step):
            raise ValueError("Provider returned a different camera or step.")

        height, width = view.rgb.shape[:2]
        box = (0, 0, height, width)
        if region is not None:
            crop = TextRegion.model_validate(region)
            selection = ImageSelection(image_shape=crop.image_shape, box_rc=crop.box_rc)
            _, box = selection.native(view)
        top, left, bottom, right = box
        size = ((right - left) * upscale, (bottom - top) * upscale)
        if max(size) > _MAX_EDGE:
            raise ValueError(
                f"Processed image exceeds {_MAX_EDGE} pixels on an edge. "
                "Select a smaller region or lower upscale."
            )
        image = Image.fromarray(view.rgb[top:bottom, left:right])
        if upscale != 1:
            image = image.resize(size, Image.Resampling.BICUBIC)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        tsv = (recognize or _tesseract)(buffer.getvalue(), layout)
        text, words, truncated = _read_words(tsv, box, upscale, min_confidence)
    except (ValueError, LookupError, OSError, RuntimeError, csv.Error) as exc:
        return ToolResult(data={"status": "error", "error": str(exc)})

    return ToolResult(
        data={
            "status": "read" if words else "no_text",
            "camera": view.camera,
            "step": view.step,
            "image_shape": [height, width],
            "source_box_rc": list(box),
            "text": text,
            "words": words,
            "truncated": truncated,
        }
    )


def _tesseract(png: bytes, layout: Layout) -> str:
    executable = shutil.which("tesseract")
    if executable is None:
        raise RuntimeError(
            "Install Tesseract with English language data and add it to PATH."
        )
    try:
        completed = subprocess.run(
            [
                executable,
                "stdin",
                "stdout",
                "-l",
                "eng",
                "--psm",
                _PAGE_SEGMENTATION[layout],
                "tsv",
            ],
            input=png,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Tesseract timed out. Try a smaller image region.") from exc
    if completed.returncode:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()[:400]
        raise RuntimeError(
            f"Tesseract failed with exit code {completed.returncode}. {detail}".strip()
        )
    return completed.stdout.decode("utf-8")


def _read_words(
    tsv: str, source_box: tuple[int, int, int, int], scale: int, minimum: float
) -> tuple[str, list[dict[str, Any]], bool]:
    records = csv.DictReader(io.StringIO(tsv), delimiter="\t", quoting=csv.QUOTE_NONE)
    fields = {"level", "left", "top", "width", "height", "conf", "text", *_LINE_FIELDS}
    if not fields.issubset(records.fieldnames or ()):
        raise ValueError("Expected word confidence and bounding box fields in OCR TSV.")

    words: list[dict[str, Any]] = []
    lines: list[list[str]] = []
    current_line = None
    chars = 0
    truncated = False
    for row in records:
        if row["level"] != "5":
            continue
        if any(row[field] is None for field in fields):
            raise ValueError("Expected a complete word record in OCR TSV.")
        if not (word := row["text"].strip()):
            continue
        try:
            confidence = float(row["conf"]) / 100
            y, x, height, width = (
                int(row[key]) for key in ("top", "left", "height", "width")
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Expected numeric confidence and bounds for an OCR word."
            ) from exc
        if not minimum <= confidence <= 1 or height <= 0 or width <= 0:
            continue
        top, left, bottom, right = source_box
        y0, x0 = top + max(0, y // scale), left + max(0, x // scale)
        y1 = min(bottom, top + (y + height + scale - 1) // scale)
        x1 = min(right, left + (x + width + scale - 1) // scale)
        if y1 <= y0 or x1 <= x0:
            continue

        chars += len(word) + bool(words)
        if len(words) == _MAX_WORDS or chars > _MAX_TEXT_CHARS:
            truncated = True
            break
        line = tuple(row[key] for key in _LINE_FIELDS)
        if line != current_line:
            lines.append([])
            current_line = line
        lines[-1].append(word)
        words.append(
            {
                "text": word,
                "confidence": confidence,
                "box_rc": [y0, x0, y1, x1],
            }
        )
    return "\n".join(" ".join(line) for line in lines), words, truncated
