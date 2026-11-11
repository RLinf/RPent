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

"""Record or replay exact visual requests without simulator or model services."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
from pathlib import Path
from typing import Any

import imageio.v3 as imageio
import numpy as np

from rpent.robots.components.molmo_client import MolmoResult
from rpent.robots.components.sam3_client import Sam3Result
from toolkits.geometry.evaluation.checklist import build_manifest
from toolkits.geometry.evaluation.observation import file_digest
from toolkits.geometry.perception import Segmenter, VisualGrounder


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read individual requests, retaining failures as explicit records."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    """Persist a completed request before proceeding to the next one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def image_digest(image: bytes) -> str:
    """Bind replay to decoded original RGB pixels rather than PNG encoding."""
    return hashlib.sha256(imageio.imread(io.BytesIO(image)).tobytes()).hexdigest()


def _key(row: dict[str, Any], *, segment: bool) -> str:
    names = (
        ("rgb_pixel_sha256", "text_prompt", "point", "min_score")
        if segment
        else ("rgb_pixel_sha256", "query")
    )
    return json.dumps({name: row.get(name) for name in names}, sort_keys=True)


def _index(path: Path, *, segment: bool) -> dict[str, dict[str, Any]]:
    response_fields = (
        ("error", "found", "score", "reason", "mask_sha256")
        if segment
        else ("error", "found", "point_xy", "image_size")
    )
    indexed = {}
    for row in read_jsonl(path):
        key = _key(row, segment=segment)
        if key in indexed and any(
            row.get(name) != indexed[key].get(name) for name in response_fields
        ):
            raise ValueError("ambiguous frozen outputs for the same visual request")
        indexed.setdefault(key, row)
    return indexed


class RecordingGrounder:
    """Save successful and failed visual calls for later geometry-only replay."""

    def __init__(self, grounder: VisualGrounder, directory: Path) -> None:
        self.grounder, self.directory = grounder, directory

    def ground(self, image: bytes, query: str) -> MolmoResult:
        """Forward RGB and query, recording their exact identity and output."""
        row = {"rgb_pixel_sha256": image_digest(image), "query": query}
        try:
            result = self.grounder.ground(image, query)
        except Exception as exc:
            append_jsonl(self.directory / "grounding.jsonl", {**row, "error": str(exc)})
            raise
        append_jsonl(
            self.directory / "grounding.jsonl",
            {
                **row,
                "found": result.found,
                "point_xy": result.point_xy,
                "image_size": result.image_size,
                "answer": result.answer,
            },
        )
        return result


class ReplayGrounder:
    """Look up grounding only when both the original RGB and query match."""

    def __init__(self, directory: Path) -> None:
        self.records = _index(directory / "grounding.jsonl", segment=False)

    def ground(self, image: bytes, query: str) -> MolmoResult:
        """Return a frozen selection; an unseen request fails without inference."""
        key = _key(
            {"rgb_pixel_sha256": image_digest(image), "query": query}, segment=False
        )
        if key not in self.records:
            raise ValueError(
                "grounding replay RGB or part prompt differs from the cache"
            )
        row = self.records[key]
        if row.get("error"):
            raise RuntimeError("recorded grounding failure: " + row["error"])
        return MolmoResult(
            found=row["found"],
            point_xy=None if row.get("point_xy") is None else tuple(row["point_xy"]),
            image_size=tuple(row["image_size"]),
            answer=row.get("answer"),
        )


class RecordingSegmenter:
    """Record exact text/point segmentation calls and separately hashed masks."""

    def __init__(self, segmenter: Segmenter, directory: Path) -> None:
        self.segmenter, self.directory = segmenter, directory
        self.index = 0

    def segment(
        self,
        image: bytes,
        *,
        text_prompt: str | None = None,
        point: list[int] | None = None,
        min_score: float = 0.2,
    ) -> Sam3Result:
        """Forward one request and preserve its mask without changing thresholds."""
        row = {
            "rgb_pixel_sha256": image_digest(image),
            "text_prompt": text_prompt,
            "point": point,
            "min_score": min_score,
        }
        try:
            result = self.segmenter.segment(
                image, text_prompt=text_prompt, point=point, min_score=min_score
            )
        except Exception as exc:
            append_jsonl(self.directory / "segments.jsonl", {**row, "error": str(exc)})
            raise
        row.update(found=result.found, score=result.score, reason=result.reason)
        if result.found and result.mask is not None:
            path = self.directory / f"mask_{self.index:04d}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, array=result.mask)
            row.update(mask_path=path.name, mask_sha256=file_digest(path))
        self.index += 1
        append_jsonl(self.directory / "segments.jsonl", row)
        return result


class ReplaySegmenter:
    """Read a mask only for the same RGB, prompt/point and score threshold."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        path = directory / "segments.jsonl"
        self.records = _index(path, segment=True) if path.exists() else {}

    def segment(
        self,
        image: bytes,
        *,
        text_prompt: str | None = None,
        point: list[int] | None = None,
        min_score: float = 0.2,
    ) -> Sam3Result:
        """Verify the request and mask hash; never call SAM3 on a cache miss."""
        key = _key(
            {
                "rgb_pixel_sha256": image_digest(image),
                "text_prompt": text_prompt,
                "point": point,
                "min_score": min_score,
            },
            segment=True,
        )
        if key not in self.records:
            raise ValueError(
                "segmentation replay RGB, prompt, point or threshold differs"
            )
        row = self.records[key]
        if row.get("error"):
            raise RuntimeError("recorded segmentation failure: " + row["error"])
        mask = None
        if row["found"]:
            name = row["mask_path"]
            if Path(name).name != name:
                raise ValueError("a replay mask must be inside its cache directory")
            path = self.directory / name
            if file_digest(path) != row["mask_sha256"]:
                raise ValueError("frozen segmentation mask changed")
            with np.load(path, allow_pickle=False) as archive:
                mask = archive["array"]
        return Sam3Result(
            found=row["found"],
            score=row.get("score"),
            reason=row.get("reason"),
            mask=mask,
            mask_shape=None if mask is None else mask.shape,
        )


def freeze_legacy(predictions: Path, grounding: Path, output: Path) -> None:
    """Import original frozen selections/masks without copying any reference data."""
    groundings = {row["request_id"]: row for row in read_jsonl(grounding)}
    manifest = build_manifest()
    expected = {
        request["request_id"]: (case, request)
        for case in manifest["cases"]
        for request in case["requests"]
    }
    index = 0
    seen = set()
    for prediction in read_jsonl(predictions / "results.jsonl"):
        request_id = prediction["request_id"]
        if request_id not in expected or request_id in seen:
            raise ValueError("unknown or duplicate legacy request")
        seen.add(request_id)
        case, request = expected[request_id]
        ground = groundings[request_id]
        append_jsonl(output / "grounding.jsonl", ground)
        root = predictions / case["case_id"]
        data = json.loads((root / f"result_{request['slot']}.json").read_text())
        for part in data["parts"].values():
            for attempt in part["segment_attempts"]:
                name = attempt["segment_artifact"]
                if Path(name).name != name:
                    raise ValueError("invalid legacy segment artifact")
                record = json.loads((root / "episode" / name / "00.json").read_text())
                row = {
                    "rgb_pixel_sha256": ground["rgb_pixel_sha256"],
                    "text_prompt": record.get("prompt"),
                    "point": record.get("point"),
                    "min_score": record["min_score"],
                    "found": record["found"],
                    "score": record.get("score"),
                    "reason": record.get("reason"),
                }
                if record["found"]:
                    mask_name = record["mask_artifact"]
                    if Path(mask_name).name != mask_name:
                        raise ValueError("invalid legacy mask artifact")
                    source = root / "episode" / mask_name / "00.npz"
                    target = output / f"mask_{index:04d}.npz"
                    shutil.copyfile(source, target)
                    row.update(mask_path=target.name, mask_sha256=file_digest(target))
                    index += 1
                append_jsonl(output / "segments.jsonl", row)
