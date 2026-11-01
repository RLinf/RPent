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

"""Build workspace maps from a recorded video.

A mapping session holds a bounded number of frames, so a long walkthrough is
built in segments: the video is sampled at a fixed rate, each segment opens its
own session, and every segment becomes its own map directory. Segments are
reconstructed independently and are never merged, so a recording longer than
the frame limit yields several maps instead of requiring a larger GPU.

Run it against a running mapping service::

    python -m rpent.tools.build_map \\
        --video workspace_walkthrough.mp4 \\
        --map-name workspace \\
        --endpoint http://127.0.0.1:8116

Add ``--labels`` to ground objects on every map through the perception
service; see :mod:`rpent.tools.ground_map`.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from rpent.tools.ground_map import (
    ClientMapSource,
    ground_map,
    save_grounded_map_via_client,
)
from rpent.utils.logging import get_logger
from rpent.utils.rpc.http_rpc import HttpRpcClient
from toolkits.grounding.client import GdinoSam2Client
from toolkits.mapping.client import LingBotMapClient, MapBuildResult
from toolkits.mapping.server import DEFAULT_MAX_SESSION_FRAMES

logger = get_logger("build_map")

#: Sampled frames per second; a slow walkthrough changes little faster.
DEFAULT_VIDEO_FPS = 5.0
#: Assumed source rate when the container does not report one.
DEFAULT_SOURCE_FPS = 30.0
DEFAULT_MAPPING_ENDPOINT = "http://127.0.0.1:8116"
DEFAULT_PERCEPTION_ENDPOINT = "http://127.0.0.1:8115"


def _open_video(video_path: str | Path) -> tuple[Any, float]:
    """Open a video and return its capture handle and source frame rate."""
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {video_path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(source_fps) or source_fps <= 0:
        source_fps = DEFAULT_SOURCE_FPS
    return capture, source_fps


def iter_video_frames(
    video_path: str | Path, *, fps: float = DEFAULT_VIDEO_FPS
) -> Iterable[tuple[np.ndarray, float]]:
    """Yield ``(rgb, timestamp_s)`` samples decoded from a video.

    Frames are yielded as RGB ``[H,W,3]`` arrays at most ``fps`` times per
    second of source video; timestamps are relative to the start of the video.
    Reading is streamed, so the whole recording is never held in memory.
    """
    capture, source_fps = _open_video(video_path)
    every = max(1, round(source_fps / fps))
    try:
        index = 0
        sampled = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % every == 0:
                yield _to_rgb(frame), sampled / fps
                sampled += 1
            index += 1
    finally:
        capture.release()


def _to_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert one OpenCV BGR frame to RGB without decoding twice."""
    import cv2

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def estimated_segments(
    video_path: str | Path,
    *,
    fps: float = DEFAULT_VIDEO_FPS,
    segment_frames: int = DEFAULT_MAX_SESSION_FRAMES,
) -> int | None:
    """Estimate how many segments a video needs, or ``None`` when unknown.

    Containers that report neither a frame count nor a duration return
    ``None``; callers then name every segment explicitly rather than assuming
    the recording fits in one map.
    """
    import cv2

    capture, source_fps = _open_video(video_path)
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if frame_count <= 0:
        return None
    every = max(1, round(source_fps / fps))
    sampled = math.ceil(frame_count / every)
    return max(1, math.ceil(sampled / segment_frames))


def segment_names(map_name: str, segments: int | None) -> list[str]:
    """Return the map name of each segment, suffixing when there are several.

    A recording that fits in one segment keeps the requested name. A longer
    recording numbers every segment, including the first, so the set of maps is
    predictable from the command line.
    """
    if segments == 1:
        return [map_name]
    count = 1 if segments is None else segments
    # ``None`` still needs at least one name; the caller may build fewer maps
    # than estimated when the container overstates its frame count.
    return [f"{map_name}_{index:02d}" for index in range(count)]


def build_maps(
    frames: Iterable[tuple[np.ndarray, float]],
    map_client: LingBotMapClient,
    *,
    map_name: str,
    segment_frames: int = DEFAULT_MAX_SESSION_FRAMES,
    segments: int | None = None,
    labels: Sequence[str] | None = None,
    perception_client: GdinoSam2Client | None = None,
) -> list[dict[str, Any]]:
    """Push sampled frames into one session per segment and build each map.

    Args:
        frames: Sampled ``(rgb, timestamp_s)`` pairs, in recording order.
        map_client: Client for the mapping service.
        map_name: Map name, or the prefix when the recording needs several
            segments.
        segment_frames: Maximum frames per session, matching the service's
            ``--max-frames``.
        segments: Number of maps the recording will produce, when known. Used
            only to decide between the plain name and numbered names.
        labels: Optional object labels to ground on every built map.
        perception_client: Grounding DINO + SAM 2 client; required with labels.

    Returns:
        One record per built map with its name, path, frame count, and any
        grounding or annotation error.
    """
    if segment_frames < 1:
        raise ValueError("segment_frames must be at least 1")
    if labels and perception_client is None:
        raise ValueError("labels require a perception client")

    names = segment_names(map_name, segments)
    iterator = iter(frames)
    # Hold one frame back so a recording that ends on a segment boundary does
    # not open one last session with nothing to push.
    pending = next(iterator, None)
    results: list[dict[str, Any]] = []
    while pending is not None:
        name = names[len(results)] if len(results) < len(names) else None
        if name is None:
            # The container understated its length; keep numbering.
            name = f"{map_name}_{len(results):02d}"
        map_client.start_session(map_name=name)
        pushed = 0
        try:
            while pending is not None and pushed < segment_frames:
                rgb, timestamp_s = pending
                map_client.push_frame(rgb, pose=None, timestamp_s=timestamp_s)
                pushed += 1
                pending = next(iterator, None)
            build = map_client.build()
        except BaseException:
            _stop_session(map_client)
            raise
        results.append(_record(build, pushed))
        if labels:
            _annotate(results[-1], build, map_client, labels, perception_client)
    if not results:
        raise ValueError("the video produced no sampled frames")
    return results


def _stop_session(map_client: LingBotMapClient) -> None:
    """Stop a session after a failed push without masking the original error."""
    try:
        map_client.stop_session()
    except Exception as exc:
        logger.warning("failed to stop mapping session: %s", exc)


def _record(build: MapBuildResult, pushed: int) -> dict[str, Any]:
    """Return the summary of one built map."""
    return {
        "map_name": Path(build.map_path).name,
        "map_path": build.map_path,
        "frames": build.frames,
        "frames_pushed": pushed,
        "backend": build.metadata.get("backend"),
        "has_depth": bool(build.metadata.get("has_depth")),
    }


def _annotate(
    record: dict[str, Any],
    build: MapBuildResult,
    map_client: LingBotMapClient,
    labels: Sequence[str],
    perception_client: GdinoSam2Client,
) -> None:
    """Ground labels on a built map and record what was written."""
    try:
        source = ClientMapSource(map_client, build.map_path, build.metadata)
        result = ground_map(source, perception_client, labels, allow_2d=True)
        save_grounded_map_via_client(result, map_client)
    except Exception as exc:
        record["annotation_error"] = str(exc)
        return
    record["labels"] = {
        label: sum(1 for obj in result.objects if obj.label == label)
        for label in labels
    }


def build_maps_from_video(
    video_path: str | Path,
    map_client: LingBotMapClient,
    *,
    map_name: str,
    fps: float = DEFAULT_VIDEO_FPS,
    segment_frames: int = DEFAULT_MAX_SESSION_FRAMES,
    labels: Sequence[str] | None = None,
    perception_client: GdinoSam2Client | None = None,
) -> list[dict[str, Any]]:
    """Build one map per segment of a recorded video.

    Args:
        video_path: Recording to sample; frames are read at ``fps``.
        map_client: Client for the mapping service.
        map_name: Map name, or the prefix when the video needs several maps.
        fps: Sampled frames per second of source video.
        segment_frames: Maximum frames per session.
        labels: Optional object labels to ground on every built map.
        perception_client: Grounding DINO + SAM 2 client; required with labels.

    Returns:
        One record per built map, in recording order.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    segments = estimated_segments(video_path, fps=fps, segment_frames=segment_frames)
    return build_maps(
        iter_video_frames(video_path, fps=fps),
        map_client,
        map_name=map_name,
        segment_frames=segment_frames,
        segments=segments,
        labels=labels,
        perception_client=perception_client,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Build and optionally ground maps from a recorded video."""
    parser = argparse.ArgumentParser(
        description="Build workspace maps from a recorded video."
    )
    parser.add_argument("--video", required=True, help="input video file")
    parser.add_argument(
        "--map-name",
        required=True,
        help="map name, or the prefix when the video needs several maps",
    )
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_MAPPING_ENDPOINT,
        help=f"mapping service endpoint (default: {DEFAULT_MAPPING_ENDPOINT})",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=DEFAULT_VIDEO_FPS,
        help=f"sampled frames per second (default: {DEFAULT_VIDEO_FPS})",
    )
    parser.add_argument(
        "--segment-frames",
        type=int,
        default=DEFAULT_MAX_SESSION_FRAMES,
        help=(
            "frames per mapping session (default: "
            f"{DEFAULT_MAX_SESSION_FRAMES}); match the server's --max-frames"
        ),
    )
    parser.add_argument("--labels", nargs="+", help="object prompts to ground")
    parser.add_argument(
        "--perception-endpoint",
        default=DEFAULT_PERCEPTION_ENDPOINT,
        help=(
            "Grounding DINO + SAM 2 endpoint used with --labels "
            f"(default: {DEFAULT_PERCEPTION_ENDPOINT})"
        ),
    )
    parser.add_argument("--output", help="write the JSON summary to this file")
    args = parser.parse_args(argv)

    labels = [str(label).strip() for label in args.labels or [] if str(label).strip()]
    if args.labels and len(labels) != len(args.labels):
        raise SystemExit("--labels must be non-empty strings")
    perception = (
        GdinoSam2Client(HttpRpcClient(args.perception_endpoint), timeout_s=300.0)
        if labels
        else None
    )
    try:
        maps = build_maps_from_video(
            args.video,
            LingBotMapClient(HttpRpcClient(args.endpoint)),
            map_name=args.map_name,
            fps=args.fps,
            segment_frames=args.segment_frames,
            labels=labels or None,
            perception_client=perception,
        )
    except Exception as exc:
        logger.error("map construction failed: %s", exc)
        raise SystemExit(1) from exc

    summary = {
        "video": str(args.video),
        "map_name": args.map_name,
        "fps": args.fps,
        "segment_frames": args.segment_frames,
        "maps": maps,
    }
    text = json.dumps(summary, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n")
    print(text)
    errors = [m["annotation_error"] for m in maps if m.get("annotation_error")]
    for error in errors:
        logger.warning("grounding failed: %s", error)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
