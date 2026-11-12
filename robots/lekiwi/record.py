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

"""Record LeKiwi video without starting or stopping motion."""

from __future__ import annotations

import argparse
import json
import logging
import math
import signal
import tempfile
import time
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from robots.lekiwi.env_client import LeKiwiEnvClient
from rpent.utils.logging import get_logger
from rpent.utils.rpc import make_rpc_client

logger = get_logger("lekiwi_record")


def record(
    env: LeKiwiEnvClient,
    directory: Path,
    *,
    fps: float = 10.0,
    duration_s: float = 600.0,
    stop_requested: Callable[[], bool] | None = None,
) -> None:
    """Stream video synchronously; finish the current frame before a requested stop."""
    if not math.isfinite(fps) or not 0 < fps <= 60:
        raise ValueError("fps must be between 0 (exclusive) and 60")
    if not math.isfinite(duration_s) or not 0 < duration_s <= 3600:
        raise ValueError("duration_s must be between 0 (exclusive) and 3600")
    started_at = time.monotonic()
    deadline = started_at + duration_s
    next_report = started_at
    next_frame = started_at
    frames = 0
    try:
        with ExitStack() as cleanup:
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
            directory = Path(tempfile.mkdtemp(prefix="recording-", dir=directory))
            logger.info("Recording starting; output directory: %s", directory)
            (directory / "camera_meta.json").write_text(
                json.dumps(env.get_camera_meta(), indent=2)
            )
            import imageio.v2 as imageio

            writer = None
            timestamps = cleanup.enter_context(
                (directory / "frames.jsonl").open("w", encoding="utf-8")
            )
            while (now := time.monotonic()) < deadline:
                if stop_requested is not None and stop_requested():
                    break
                if now >= next_report:
                    logger.info(
                        "Recording: %s frames; output directory: %s", frames, directory
                    )
                    next_report = now + 1.0
                if now < next_frame:
                    time.sleep(min(0.1, next_frame - now, deadline - now))
                    continue
                frame = env.get_observation().get("main_images")
                if frame is None:
                    raise RuntimeError("Camera frame unavailable; recording ended")
                received_at = time.time()
                elapsed_s = time.monotonic() - started_at
                # RLinf cameras return BGR; imageio encodes RGB.
                image = np.asarray(frame)
                height, width = image.shape[:2]
                if writer is None:
                    # H.264's usual 4:2:0 format requires even dimensions.
                    # Use 4:4:4 for odd frames to preserve the delivered size.
                    writer = imageio.get_writer(
                        directory / "video.mp4",
                        fps=fps,
                        codec="libx264",
                        macro_block_size=1,
                        pixelformat="yuv444p" if width % 2 or height % 2 else "yuv420p",
                    )
                    cleanup.callback(writer.close)
                writer.append_data(image[..., ::-1])
                timestamps.write(
                    json.dumps(
                        {
                            "frame_index": frames,
                            "received_at_unix_s": received_at,
                            "elapsed_s": elapsed_s,
                            "video_time_s": frames / fps,
                            "width": width,
                            "height": height,
                        }
                    )
                    + "\n"
                )
                frames += 1
                next_frame = now + 1 / fps
        logger.info(
            "Recording recorded: %s frames; output directory: %s", frames, directory
        )
    except Exception as exc:
        logger.exception("LeKiwi recording failed")
        raise RuntimeError(str(exc)) from exc


def main(argv: list[str] | None = None) -> None:
    """Start recording immediately; Ctrl-C flushes the video and exits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-endpoint", required=True)
    parser.add_argument(
        "--recording-dir",
        type=Path,
        default=Path("logs/lekiwi_recordings"),
        help="Video output root on this computer",
    )
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument(
        "--duration", type=float, default=600.0, help="Seconds, at most 3600"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    stopping = False

    def interrupt(signum, frame):
        nonlocal stopping
        stopping = True

    with ExitStack() as cleanup:
        # Let the active read/write finish so interruption still leaves a valid MP4.
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous = signal.signal(signum, interrupt)
            cleanup.callback(signal.signal, signum, previous)
        rpc = make_rpc_client(args.env_endpoint)
        cleanup.callback(rpc.close)
        env = LeKiwiEnvClient(rpc, reset_on_connect=False)
        record(
            env,
            args.recording_dir.expanduser().resolve(),
            fps=args.fps,
            duration_s=args.duration,
            stop_requested=lambda: stopping,
        )
        if stopping:
            logger.info("Recording interrupted")


if __name__ == "__main__":
    main()
