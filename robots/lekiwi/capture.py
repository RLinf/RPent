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

"""Keyboard driving and stereo PNG capture through an existing LeKiwi Env."""

from __future__ import annotations

import argparse
import logging
import signal
import time
from collections.abc import Callable, Collection
from contextlib import ExitStack
from typing import Any
from uuid import uuid4

import numpy as np

from robots.lekiwi.env_client import LeKiwiEnvClient
from rpent.utils.logging import get_logger
from rpent.utils.rpc import make_rpc_client

logger = get_logger("lekiwi_capture")


def keyboard_velocity(keys: Collection[str], limits: np.ndarray) -> np.ndarray:
    """Map LeRobot's held keys to body m/s, m/s and deg/s at the Env limits."""
    return limits * np.array(
        [
            ("w" in keys) - ("s" in keys),
            ("a" in keys) - ("d" in keys),
            ("z" in keys) - ("x" in keys),
        ],
        dtype=np.float32,
    )


def capture(
    env: LeKiwiEnvClient,
    keyboard: Any,
    *,
    preview: Callable[[dict], None] | None = None,
) -> None:
    """Drive while recording; stop the base before closing this recording.

    The caller owns the network client and keyboard. The Env owns the camera,
    serial bus and PNG files. No observations or actions are saved on the client.
    """
    if env.meta["control_mode"] != "manual":
        raise RuntimeError("Start the Env in capture mode before collecting images")
    limits = np.asarray(env.meta["velocity_limits"], dtype=np.float32)
    if limits.shape != (3,) or not np.isfinite(limits).all() or (limits <= 0).any():
        raise ValueError("Env must provide three positive finite velocity limits")
    if not keyboard.is_connected:
        raise RuntimeError(
            "Keyboard unavailable: use a local desktop; on macOS grant terminal "
            "Accessibility/Input Monitoring permission; on Linux use X11"
        )

    identity = uuid4().hex
    with ExitStack() as cleanup:
        # Register before start: a lost response can still leave a recorder running.
        cleanup.callback(_stop_recording, env, identity)
        cleanup.callback(env.stop)
        env.stop()
        state = env.recording("start", identity=identity)
        logger.info(
            "W/S forward/back, A/D strafe, Z/X turn; release to stop. "
            "Q or Esc ends capture. Velocity limits: %s",
            limits.tolist(),
        )
        next_status = 0.0
        moving = False
        previous_tick = time.monotonic()
        while keyboard.is_connected:
            started = time.monotonic()
            if started - previous_tick >= 1.0:
                raise TimeoutError("Capture control loop stalled; restart to resume")
            previous_tick = started
            if started >= next_status:
                state = env.recording("state", identity=identity)
                logger.info(
                    "Capture %s: %s pairs; robot directory: %s",
                    state["phase"],
                    state["frames"],
                    state.get("path", "pending"),
                )
                next_status = started + 1.0
            if state["phase"] == "error":
                raise RuntimeError(state.get("error", "Stereo recording failed"))
            if state["phase"] == "recorded":
                break
            if preview is not None:
                observation = env.get_observation()
                if observation.get("main_images") is None:
                    raise RuntimeError("Camera preview is unavailable; capture stopped")
                preview({"front": observation["main_images"]})

            # Read keys after network/preview work, never refresh a cached command.
            if not keyboard.is_connected:
                break
            keys = {
                key.lower() for key in keyboard.get_action() if isinstance(key, str)
            }
            if "q" in keys:
                break
            if not keyboard.is_connected:
                break
            if time.monotonic() - started >= 1.0:
                raise TimeoutError("Capture control loop stalled; restart to resume")
            velocity = keyboard_velocity(keys, limits)
            if state["phase"] == "recording" and velocity.any():
                env.hold(velocity, ttl_s=1.0)
                moving = True
            elif moving:
                env.stop()
                moving = False
            if time.monotonic() - started >= 1.0:
                raise TimeoutError("Capture control loop stalled; restart to resume")
            time.sleep(max(0.0, 0.1 - (time.monotonic() - started)))


def _stop_recording(env: LeKiwiEnvClient, identity: str) -> None:
    try:
        state = env.recording("stop", identity=identity)
    except Exception:
        logger.exception("Could not stop recording %s; check the Env", identity)
        raise
    logger.info(
        "Capture %s: %s pairs saved on robot at %s",
        state["phase"],
        state["frames"],
        state.get("path", "unknown"),
    )
    if state["phase"] == "error":
        raise RuntimeError(state.get("error", "Stereo recording failed"))


def main(argv: list[str] | None = None) -> None:
    """Connect the operator keyboard and optional preview to a capture Env."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-endpoint", required=True, help="Existing LeKiwi Env URL")
    parser.add_argument(
        "--no-preview", action="store_true", help="Drive with direct sight of the robot"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig

    preview = None
    if not args.no_preview:
        from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

        init_rerun(session_name="lekiwi_capture")
        preview = log_rerun_data
    keyboard = KeyboardTeleop(KeyboardTeleopConfig())

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    previous_handler = signal.signal(signal.SIGTERM, interrupt)
    try:
        with ExitStack() as cleanup:
            keyboard.connect()
            cleanup.callback(
                lambda: keyboard.disconnect() if keyboard.is_connected else None
            )
            rpc = make_rpc_client(args.env_endpoint)
            cleanup.callback(rpc.close)
            env = LeKiwiEnvClient(rpc)
            capture(env, keyboard, preview=preview)
    except KeyboardInterrupt:
        logger.info("Capture interrupted")
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


if __name__ == "__main__":
    main()
