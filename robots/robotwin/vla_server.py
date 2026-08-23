"""Launcher for the official LingBot-VLA WebSocket server."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

# Support direct execution from an RPent checkout before package imports.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from robots.robotwin.env_spec import MODEL_SPEC, vla_runtime_contract
from rpent.tools.vla_facade_base import BaseVLAFacade
from rpent.utils.daemon import watch_parent_death


def _on_parent_death() -> None:
    """Print a marker and exit when the parent process closes stdin."""
    print("parent_watch_triggered=true", flush=True)
    os._exit(0)


class LingBotVLAFacade(BaseVLAFacade):
    """Adapt LingBot's native policy to the common RPent VLA facade."""

    def __init__(self, policy: Any):
        self._policy = policy
        super().__init__()

    def predict(self, obs, options=None):
        del options
        return self._policy.infer(obs)

    def infer(self, obs):
        """Compatibility entry point required by ``WebsocketPolicyServer``."""
        return self._dispatch("vla.predict", (obs, None), {})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Start the official LingBot-VLA WebSocket policy server"
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--norm-path", required=True)
    parser.add_argument("--use-length", type=int, default=MODEL_SPEC.use_length)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--num-denoising-step", type=int, default=10)
    parser.add_argument("--use-compile", action="store_true")
    parser.add_argument(
        "--parent-watch",
        action="store_true",
        help="Exit this launcher when the parent process closes stdin.",
    )
    parser.add_argument(
        "--lingbot-robot-config",
        type=Path,
        required=True,
        help="Path to the LingBot FeatureTransform robot config YAML.",
    )
    args = parser.parse_args()

    robot_config = args.lingbot_robot_config.expanduser().resolve()
    if not robot_config.is_file():
        raise FileNotFoundError(f"LingBot robot config not found: {robot_config}")

    if args.parent_watch:
        # The upstream server has no shutdown API. Ending this process when its
        # parent exits releases the socket and CUDA context.
        watch_parent_death(_on_parent_death)

    from deploy.lingbot_vla_policy import LingbotVLAServer
    from deploy.websocket_policy_server import WebsocketPolicyServer

    native_policy = LingbotVLAServer(
        args.model_path,
        use_length=args.use_length,
        robot_norm_path=args.norm_path,
        num_denoising_step=args.num_denoising_step,
        use_compile=args.use_compile,
        robot_config=robot_config,
    )
    policy = LingBotVLAFacade(native_policy)
    WebsocketPolicyServer(
        policy,
        port=args.port,
        metadata=vla_runtime_contract(),
    ).serve_forever()


if __name__ == "__main__":
    main()
