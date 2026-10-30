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

"""RPC server that runs on the R2 and owns its RLinf environment."""

from __future__ import annotations

import argparse
from typing import Any

from robots.r2.runtime_config import load_site_config
from rpent.robots.components.env_facade_base import BaseEnvFacade
from rpent.utils.logging import get_logger

logger = get_logger("r2_env_server")


class R2EnvFacade(BaseEnvFacade):
    """Expose the R2 ``env.*`` protocol from one :class:`R2AgentEnv`.

    Every motion holds the facade's write lock, so planner calls never
    overlap on the hardware; observations take the read lock.
    """

    _MOTIONS = (
        "move_to",
        "move_delta",
        "rotate_delta",
        "set_gripper",
        "look",
        "go_home",
        "move_base",
    )

    def __init__(self, env: Any) -> None:
        self._env = env
        super().__init__()

    def _register_rpc(self) -> None:
        super()._register_rpc()
        for name in self._MOTIONS:
            self._rpc[f"env.{name}"] = getattr(self._env, name)
        self._rpc["env.observe"] = self._env.observe
        self._rpc["env.get_robot_state"] = self._env.robot_state
        self._readonly_methods.update({"env.observe", "env.get_robot_state"})

    def get_env_meta(self) -> dict[str, Any]:
        return self._env.get_env_meta()

    def reset(self) -> dict[str, Any]:
        return self._env.reset()

    def get_camera_meta(self) -> dict[str, Any]:
        return self._env.get_camera_meta()

    def get_task_language(self) -> None:
        # The planner owns the task; the robot serves any instruction.
        return None

    def close(self) -> None:
        self._env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robot-config",
        default=None,
        help="R2 site YAML (defaults to robots/r2/config/example.yaml)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address; use the robot's LAN address when the planner runs elsewhere",
    )
    parser.add_argument("--port", type=int, default=8120)
    parser.add_argument(
        "--transport",
        choices=("http", "socket"),
        default="http",
        help="socket unpickles requests, so use it only on a trusted network",
    )
    parser.add_argument("--parent-watch", action="store_true")
    args = parser.parse_args()

    from robots.r2.rlinf_env import R2AgentEnv

    try:
        site = load_site_config(args.robot_config)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    env = R2AgentEnv.from_site(site)
    facade = R2EnvFacade(env)
    logger.info("R2 env server ready: %s", env.get_env_meta())
    try:
        # serve() closes the facade, and with it the robot, when it returns.
        facade.serve(
            transport=args.transport,
            host=args.host,
            port=args.port,
            parent_watch=args.parent_watch,
        )
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
