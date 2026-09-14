# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Control-machine RPC boundary for the existing YAM runtime."""

from __future__ import annotations

import argparse
import json
import logging
import threading

from robots.yam.contracts import env_runtime_contract
from rpent.robots.components.env_facade_base import BaseEnvFacade


class YamEnvFacade(BaseEnvFacade):
    """Serializes stateful calls; stop requests only signal the active writer.

    Startup/clear-error/engage belong to the on-site operator. No request
    automatically clears a motor fault, re-enables torque, or parks the arms.
    Real RGBD buffers are owned by the env: a snapshot's wrist extrinsics and
    depth must stay attached to its RGB frame instead of a later FK query.
    """

    def __init__(self, env, *, metadata=None):
        self._env = env
        self._metadata = metadata or env_runtime_contract(
            task_name=env.get_task_language(),
        )
        super().__init__()

    def _register_rpc(self):
        super()._register_rpc()
        self._rpc.update(
            {
                "env.observe": self.observe,
                "env.is_started": self.is_started,
                "env.plan_arm_path": self.plan_arm_path,
                "env.request_stop": self.request_stop,
                "env.read_control_state": self.read_control_state,
                "env.control_step": self.control_step,
                "env.reset_to_configured_qpos": self.reset_to_configured_qpos,
                "env.recover_up": self.recover_up,
                "env.release_gripper": self.release_gripper,
            }
        )
        # Camera snapshots and FK share state; serialize them with execution.
        self._readonly_methods.difference_update(
            {
                "env.render_camera",
                "env.get_camera_meta",
            }
        )

    def _dispatch(self, method, args, kwargs):
        # Never wait behind a motion chunk just to signal cancellation. This
        # path sets an Event; all CAN writes, including hold(), stay on writer.
        if method == "env.request_stop":
            return self.request_stop()
        if method == "shutdown":
            # Home must succeed while the RPC server is still alive, so an
            # operator can retry a failed shutdown without losing motor output.
            with self._dispatch_lock.write():
                self.close()
                self._shutdown_event.set()
            return {"ok": True}
        return super()._dispatch(method, args, kwargs)

    def get_env_meta(self):
        return dict(self._metadata)

    def is_started(self):
        return self._env.is_started()

    def observe(self):
        return self._env.observe()

    def read_control_state(self):
        return self._env.read_control_state()

    def control_step(self, action, *, expected_episode_id=None):
        return self._env.control_step(action, expected_episode_id=expected_episode_id)

    def reset(self, *, seed=None, options=None):
        return self._env.reset(seed=seed, options=options)

    def step(self, action, *, action_type="qpos", expected_episode_id=None):
        return self._env.step(
            action,
            action_type=action_type,
            expected_episode_id=expected_episode_id,
        )

    def chunk_step(
        self,
        actions,
        *,
        action_type="qpos",
        return_all_frames=False,
        expected_episode_id=None,
    ):
        return self._env.chunk_step(
            actions,
            action_type=action_type,
            return_all_frames=return_all_frames,
            expected_episode_id=expected_episode_id,
        )

    def render_camera(self, camera_name, *, depth=False):
        return self._env.render_camera(camera_name, depth=depth)

    def get_camera_meta(self, camera_name):
        return self._env.get_camera_meta(camera_name)

    def get_task_language(self):
        return self._env.get_task_language()

    def plan_arm_path(self, arm, target_pose):
        return self._env.plan_arm_path(arm, target_pose)

    def request_stop(self):
        self._env.request_stop()
        return {"stop_requested": True, "hold_confirmed": False}

    def release_gripper(self, arm, *, expected_episode_id):
        return self._env.release_gripper(arm, expected_episode_id=expected_episode_id)

    def recover_up(self, arm, distance_m=0.04, *, expected_episode_id):
        return self._env.recover_up(
            arm, distance_m, expected_episode_id=expected_episode_id
        )

    def reset_to_configured_qpos(self):
        return self._env.reset_to_configured_qpos()

    def close(self):
        self._env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True, help="Local JSON hardware configuration"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8110)
    parser.add_argument("--transport", choices=("http", "socket"), default="http")
    parser.add_argument("--parent-watch", action="store_true")
    args = parser.parse_args()
    from robots.yam.rlinf_env import YamAgentEnv

    with open(args.config) as stream:
        config = json.load(stream)
    if not config.get("park_on_close", {}).get("enabled"):
        parser.error(
            "site config must enable park_on_close with the confirmed home pose"
        )
    env = YamAgentEnv(config)
    metadata = env_runtime_contract(
        task_name=config["task_name"],
        seed=config.get("seed", 0),
        max_episode_steps=config.get("max_episode_steps", 1000),
        joint_servo=config.get("joint_servo"),
    )
    metadata["execution"]["joint_limit_min"] = env.lower.tolist()
    metadata["execution"]["compact_control"] = True
    metadata["execution"]["joint_limit_max"] = env.upper.tolist()
    facade = YamEnvFacade(
        env,
        metadata=metadata,
    )
    try:
        facade.serve(
            host=args.host,
            port=args.port,
            transport=args.transport,
            parent_watch=args.parent_watch,
        )
    except KeyboardInterrupt:
        pass
    finally:
        while True:
            try:
                facade.close()
                break
            except Exception:
                logging.exception("YAM shutdown incomplete; keeping runtime alive")
                try:
                    input("home 回位未完成；检查路径后按回车重试。请勿直接关闭终端：")
                except EOFError:
                    logging.error(
                        "No input; support the arms before manually stopping this process."
                    )
                    threading.Event().wait()


if __name__ == "__main__":
    main()
