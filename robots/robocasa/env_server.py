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

"""RoboCasa env server — hosts the raw robosuite env in a subprocess, exposes basic calls via RPC."""

import argparse
import inspect
import os
import queue
import re
import sys
import threading
import traceback
import uuid
from collections.abc import Mapping

import numpy as np

from rpent.robots.components.env_facade_base import BaseEnvFacade
from rpent.utils.daemon import watch_parent_death
from rpent.utils.logging import get_logger
from rpent.utils.rpc.http_rpc import HttpRpcServer
from rpent.utils.rpc.socket_rpc import SocketRpcServer

logger = get_logger("env_server")


DEFAULT_CAMS = [
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
]

ENV_PROTOCOL_VERSION = 1
NAV_CAMERA = "mobilebase0_navview"
RLDX_ACTION_FIELDS = (
    ("action.end_effector_position", 3),
    ("action.end_effector_rotation", 3),
    ("action.gripper_close", 1),
    ("action.base_motion", 4),
    ("action.control_mode", 1),
)
RLDX_ACTION_DIM = sum(size for _, size in RLDX_ACTION_FIELDS)
MAX_RENDER_DIM = 4096
MAX_RENDER_PIXELS = MAX_RENDER_DIM * MAX_RENDER_DIM
MAX_ACTION_CHUNK = 512


def _env_flag(name, default):
    raw = os.environ.get(name, default)
    if raw not in {"0", "1"}:
        raise ValueError(f"{name} must be 0 or 1, got {raw!r}")
    return raw == "1"


def _positive_int(value, name, *, maximum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value <= 0 or value > maximum:
        raise ValueError(f"{name} must be in [1, {maximum}], got {value}")
    return value


def _finite_array(value, name, *, shape):
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.bool_
    ):
        raise TypeError(f"{name} must contain numeric values")
    array = array.astype(np.float64, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _finite_action_part(value, name, size):
    array = np.asarray(value)
    if size == 1 and array.shape == ():
        array = array.reshape(1)
    return _finite_array(array, name, shape=(size,))


def _split_kwargs(split):
    """Replicate robocasa.utils.env_utils.create_env's split -> layout logic."""
    if split == "target":
        return {
            "obj_instance_split": "target",
            "layout_ids": None,
            "style_ids": None,
            "layout_and_style_ids": list(zip(range(1, 11), range(1, 11))),
        }
    if split == "pretrain":
        return {
            "obj_instance_split": "pretrain",
            "layout_ids": -2,
            "style_ids": -2,
            "layout_and_style_ids": None,
        }
    if split == "all":
        return {
            "obj_instance_split": None,
            "layout_ids": -3,
            "style_ids": -3,
            "layout_and_style_ids": None,
        }
    if split is None:
        return {
            "obj_instance_split": None,
            "layout_ids": None,
            "style_ids": None,
            "layout_and_style_ids": None,
        }
    raise ValueError('split must be {None,"all","pretrain","target"}')


class RoboCasaEnvFacade(BaseEnvFacade):
    """Wraps the raw robosuite env and exposes ONLY basic calls via RPC."""

    def __init__(
        self,
        task_name,
        split="target",
        seed=0,
        camera_h=256,
        camera_w=256,
        cameras=None,
        use_camera_obs=False,
    ):
        self._perception_isolation = _env_flag("RLDX_PERCEPTION_ISOLATION", "0")
        self._allow_reset = _env_flag("RLDX_ALLOW_RESET", "0")
        self._initial_reset_consumed = False
        super().__init__()
        import robocasa  # noqa: F401 — registers robocasa envs
        import robosuite
        from robosuite.controllers import load_composite_controller_config

        self.task_name = task_name
        self.split = split
        self.seed = seed
        self.cameras = list(cameras) if cameras else list(DEFAULT_CAMS)
        self.camera_h, self.camera_w = camera_h, camera_w

        controller_config = load_composite_controller_config(
            controller=None, robot="PandaOmron"
        )
        env_kwargs = dict(
            env_name=task_name,
            robots="PandaOmron",
            controller_configs=controller_config,
            camera_names=self.cameras,
            camera_widths=camera_w,
            camera_heights=camera_h,
            has_renderer=False,
            has_offscreen_renderer=True,
            ignore_done=True,
            use_object_obs=True,
            use_camera_obs=use_camera_obs,  # off -> no per-step render (EGL-safe OSC loops)
            camera_depths=False,  # depth rendered on demand
            seed=seed,
            **_split_kwargs(split),
        )
        self.env = robosuite.make(**env_kwargs)
        self._runtime_id = uuid.uuid4().hex
        self._episode_id = 0
        self._sim_step = 0
        self._success_latched = False
        self._camera_names = self._discover_camera_names()
        required_cameras = set(DEFAULT_CAMS)
        if self._perception_isolation:
            required_cameras.add(NAV_CAMERA)
        missing_cameras = required_cameras.difference(self._camera_names)
        if missing_cameras:
            raise RuntimeError(
                "RoboCasa camera contract is not satisfied; missing "
                f"{sorted(missing_cameras)}. {NAV_CAMERA!r} requires the PandaOmron "
                "navview XML patch on robosuite commit 85abee228d1c43ab1939bce33028099945d453b4."
            )
        if int(self.env.action_dim) != RLDX_ACTION_DIM:
            raise RuntimeError(
                f"RLDX/RoboCasa action schema requires {RLDX_ACTION_DIM} values, "
                f"but env.action_dim={self.env.action_dim}"
            )
        self._meta = {
            "task_name": self.task_name,
            "split": self.split,
            "seed": self.seed,
            "camera_h": self.camera_h,
            "camera_w": self.camera_w,
        }

    def _register_rpc(self):
        """Register all RPC methods."""
        super()._register_rpc()
        self._rpc["env.check_success"] = self.check_success
        self._rpc["env.get_camera_transform"] = self.get_camera_transform
        self._rpc["env.grasp_contact"] = self.grasp_contact
        self._rpc["env.reassemble_env_action"] = self.reassemble_env_action
        if not self._perception_isolation:
            self._rpc["env.get_success_criteria_text"] = self.get_success_criteria_text
            self._rpc["env.get_task_progress"] = self.get_task_progress
        self._rpc["env.get_runtime_info"] = self.get_runtime_info
        # Read-only methods
        self._readonly_methods.update(
            [
                "env.get_camera_transform",
                "env.check_success",
                "env.grasp_contact",
                "env.get_runtime_info",
            ]
        )
        if not self._perception_isolation:
            self._readonly_methods.update(
                {
                    "env.get_success_criteria_text",
                    "env.get_task_progress",
                }
            )

    def get_env_meta(self):
        return self._meta

    def _discover_camera_names(self):
        model = self.env.sim.model
        raw_names = getattr(model, "camera_names", ())
        names = (
            {str(name) for name in raw_names if name}
            if raw_names is not None
            else set()
        )
        for name in DEFAULT_CAMS + [NAV_CAMERA]:
            try:
                model.camera_name2id(name)
            except Exception:
                continue
            names.add(name)
        return names

    def get_runtime_info(self):
        return {
            "protocol_version": ENV_PROTOCOL_VERSION,
            "runtime_id": self._runtime_id,
            "episode_id": self._episode_id,
            "sim_step": self._sim_step,
            "success_latched": self._success_latched,
            "perception_isolation": self._perception_isolation,
            "robot": "PandaOmron",
            "action_schema": {
                "name": "robocasa.panda_omron.flat.v1",
                "flat_dim": RLDX_ACTION_DIM,
                "fields": [
                    {"name": name, "size": size} for name, size in RLDX_ACTION_FIELDS
                ],
            },
            "cameras": sorted(self._camera_names),
        }

    # ---- lifecycle ----
    def reset(self):
        if (
            getattr(self, "_perception_isolation", False)
            and not getattr(self, "_allow_reset", False)
            and getattr(self, "_initial_reset_consumed", False)
        ):
            raise PermissionError(
                "reset is disabled after formal episode initialization"
            )
        # RLDX_RESET_SEED=<episode_seed> -> reproduce the EXACT scene the fullshot eval
        # generated for that episode, seeded the SAME way as the eval's VideoRecordingWrapper
        # (random.seed + np.random.seed + robosuite env.rng/seed) BEFORE reset. Lets the
        # hybrid run on the IDENTICAL reset layouts fullshot was scored on (true paired
        # comparison). The eval formula: episode_seed = (run_seed + env_idx)*100000 + episode_id.
        rs_env = os.environ.get("RLDX_RESET_SEED")
        if rs_env:
            import random

            try:
                sd = int(rs_env)
            except ValueError as exc:
                raise ValueError("RLDX_RESET_SEED must be an integer") from exc
            if sd < 0 or sd > np.iinfo(np.uint32).max:
                raise ValueError("RLDX_RESET_SEED must be in [0, 2**32 - 1]")
            random.seed(sd)
            np.random.seed(sd)
            if hasattr(self.env, "seed"):
                self.env.seed = sd
            if hasattr(self.env, "rng"):
                self.env.rng = np.random.default_rng(sd)
        obs = self.env.reset()
        if not isinstance(obs, dict):
            raise TypeError(
                f"env.reset must return an observation dict, got {type(obs).__name__}"
            )
        self._episode_id += 1
        self._sim_step = 0
        self._success_latched = bool(self.env._check_success())
        self._initial_reset_consumed = True
        return obs

    def step(self, flat_action):
        """flat_action: np.ndarray[12] = [eef_pos(3), eef_rot(3), gripper(1),
        base_motion(4), control_mode(1)] in the PandaOmron composite layout."""
        a = _finite_array(
            flat_action,
            "flat_action",
            shape=(int(self.env.action_dim),),
        )
        if self._success_latched:
            raise RuntimeError(
                "episode success is already latched; further motion is disabled"
            )
        obs, reward, done, info = self.env.step(a)
        if not isinstance(obs, dict):
            raise TypeError(
                f"env.step must return an observation dict, got {type(obs).__name__}"
            )
        self._sim_step += 1
        success_now = bool(self.env._check_success())
        self._success_latched = self._success_latched or success_now
        info = dict(info) if isinstance(info, dict) else {"env_info": info}
        info.update(
            {
                "rpent_runtime_id": self._runtime_id,
                "rpent_episode_id": self._episode_id,
                "rpent_sim_step": self._sim_step,
                "rpent_success_now": success_now,
                "rpent_success_latched": self._success_latched,
            }
        )
        return obs, reward, done, info

    def chunk_step(self, flat_actions, *, return_all_frames=False):
        """Apply a bounded action chunk, stopping at the first latched success."""
        actions = np.asarray(flat_actions)
        if actions.ndim != 2 or actions.shape[1:] != (int(self.env.action_dim),):
            raise ValueError(
                "flat_actions must have shape "
                f"(steps, {self.env.action_dim}), got {actions.shape}"
            )
        if not 1 <= actions.shape[0] <= MAX_ACTION_CHUNK:
            raise ValueError(
                f"action chunk length must be in [1, {MAX_ACTION_CHUNK}], "
                f"got {actions.shape[0]}"
            )
        if not np.issubdtype(actions.dtype, np.number) or np.issubdtype(
            actions.dtype, np.bool_
        ):
            raise TypeError("flat_actions must contain numeric values")
        if not np.isfinite(actions).all():
            raise ValueError("flat_actions must contain only finite values")
        if not isinstance(return_all_frames, (bool, np.bool_)):
            raise TypeError("return_all_frames must be boolean")

        observations = []
        rewards = []
        dones = []
        infos = []
        for action in actions:
            obs, reward, done, info = self.step(action)
            observations.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
            if self._success_latched:
                break
        obs_field = observations if return_all_frames else observations[-1]
        return (
            obs_field,
            np.asarray(rewards, dtype=np.float64),
            np.asarray(dones, dtype=np.bool_),
            infos,
        )

    def check_success(self):
        return self._success_latched

    def _validate_camera_request(self, camera_name, height, width):
        if not isinstance(camera_name, str) or not camera_name:
            raise TypeError("camera_name must be a non-empty string")
        if camera_name not in self._camera_names:
            raise ValueError(
                f"unknown camera {camera_name!r}; available={sorted(self._camera_names)}"
            )
        height = self.camera_h if height is None else height
        width = self.camera_w if width is None else width
        height = _positive_int(height, "height", maximum=MAX_RENDER_DIM)
        width = _positive_int(width, "width", maximum=MAX_RENDER_DIM)
        if height * width > MAX_RENDER_PIXELS:
            raise ValueError(f"render size {height}x{width} exceeds pixel budget")
        return camera_name, height, width

    def render_camera(self, camera_name, height, width, depth):
        """Render one validated camera in robosuite-native orientation."""
        import robosuite.utils.camera_utils as CU

        camera_name, height, width = self._validate_camera_request(
            camera_name, height, width
        )
        if not isinstance(depth, (bool, np.bool_)):
            raise TypeError("depth must be a boolean")
        out = self.env.sim.render(
            width=width,
            height=height,
            camera_name=camera_name,
            depth=depth,
        )
        if depth:
            rgb, normalized_depth = out
            rgb = np.asarray(rgb)
            normalized_depth = np.asarray(normalized_depth)
            if rgb.shape != (height, width, 3):
                raise ValueError(
                    "rendered RGB must have shape "
                    f"{(height, width, 3)}, got {rgb.shape}"
                )
            if normalized_depth.shape not in {
                (height, width),
                (height, width, 1),
            }:
                raise ValueError(
                    "rendered depth must have shape "
                    f"{(height, width)} or {(height, width, 1)}, "
                    f"got {normalized_depth.shape}"
                )
            normalized_depth = np.nan_to_num(
                normalized_depth,
                nan=1.0,
                posinf=1.0,
                neginf=0.0,
            )
            normalized_depth = np.clip(normalized_depth, 0.0, 1.0)
            if normalized_depth.ndim == 3:
                real_depth = CU.get_real_depth_map(self.env.sim, normalized_depth)[
                    ..., 0
                ]
            else:
                real_depth = CU.get_real_depth_map(
                    self.env.sim, normalized_depth[..., None]
                )[..., 0]
            return rgb, real_depth
        rgb = np.asarray(out)
        if rgb.shape != (height, width, 3):
            raise ValueError(
                f"rendered RGB must have shape {(height, width, 3)}, got {rgb.shape}"
            )
        return rgb

    def get_camera_meta(self, camera_name, height=None, width=None):
        import robosuite.utils.camera_utils as CU

        camera_name, height, width = self._validate_camera_request(
            camera_name, height, width
        )
        K = CU.get_camera_intrinsic_matrix(self.env.sim, camera_name, height, width)
        Ext = CU.get_camera_extrinsic_matrix(self.env.sim, camera_name)  # cam->world
        K = _finite_array(K, "camera intrinsic", shape=(3, 3))
        Ext = _finite_array(Ext, "camera extrinsic", shape=(4, 4))
        m = self.env.sim.model
        extent = m.stat.extent
        near = float(m.vis.map.znear * extent)
        far = float(m.vis.map.zfar * extent)
        if not np.isfinite([near, far]).all() or not (0 < near < far):
            raise ValueError(f"invalid camera depth range near={near}, far={far}")
        return {
            "camera_name": camera_name,
            "height": height,
            "width": width,
            "intrinsic": K.tolist(),
            "extrinsic_cam2world": Ext.tolist(),
            "depth_near": near,
            "depth_far": far,
            "runtime_id": self._runtime_id,
            "episode_id": self._episode_id,
            "sim_step": self._sim_step,
        }

    def get_camera_transform(self, camera_name, height=None, width=None):
        import robosuite.utils.camera_utils as CU

        camera_name, height, width = self._validate_camera_request(
            camera_name, height, width
        )
        T = CU.get_camera_transform_matrix(self.env.sim, camera_name, height, width)
        T = _finite_array(T, "camera transform", shape=(4, 4))
        try:
            inverse = np.linalg.inv(T)
        except np.linalg.LinAlgError as exc:
            raise ValueError("camera transform is singular") from exc
        return inverse  # T_p2w

    def get_task_language(self) -> str | None:
        return self.env.get_ep_meta().get("lang")

    def grasp_contact(self):
        """Check if the gripper is currently contacting a task object."""
        try:
            robo = self.env  # robosuite Kitchen env
            grip = robo.robots[0].gripper  # {"right": GripperModel}
            for name, obj in robo.objects.items():
                try:
                    if robo._check_grasp(grip, obj):
                        return True, name
                except Exception:
                    continue
        except Exception:
            pass
        return False, None

    def reassemble_env_action(self, unmap_result):
        """Reassemble the unmap result into a flat action using the env's robots."""
        from robosuite.controllers.composite.composite_controller import (
            HybridMobileBase,
        )

        if not isinstance(unmap_result, Mapping):
            raise TypeError("unmap_result must be a mapping")
        parts = dict(unmap_result)
        env_action = []
        for robot in self.env.robots:
            cc = robot.composite_controller
            pf = robot.robot_model.naming_prefix
            a = np.zeros(cc.action_limits[0].shape)
            for part_name in cc.part_controllers:
                s, e = cc._action_split_indexes[part_name]
                key = f"{pf}{part_name}"
                if key not in parts:
                    raise ValueError(f"unmap_result is missing {key!r}")
                a[s:e] = _finite_action_part(parts.pop(key), key, e - s)
            if isinstance(cc, HybridMobileBase):
                key = f"{pf}base_mode"
                if key not in parts:
                    raise ValueError(f"unmap_result is missing {key!r}")
                a[-1] = _finite_action_part(parts.pop(key), key, 1)[0]
            env_action.append(a)
        if parts:
            raise ValueError(f"unmap_result has unexpected keys: {sorted(parts)}")
        return _finite_array(
            np.concatenate(env_action),
            "reassembled env action",
            shape=(int(self.env.action_dim),),
        )

    def get_success_criteria_text(self):
        """Return the success_criteria.md text for this task."""
        if self._perception_isolation:
            raise PermissionError(
                "success criteria are unavailable in perception-isolated evaluation"
            )
        env = self.env
        out = []
        try:
            src = inspect.getsource(type(env)._check_success)
            out.append(
                "# SUCCESS CONDITION for this task (env._check_success)\n"
                "# You must make this return True. Object positions are NOT given —\n"
                "# localize every named object/fixture from the camera+world maps.\n\n"
                + src
            )
            try:
                import robocasa.utils.object_utils as OU

                for fn in sorted(set(re.findall(r"OU\.(\w+)\(", src))):
                    f = getattr(OU, fn, None)
                    if f is not None:
                        try:
                            out.append(
                                "## helper OU.%s\n%s" % (fn, inspect.getsource(f))
                            )
                        except Exception:
                            pass
            except Exception:
                pass
            for fix, meth in sorted(set(re.findall(r"self\.(\w+)\.(\w+)\(", src))):
                obj = getattr(env, fix, None)
                if obj is not None and hasattr(type(obj), meth):
                    try:
                        out.append(
                            "## %s.%s\n%s"
                            % (fix, meth, inspect.getsource(getattr(type(obj), meth)))
                        )
                    except Exception:
                        pass
        except Exception as ex:
            out.append("(_check_success extraction failed: %s)" % ex)
        return "\n\n".join(out)[:9000]

    def get_task_progress(self):
        """Return the progress dict for this task."""
        if self._perception_isolation:
            raise PermissionError(
                "task progress is unavailable in perception-isolated evaluation"
            )
        env = self.env
        prog = {}
        code = type(env)._check_success.__code__
        try:
            src = inspect.getsource(type(env)._check_success)
            # capture both `self.attr` AND dotted `self.fixture._attr` paths used in the
            # success check (e.g. self.coffee_machine._turned_on) — a bare-attr regex
            # would only grab "coffee_machine" (the fixture object) and miss the real
            # gating flag. Resolve each dotted path to its live scalar/bool value.
            for path in sorted(
                set(re.findall(r"self\.([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)", src))
            ):
                obj = env
                ok = True
                for part in path.split("."):
                    obj = getattr(obj, part, None)
                    if obj is None:
                        ok = False
                        break
                if not ok:
                    continue
                key = path.replace(".", "_")
                if isinstance(obj, (bool, np.bool_)):
                    prog[key] = bool(obj)
                elif isinstance(obj, (int, np.integer)):
                    prog[key] = int(obj)
                elif isinstance(obj, (float, np.floating)):
                    prog[key] = round(float(obj), 4)
        except Exception:
            pass
        # trace ONE read-only call of _check_success; grab its return-frame locals
        captured = {}

        def _tracer(frame, event, arg):
            if event == "call" and frame.f_code is code:

                def _local(f, e, a):
                    if e == "return":
                        captured.update(f.f_locals)
                    return _local

                return _local
            return None

        old = sys.gettrace()
        try:
            sys.settrace(_tracer)
            env._check_success()
        except Exception:
            pass
        finally:
            sys.settrace(old)
        for k, v in captured.items():
            if k == "self" or k in prog:
                continue
            if isinstance(v, (bool, np.bool_)):
                prog[k] = bool(v)
            elif isinstance(v, (int, np.integer)):
                prog[k] = int(v)
            elif isinstance(v, (float, np.floating)):
                prog[k] = round(float(v), 4)
        return prog

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass

    def serve(self, *, transport, host, port, parent_watch=False):
        """Override: single render-thread dispatch so EGL context stays current."""
        work_queue = queue.Queue()

        def render_loop():
            while (item := work_queue.get()) is not None:
                event, req = item
                try:
                    req["result"] = self._dispatch(
                        req["method"], req["args"], req["kwargs"]
                    )
                except Exception:
                    req["error"] = traceback.format_exc()
                event.set()

        threading.Thread(target=render_loop, name="egl-render", daemon=True).start()

        def dispatch(method, args, kwargs):
            if method == "healthz":
                return {"status": "ok"}
            if method == "shutdown":
                self._shutdown_event.set()
                return {"ok": True}
            event = threading.Event()
            req = {
                "method": method,
                "args": args,
                "kwargs": kwargs,
                "result": None,
                "error": None,
            }
            work_queue.put((event, req))
            event.wait()
            if req["error"]:
                raise RuntimeError(req["error"])
            return req["result"]

        server_cls = HttpRpcServer if transport == "http" else SocketRpcServer
        server = server_cls((host, port), dispatch)
        bound_host, bound_port = server.server_address
        bound_host = "127.0.0.1" if bound_host == "0.0.0.0" else bound_host
        url = f"{transport}://{bound_host}:{bound_port}"
        print(f"RPC server listening on {url}", flush=True)
        logger.info("RPC server listening on %s", url)

        if parent_watch:
            watch_parent_death(self._shutdown_event.set)

        try:
            threading.Thread(target=server.serve_forever, daemon=True).start()
            self._shutdown_event.wait()
        finally:
            work_queue.put(None)
            server.shutdown()
            server.server_close()
            self.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--transport", choices=["socket", "http"], default="http")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=0)
    p.add_argument(
        "--parent-watch",
        action="store_true",
        help="watch parent process via stdin pipe and exit when it dies",
    )
    p.add_argument(
        "--cuda-device",
        type=int,
        default=None,
        help="GPU device to pin MuJoCo EGL rendering and the torch "
        "default device to (physical CUDA ordinal).",
    )
    p.add_argument("--task-name", default="OpenDrawer")
    p.add_argument("--split", default="target")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.cuda_device is not None:
        # Deliberately do NOT set CUDA_VISIBLE_DEVICES. robosuite (imported
        # transitively via libero) asserts at import time that
        # ``MUJOCO_EGL_DEVICE_ID in CUDA_VISIBLE_DEVICES`` (substring check),
        # which assumes the EGL index equals the CUDA ordinal and crashes on
        # multi-GPU boxes where the EGL order differs. That assertion is gated
        # on ``CUDA_VISIBLE_DEVICES != ""``, so leaving it unset skips it in
        # both this process and the multiprocessing-spawned render workers
        # (which inherit the env). Pin the two backends directly instead:
        #   - MuJoCo render device <- MUJOCO_EGL_DEVICE_ID (configure_egl_device)
        #   - torch default device  <- torch.cuda.set_device(N)
        prev = os.environ.get("CUDA_VISIBLE_DEVICES")
        if prev is not None:
            logger.warning(
                "CUDA_VISIBLE_DEVICES=%s is set; clearing it and pinning via "
                "MUJOCO_EGL_DEVICE_ID + torch.cuda.set_device(--cuda-device=%s) "
                "instead (robosuite's CVD assertion is incompatible with EGL<->CUDA mapping)",
                prev,
                args.cuda_device,
            )
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        from rpent.utils.egl import configure_egl_device

        configure_egl_device(args.cuda_device)
        import torch

        torch.cuda.set_device(args.cuda_device)

    facade = RoboCasaEnvFacade(
        args.task_name,
        split=args.split,
        seed=args.seed,
    )
    facade.serve(
        transport=args.transport,
        host=args.host,
        port=args.port,
        parent_watch=args.parent_watch,
    )


if __name__ == "__main__":
    main()
