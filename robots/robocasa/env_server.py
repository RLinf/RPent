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
        self._rpc["env.get_success_criteria_text"] = self.get_success_criteria_text
        self._rpc["env.get_task_progress"] = self.get_task_progress
        # Read-only methods
        self._readonly_methods.update(
            [
                "env.check_success",
                "env.get_camera_transform",
                "env.grasp_contact",
                "env.get_success_criteria_text",
                "env.get_task_progress",
            ]
        )

    def get_env_meta(self):
        return self._meta

    # ---- lifecycle ----
    def reset(self):
        # RLDX_RESET_SEED=<episode_seed> -> reproduce the EXACT scene the fullshot eval
        # generated for that episode, seeded the SAME way as the eval's VideoRecordingWrapper
        # (random.seed + np.random.seed + robosuite env.rng/seed) BEFORE reset. Lets the
        # hybrid run on the IDENTICAL reset layouts fullshot was scored on (true paired
        # comparison). The eval formula: episode_seed = (run_seed + env_idx)*100000 + episode_id.
        rs_env = os.environ.get("RLDX_RESET_SEED")
        if rs_env:
            import random

            sd = int(rs_env)
            random.seed(sd)
            np.random.seed(sd)
            if hasattr(self.env, "seed"):
                self.env.seed = sd
            if hasattr(self.env, "rng"):
                self.env.rng = np.random.default_rng(sd)
        return self.env.reset()

    def step(self, flat_action):
        """flat_action: np.ndarray[12] = [eef_pos(3), eef_rot(3), gripper(1),
        base_motion(4), control_mode(1)] in the PandaOmron composite layout."""
        a = np.asarray(flat_action, dtype=np.float64).reshape(-1)
        assert a.shape[0] == self.env.action_dim, (
            f"action dim {a.shape[0]} != env.action_dim {self.env.action_dim}"
        )
        obs, reward, done, info = self.env.step(a)
        return obs, reward, done, info

    def check_success(self):
        return bool(self.env._check_success())

    def render_camera(self, camera_name, height, width, depth):
        """sim.render in ROBOSUITE-NATIVE orientation (matches the camera
        transform matrices). rgb uint8 HxWx3, depth metric HxW."""
        import robosuite.utils.camera_utils as CU

        out = self.env.sim.render(
            width=width, height=height, camera_name=camera_name, depth=depth
        )
        if depth:
            rgb, d = out
            # Sanitize the raw OpenGL normalized depth into [0,1]: replace NaN/inf
            # (degenerate camera pose) then clip numerical overshoot. Otherwise an
            # assertion inside get_real_depth_map crashes the whole env server process.
            d = np.nan_to_num(d, nan=1.0, posinf=1.0, neginf=0.0)
            d = np.clip(d, 0.0, 1.0)
            if d.ndim == 3:
                depth = CU.get_real_depth_map(self.env.sim, d)[..., 0]
            else:
                depth = CU.get_real_depth_map(self.env.sim, d[..., None])[..., 0]
            return rgb, depth
        return out

    def get_camera_meta(self, camera_name, height=None, width=None):
        import robosuite.utils.camera_utils as CU

        K = CU.get_camera_intrinsic_matrix(self.env.sim, camera_name, height, width)
        Ext = CU.get_camera_extrinsic_matrix(self.env.sim, camera_name)  # cam->world
        m = self.env.sim.model
        extent = m.stat.extent
        return {
            "camera_name": camera_name,
            "height": height,
            "width": width,
            "intrinsic": np.asarray(K, dtype=np.float64).tolist(),
            "extrinsic_cam2world": np.asarray(Ext, dtype=np.float64).tolist(),
            "depth_near": float(m.vis.map.znear * extent),
            "depth_far": float(m.vis.map.zfar * extent),
        }

    def get_camera_transform(self, camera_name, height=None, width=None):
        import robosuite.utils.camera_utils as CU

        T = CU.get_camera_transform_matrix(self.env.sim, camera_name, height, width)
        return np.linalg.inv(T)  # T_p2w

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

        env_action = []
        for robot in self.env.robots:
            cc = robot.composite_controller
            pf = robot.robot_model.naming_prefix
            a = np.zeros(cc.action_limits[0].shape)
            for part_name in cc.part_controllers:
                s, e = cc._action_split_indexes[part_name]
                a[s:e] = unmap_result.pop(f"{pf}{part_name}")
            if isinstance(cc, HybridMobileBase):
                a[-1] = unmap_result.pop(f"{pf}base_mode")
            env_action.append(a)
        return np.concatenate(env_action)

    def get_success_criteria_text(self):
        """Return the success_criteria.md text for this task."""
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
                        req["method"],
                        req["args"],
                        req["kwargs"],
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
