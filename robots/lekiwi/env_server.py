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

"""RPC server owning one RLinf LeKiwi ``RealWorldEnv`` in this process.

The env runs in-process — no Ray, no scheduler cluster — because the robot's
computer is a small board where five scheduler manager processes (~0.7 GB)
would exhaust memory. ``node_rank`` stays ``None`` throughout the hardware
config, which is how RLinf's parts know to open their devices locally rather
than hosting them on a scheduler worker (``RobotPart.connect``). The RPC
surface is unchanged, so the client and the agent attach exactly as before.
"""

from __future__ import annotations

import argparse
import math
import os
import queue
import signal
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from robots.lekiwi.runtime_config import (
    LEKIWI_ROOT,
    get_robot_config_path,
    load_runtime_config,
    runtime_settings_from_args,
    set_robot_config_path,
)
from rpent.robots.components.env_facade_base import BaseEnvFacade
from rpent.utils.logging import get_logger
from rpent.utils.serialization import to_numpy_tree

logger = get_logger("lekiwi_env_server")


class LeKiwiEnvFacade(BaseEnvFacade):
    """Expose the LeKiwi ``env.*`` protocol from an in-process env backend."""

    _METHODS = (
        "get_env_meta",
        "reset",
        "get_robot_state",
        "get_observation",
        "get_camera_meta",
        "drive",
        "hold",
        "stop",
    )
    # These handlers synchronize internally instead of holding the RPC write lock.
    _RUNTIME_METHODS = (
        "get_stereo_frame",
        "get_navigation_state",
        "navigate_to",
        "list_map_nodes",
        "inspect_map_node",
        "view_map",
        "recording_start",
        "recording_state",
        "recording_stop",
    )
    _READONLY = (
        "get_env_meta",
        "get_robot_state",
        "get_observation",
        "get_camera_meta",
        "get_navigation_state",
        "get_stereo_frame",
        "list_map_nodes",
        "inspect_map_node",
        "view_map",
    )

    def __init__(self, backend: Any) -> None:
        self._backend = backend
        self._control = getattr(backend, "control", None)
        super().__init__()

    def _register_rpc(self) -> None:
        for name in (*self._METHODS, *self._RUNTIME_METHODS):
            self._rpc[f"env.{name}"] = getattr(self._backend, name)
        self._readonly_methods.update(f"env.{name}" for name in self._READONLY)
        self._runtime_methods = {f"env.{name}" for name in self._RUNTIME_METHODS}
        if self._control is not None:
            self._rpc.update(
                {
                    "base.navigation_info": self._control.navigation_info,
                    "base.set_velocity": self._control.set_velocity,
                    "base.get_state": self._control.state,
                }
            )
            self._readonly_methods.add("base.get_state")

    def _dispatch(self, method, args, kwargs, *, session_id=None):
        # Stop must not wait behind an active motion RPC or session lock.
        if method == "env.stop":
            return self._backend.stop()
        return super()._dispatch(method, args, kwargs, session_id=session_id)

    def _dispatch_nosession(self, method, args, kwargs):
        # Shared hardware handlers synchronize themselves. In particular,
        # stop must not wait for a bounded drive's RPC write lock.
        if self._control is not None and (
            method.startswith("base.")
            or method in {"env.drive", "env.hold"}
            or method in self._runtime_methods
        ):
            return self._rpc[method](*args, **kwargs)
        return super()._dispatch_nosession(method, args, kwargs)

    def _bind_and_announce(self, transport, host, port, dispatch):
        server = super()._bind_and_announce(transport, host, port, dispatch)
        control = self._control
        if control is not None:
            try:
                control.start()
                endpoint = f"{transport}://127.0.0.1:{server.server_address[1]}"
                self._backend.runtime.endpoint = endpoint
                if self._backend.runtime_settings["mode"] == "navigation":
                    threading.Thread(
                        target=self._start_navigation, args=(endpoint,), daemon=True
                    ).start()
            except BaseException:
                server.server_close()
                control.close()
                raise
        return server

    def _start_navigation(self, endpoint: str) -> None:
        from rpent.utils.rpc import make_rpc_client
        from rpent.utils.rpc.client_utils import wait_for_ready

        client = make_rpc_client(endpoint)
        try:
            wait_for_ready(client, timeout_s=10)
            if not self._shutdown_event.is_set():
                self._backend.start_navigation()
        except Exception:
            logger.exception("Navigation startup failed")
            self._shutdown_event.set()
        finally:
            client.close()


def _local_worker_info(cfg: Any) -> Any:
    """Describe a one-worker, one-robot placement that owns hardware locally.

    ``cluster_node_rank`` and the hardware config's ``node_rank`` are ``None``:
    that is what makes every part open its device in this process. The cluster
    section's ``node_rank: 0`` would host each part on a scheduler worker and
    drag in the whole Ray cluster, which the robot's computer cannot carry.
    """
    from omegaconf import OmegaConf
    from rlinf.robotics.discovery import RobotInfo
    from rlinf.scheduler import AcceleratorType, WorkerAddress, WorkerInfo

    hardware = OmegaConf.to_container(cfg.cluster.node_groups[0].hardware, resolve=True)
    from rlinf.robotics.robots.lekiwi import LeKiwiConfig

    robot_config = LeKiwiConfig(**{**hardware["configs"][0], "node_rank": None})
    # LeKiwi reports its type as its model (RobotConfig.hardware_model).
    robot_info = RobotInfo(
        type=hardware["type"],
        model=hardware["type"],
        config=robot_config,
    )
    return WorkerInfo(
        address=WorkerAddress("lekiwi", 0),
        rank=0,
        group_world_size=1,
        cluster_node_rank=None,
        accelerator_type=AcceleratorType.NO_ACCEL,
        accelerator_model="",
        accelerator_rank=0,
        node_ip="",
        node_port=0,
        available_accelerators=[],
        hardware_infos=[robot_info],
    )


class LeKiwiEnvBackend:
    """Expose one RLinf-owned robot and a single step loop through RPC."""

    def __init__(
        self,
        cfg: Any,
        controller_config: dict[str, Any],
        *,
        control_settings,
        simulated=False,
    ) -> None:
        from rlinf.envs.real.env import RealWorldEnv

        from robots.lekiwi.control import EnvControl
        from robots.lekiwi.rpent_env import register_rpent_lekiwi_env

        register_rpent_lekiwi_env()
        self.cfg = cfg
        self.controller = dict(controller_config)
        self.env = RealWorldEnv(
            cfg.env.eval,
            num_envs=1,
            seed_offset=0,
            total_num_processes=1,
            worker_info=_local_worker_info(cfg),
        )
        raw = self._raw_rlinf_env()
        self.camera = next(iter(getattr(raw, "_cameras", [])), None)
        self.action_dim = int(self.env.action_space.shape[-1])
        self.step_frequency = float(raw.config.step_frequency)
        self.velocity_limits = np.asarray(
            self.env.env.single_action_space.high, dtype=np.float32
        )
        self.step_period_s = 1 / self.step_frequency
        self.control = EnvControl(
            self.env, raw, simulated=simulated, **control_settings
        )
        self._stop_requested = threading.Event()
        self._manual_operation = threading.Lock()
        self.runtime = None
        self.recorder = None
        self.runtime_settings = {}
        self._services_lock = threading.RLock()

    def get_env_meta(self) -> dict[str, Any]:
        return {
            "ok": True,
            "action_dim": self.action_dim,
            "velocity_limits": self.velocity_limits.tolist(),
            "step_frequency": self.step_frequency,
            "camera_available": True,
            "control_mode": self.control.mode,
        }

    def close_env(self) -> None:
        self._stop_requested.set()
        runtime = self.runtime
        try:
            if runtime is not None:
                runtime.close()
            if self.recorder is not None:
                self.recorder.stop()
        finally:
            try:
                self.control.close()
            finally:
                self.env.close()

    @staticmethod
    def _strip_batch(value: Any) -> Any:
        array = to_numpy_tree(value)
        if isinstance(array, np.ndarray) and array.ndim > 0 and array.shape[0] == 1:
            return array[0]
        if isinstance(array, list) and len(array) == 1:
            return array[0]
        return array

    def _raw_rlinf_env(self) -> Any:
        """Return RPent's unwrapped RLinf environment compatibility layer."""
        return self.env.env.envs[0].unwrapped

    def reset(self) -> dict[str, Any]:
        with self._manual_operation, self.control.step_lock:
            with self.control.lock:
                if self.control.mode != "manual":
                    raise RuntimeError("Stop navigation before resetting the Env")
                self.control.stop()
            observation, info = self.env.reset()
            self.control._update_sample()
        return {
            "ok": True,
            "info": to_numpy_tree(info),
            "robot_state": self.get_robot_state(),
            "states": self._strip_batch(observation.get("states")),
            "main_images": self._strip_batch(observation.get("main_images")),
            "steps": self.control.steps,
        }

    def get_robot_state(self) -> dict[str, Any]:
        state = self.control.state()
        velocity = state.get("measured_velocity", [0.0, 0.0, 0.0])
        return {
            "velocity": [*velocity[:2], math.degrees(velocity[2])],
            "action_dim": self.action_dim,
            "velocity_limits": self.velocity_limits.tolist(),
            "step_frequency": self.step_frequency,
            "steps": self.control.steps,
            "control_error": state["fault"],
        }

    def get_observation(self) -> dict[str, Any]:
        """Return the newest step observation, or ``{}`` when it is stale."""
        with self.control.lock:
            observation = self.control.observation
            if (
                observation is None
                or time.monotonic() - self.control.observed_at > 2 * self.step_period_s
            ):
                return {}
            payload = {
                "states": self._strip_batch(observation.get("states")),
                "main_images": self._strip_batch(observation.get("main_images")),
                "steps": self.control.steps,
            }
        extras = observation.get("extra_view_images")
        if extras is not None:
            payload["extra_view_images"] = self._strip_batch(extras)
        return payload

    def get_camera_meta(self) -> dict[str, Any]:
        metadata = to_numpy_tree(self._raw_rlinf_env().get_camera_metadata())
        main_key = self.cfg.env.eval.get("main_image_key")
        names = sorted(metadata.get("cameras", {}))
        extras = [name for name in names if name != main_key]
        metadata["observation_camera_map"] = {
            "main": main_key,
            **{f"extra_{index}": name for index, name in enumerate(extras)},
        }
        return metadata

    def _submit_manual(self, velocity, duration):
        requested = self._validate_velocity(velocity)
        action = self._clip(requested)
        result = self.control.submit(
            [*action[:2], math.radians(float(action[2]))],
            mode="manual",
            expires_at=time.time() + duration,
        )
        return requested, [*result["target"][:2], math.degrees(result["target"][2])]

    def drive(self, velocity: Any, *, duration_s: float) -> dict[str, Any]:
        if not math.isfinite(duration_s) or duration_s <= 0:
            raise ValueError("duration_s must be finite and positive")
        duration_s = min(duration_s, self.controller["drive_max_duration_s"])
        with self._manual_operation:
            self._stop_requested.clear()
            with self.control.lock:
                requested, commanded = self._submit_manual(velocity, duration_s)
                generation = self.control.generation
                started_steps = self.control.steps
            end = time.monotonic() + duration_s
            cancelled = False
            try:
                while time.monotonic() < end:
                    if (
                        self._stop_requested.wait(
                            min(self.step_period_s, max(0, end - time.monotonic()))
                        )
                        or self.control.generation != generation
                    ):
                        cancelled = True
                        break
                    if fault := self.control.state()["fault"]:
                        raise RuntimeError(fault)
            finally:
                with self.control.step_lock, self.control.lock:
                    if self.control.generation == generation:
                        self.control.stop()
            observation = self.control.observation
            return {
                "ok": not cancelled,
                "cancelled": cancelled,
                "requested_velocity": requested.tolist(),
                "commanded_velocity": commanded,
                "steps": self.control.steps - started_steps,
                "duration_s": duration_s,
                "robot_state": self.get_robot_state(),
                "states": self._strip_batch(observation.get("states"))
                if observation
                else None,
            }

    def hold(self, velocity: Any, *, ttl_s: float = 1.0) -> dict[str, Any]:
        if not math.isfinite(ttl_s) or ttl_s <= 0:
            raise ValueError("ttl_s must be finite and positive")
        ttl_s = min(ttl_s, self.controller["hold_max_ttl_s"])
        if not self._manual_operation.acquire(blocking=False):
            raise RuntimeError("A bounded drive or reset is active")
        try:
            requested, commanded = self._submit_manual(velocity, ttl_s)
        finally:
            self._manual_operation.release()
        return {
            "ok": True,
            "requested_velocity": requested.tolist(),
            "commanded_velocity": commanded,
            "ttl_s": ttl_s,
            "steps": self.control.steps,
        }

    def stop(self) -> dict[str, Any]:
        """Cancel motion without closing the navigation service."""
        self._stop_requested.set()
        runtime = self.runtime
        if runtime is None:
            self.control.stop()
        else:
            runtime.cancel()
        return {"ok": True, "robot_state": self.get_robot_state()}

    def _validate_velocity(self, velocity: Any) -> np.ndarray:
        requested = np.asarray(velocity, dtype=np.float32)
        if requested.shape != (self.action_dim,):
            raise ValueError(
                f"velocity must contain exactly {self.action_dim} values, "
                f"got shape {requested.shape}"
            )
        if not np.isfinite(requested).all():
            raise ValueError("velocity must contain only finite values")
        return requested

    def _clip(self, requested: np.ndarray) -> np.ndarray:
        return np.clip(requested, -self.velocity_limits, self.velocity_limits)

    def configure_services(self, settings: dict) -> None:
        from robots.lekiwi.navigation.runtime import NavigationRuntime

        self.runtime_settings = settings
        self.runtime = NavigationRuntime(settings, self.control)

    def get_stereo_frame(self) -> dict | None:
        """Return a recent stereo frame, or None while capture recovers."""
        try:
            stamp, frame = self.camera.get_frame(
                timeout=0.5, max_age=0.5, return_timestamp=True
            )
        except queue.Empty:
            return None
        return {"frame": frame, "source_stamp": stamp, "clock_domain": "host_monotonic"}

    def start_navigation(self) -> dict:
        with self._services_lock:
            if self.recorder is not None and self.recorder.alive:
                raise RuntimeError("Stop recording before starting navigation")
            return self.runtime.start()

    def get_navigation_state(self) -> dict:
        return self.runtime.state()

    def list_map_nodes(self, *, label=None) -> dict:
        return self.runtime.list_map_nodes(label=label)

    def inspect_map_node(self, node_id: int) -> dict:
        return self.runtime.inspect_map_node(node_id)

    def view_map(self, *, center_node_id=None, radius_m=None) -> dict:
        return self.runtime.view_map(center_node_id=center_node_id, radius_m=radius_m)

    def navigate_to(self, target: int | str, *, timeout_s: float = 60) -> dict:
        return self.runtime.navigate_to(target, timeout_s=timeout_s)

    def recording_start(self, identity: str) -> dict:
        from robots.lekiwi.recording import StereoRecording
        from robots.lekiwi.runtime_config import load_device_config

        with self._services_lock:
            if self.control.simulated or self.camera is None:
                raise RuntimeError("Recording requires a real Env")
            if self.control.mode != "manual":
                raise RuntimeError("Stop navigation before recording")
            if self.recorder is not None and self.recorder.alive:
                raise RuntimeError("A recording is already active")
            config = dict(self.runtime_settings["recording"])
            devices = load_device_config(self.runtime_settings["robot_config"])
            config.update(serial=devices["serial"], calibration=devices["calibration"])
            self.recorder = StereoRecording(self.camera, config, identity)
            self.recorder.start()
            return dict(self.recorder.state)

    def recording_state(self, identity: str) -> dict:
        with self._services_lock:
            if self.recorder is None or identity != self.recorder.state["identity"]:
                raise ValueError("Unknown recording")
            return dict(self.recorder.state)

    def recording_stop(self, identity: str) -> dict:
        with self._services_lock:
            self.recording_state(identity)
            self.recorder.stop()
            return dict(self.recorder.state)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=["http", "socket"], default="http")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--robot-config", default=None)
    parser.add_argument("--parent-watch", action="store_true")
    parser.add_argument("--mock", action="store_true", help="Use simulated wheels")
    parser.add_argument(
        "--mode",
        choices=["capture", "navigation"],
        default="capture",
        help="Startup mode (default: capture)",
    )
    parser.add_argument(
        "--ros-domain-id",
        type=int,
        default=79,
        help="ROS domain (default: 79)",
    )
    parser.add_argument(
        "--ros-python",
        type=Path,
        help="Native ROS Python; default: current interpreter",
    )
    parser.add_argument(
        "--map",
        dest="map_yaml",
        type=Path,
        help="Nav2 map YAML; default: map.yaml beside robot YAML",
    )
    parser.add_argument(
        "--capture-seconds",
        type=int,
        default=600,
        help="Stereo recording duration in seconds (default: 600)",
    )
    parser.add_argument(
        "--recording-dir",
        type=Path,
        default=LEKIWI_ROOT / "local/data/recordings",
        help="Stereo image directory; default: local/data/recordings",
    )
    parser.add_argument(
        "--recording-fps",
        type=float,
        default=5,
        help="Maximum saved stereo pairs per second (default: 5)",
    )
    parser.add_argument("--max-linear", type=float, default=None)
    parser.add_argument("--max-angular-deg", type=float, default=None)
    parser.add_argument("--command-timeout", type=float, default=0.3)
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the resolved RLinf config and exit without launching.",
    )
    args = parser.parse_args()
    if not 0 <= args.ros_domain_id <= 232:
        parser.error("--ros-domain-id must be in 0..232")
    if args.host not in {"localhost", "127.0.0.1", "0.0.0.0"}:
        parser.error("Use localhost or 0.0.0.0 so local ROS workers can reach the Env")
    args.robot_config = str(
        Path(args.robot_config or get_robot_config_path()).expanduser().resolve()
    )
    return args


def _export_source_paths() -> None:
    """Let ROS worker processes import the same RPent and RLinf sources."""
    import rlinf

    sources = [str(Path(__file__).resolve().parents[2])]
    sources.extend(str(Path(root).resolve().parent) for root in rlinf.__path__)
    sources.extend(filter(None, os.environ.get("PYTHONPATH", "").split(os.pathsep)))
    os.environ["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(sources))


def main(
    *,
    backend_class: type[LeKiwiEnvBackend] = LeKiwiEnvBackend,
    load_runtime_config: Any = load_runtime_config,
    facade_class: type[LeKiwiEnvFacade] = LeKiwiEnvFacade,
) -> int:
    from robots.lekiwi.runtime_config import (
        control_settings,
        load_device_config,
    )

    args = _parse_args()
    runtime_settings = runtime_settings_from_args(args)
    runtime = load_runtime_config(args.robot_config)
    env_options = runtime.rlinf.env.eval.override_cfg
    settings = control_settings(args, env_options)
    env_options.max_linear_velocity = settings["max_linear"]
    env_options.max_angular_velocity = math.degrees(settings["max_angular"])
    env_options.mock_wheels = args.mock
    dummy = bool(env_options.get("is_dummy", False))
    simulated = args.mock or dummy
    if not dummy:
        period = 1 / float(env_options.get("step_frequency", 10.0))
        if period >= settings["command_timeout"]:
            raise ValueError("RLinf step period must be shorter than command_timeout")
    if args.print_config:
        import json

        from omegaconf import OmegaConf

        hardware = runtime.rlinf.cluster.node_groups[0].hardware.configs[0]
        print(
            json.dumps(
                {
                    "rlinf": OmegaConf.to_container(runtime.rlinf, resolve=True),
                    "hardware": {
                        key: hardware[key]
                        for key in ("port", "wheel_radius", "base_radius", "max_raw")
                    },
                    "control": settings,
                    "simulated": simulated,
                    "runtime": runtime_settings,
                    "devices": {} if dummy else load_device_config(args.robot_config),
                },
                indent=2,
            )
        )
        return 0

    set_robot_config_path(args.robot_config)
    _export_source_paths()
    backend = None
    try:
        backend = backend_class(
            runtime.rlinf,
            runtime.controller,
            control_settings=settings,
            simulated=simulated,
        )
        backend.configure_services(runtime_settings)
        facade = facade_class(backend)
        signal.signal(signal.SIGTERM, lambda *_: facade._shutdown_event.set())
        facade.serve(
            transport=args.transport,
            host=args.host,
            port=args.port,
            parent_watch=args.parent_watch,
        )
    except KeyboardInterrupt:
        pass
    finally:
        if backend is not None:
            backend.close_env()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
