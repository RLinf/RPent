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

"""Bounded integration checks against operator-owned BEHAVIOR services.

The operator starts services and the official evaluator separately. These
checks load no checkpoints or credentials and create no child processes.
Closing client connections never sends reset or shutdown. The policy chain
calls the standard Toolkit directly, not a CLI or VLM Planner.
"""

from __future__ import annotations

import os
from argparse import Namespace
from contextlib import ExitStack
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np

from robots.behavior.env_client import BehaviorEnvClient
from robots.behavior.robot_spec import get_robot_spec
from robots.behavior.tools import (
    CAMERA_NAMES,
    extract_public_proprio,
    snapshot_observation,
)
from rpent.dashboard.events import NullDashboardEventSink
from rpent.evaluation import write_json_atomic
from rpent.robots.components.pi05_vla_client import Pi05VLAClient
from rpent.utils.rpc import make_rpc_client
from tests.e2e_tests.common import (
    array_summary,
    parse_runtime_args,
    require_array,
    require_rgb,
)


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required for the opt-in BEHAVIOR checks")
    return value


def runtime_args() -> Namespace:
    """Read test endpoints and task settings through the public robot parser."""
    args = parse_runtime_args(
        get_robot_spec(),
        [
            "--task-id",
            _required_env("RPENT_BEHAVIOR_TASK_ID"),
            "--task-language",
            _required_env("RPENT_BEHAVIOR_TASK_LANGUAGE"),
            "--env-endpoint",
            _required_env("RPENT_BEHAVIOR_ENV_ENDPOINT"),
            "--am-endpoint",
            _required_env("RPENT_BEHAVIOR_VLA_ENDPOINT"),
            "--control-mode",
            "agent",
            "--instruction-mode",
            "main",
            "--execution-horizon",
            "32",
            "--max-chunks-per-call",
            "1",
            "--env-rpc-timeout-s",
            "1230",
            "--tool-allowlist",
            "view_env_state",
            "vla_act",
            "finish",
        ],
    )
    args.output_dir = Path(_required_env("RPENT_E2E_OUTPUT_DIR")).expanduser().resolve()
    get_robot_spec().validate_args(args)
    return args


class _ObservedEnvClient(BehaviorEnvClient):
    """Record submitted native actions while delegating every call to real RPC."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.submitted: list[np.ndarray] = []

    def chunk_step(self, actions: Any, **kwargs: Any) -> tuple:
        self.submitted.append(np.asarray(actions).copy())
        return super().chunk_step(actions, **kwargs)


class _ObservedVLAClient(Pi05VLAClient):
    """Retain actual service predictions for comparison with submitted actions."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.predictions: list[np.ndarray] = []

    def predict(self, env_obs: dict, options: dict | None = None) -> np.ndarray:
        actions = super().predict(env_obs, options)
        self.predictions.append(actions.copy())
        return actions


def _require_tool_images(result: Any) -> None:
    from PIL import Image

    if result.is_error:
        raise RuntimeError(f"BEHAVIOR tool returned an error: {result.data}")
    if len(result.images) != 3:
        raise RuntimeError("BEHAVIOR tool must return all three camera images")
    for name, data in zip(CAMERA_NAMES, result.images, strict=True):
        with Image.open(BytesIO(data)) as decoded:
            if decoded.format != "PNG":
                raise RuntimeError(f"{name} tool image is not PNG")
            require_rgb(np.asarray(decoded.convert("RGB")), name)


class BehaviorScenario:
    """One exclusive test episode using borrowed Env and VLA endpoints."""

    def __init__(self, args: Namespace):
        spec = get_robot_spec()
        spec.validate_args(args)
        self.config = spec.parse_config(args)
        self.output_dir = self.config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.args = args
        self._connections: list[Any] = []
        write_json_atomic(
            self.output_dir / "manifest.json",
            {
                "robot": "behavior",
                "task_id": args.task_id,
                "task_language": args.task_language,
                "env_endpoint": args.env_endpoint,
                "vla_endpoint": args.am_endpoint,
                "execution_horizon": 32,
                "max_chunks_per_call": 1,
                "runtime_ownership": "borrowed_services",
                "scope": "component requests and direct Toolkit chain; no CLI or VLM",
                "official_task_success": None,
            },
        )

    def _connect(self, endpoint: str) -> Any:
        client = make_rpc_client(endpoint)
        self._connections.append(client)
        return client

    @cached_property
    def env(self) -> _ObservedEnvClient:
        return _ObservedEnvClient(
            self._connect(self.args.env_endpoint),
            task_id=self.args.task_id,
            expected_task_language=self.args.task_language,
            timeout_s=1230,
        )

    @cached_property
    def model(self) -> _ObservedVLAClient:
        return _ObservedVLAClient(
            self._connect(self.args.am_endpoint), embodiment="behavior"
        )

    @cached_property
    def environment(self) -> tuple[dict[str, Any], dict[str, Any]]:
        raw = self.env.get_observation()
        public = snapshot_observation(
            raw, task_id=self.args.task_id, task_language=self.args.task_language
        )
        proprio = extract_public_proprio(raw)
        cameras = {
            name: array_summary(require_rgb(public["views"][name]["rgb"], name))
            for name in CAMERA_NAMES
        }
        result = {
            "status": "passed",
            "definition": "real Env observation request; no actions or reset",
            "cameras": cameras,
            "proprio": array_summary(proprio),
            "official_task_success": None,
        }
        write_json_atomic(self.output_dir / "environment-component.json", result)
        return public["policy_observation"], result

    @cached_property
    def vla(self) -> dict[str, Any]:
        observation, _ = self.environment
        prediction = require_array(
            self.model.predict(
                {**observation, "task_descriptions": self.args.task_language}
            ),
            "BEHAVIOR VLA prediction",
            ndim=2,
            last_dim=23,
        )
        np.save(self.output_dir / "component-prediction.npy", prediction)
        result = {
            "status": "passed",
            "definition": "real VLA prediction from official observation; no execution",
            "prediction": array_summary(prediction),
            "prediction_file": "component-prediction.npy",
            "official_task_success": None,
        }
        write_json_atomic(self.output_dir / "vla-component.json", result)
        return result

    @cached_property
    def policy_chain(self) -> dict[str, Any]:
        from robots.behavior.toolkit import BehaviorToolkit
        from rpent.memory import MemoryManager

        toolkit = BehaviorToolkit(
            runtime_kwargs={
                "env": self.env,
                "model": self.model,
                "task_id": self.args.task_id,
                "task_language": self.args.task_language,
                "execution_horizon": 32,
                "max_chunks_per_call": 1,
                "instruction_mode": "main",
                "tool_allowlist": ["view_env_state", "vla_act", "finish"],
            },
            dashboard_events=NullDashboardEventSink(),
            memory=MemoryManager(self.output_dir / "memory", memory_access="read_only"),
            state_output_dir=self.output_dir / "toolkit",
        )
        try:
            _require_tool_images(toolkit.execute_tool("view_env_state", {"step": 0}))
            predictions_before = len(self.model.predictions)
            submissions_before = len(self.env.submitted)
            action_result = toolkit.execute_tool("vla_act", {"chunks": 1})
            _require_tool_images(action_result)
            if len(self.model.predictions) != predictions_before + 1:
                raise RuntimeError("bounded chain must make one real VLA request")
            if len(self.env.submitted) != submissions_before + 1:
                raise RuntimeError("bounded chain must submit one native action chunk")
            predicted = self.model.predictions[-1]
            submitted = self.env.submitted[-1]
            np.testing.assert_array_equal(submitted, predicted[:32])
            count = len(submitted)
            execution = action_result.data["log"]["result"]
            if (
                not 0 < count <= 32
                or any(
                    execution.get(key) != count
                    for key in (
                        "requested_actions",
                        "executed_actions",
                        "issued_actions",
                        "confirmed_actions",
                    )
                )
                or execution.get("unconfirmed_actions") != 0
            ):
                raise RuntimeError(
                    f"native actions were not fully confirmed: {execution}"
                )
            if execution.get("observation_after_last_issued_action") is not True:
                raise RuntimeError("no fresh observation after the submitted actions")
            for name in CAMERA_NAMES:
                require_rgb(toolkit.state.load(f"{name}_rgb.png", step=1), name)
            np.save(self.output_dir / "chain-prediction.npy", predicted)
            np.save(self.output_dir / "chain-submitted-actions.npy", submitted)
            finish = toolkit.execute_tool(
                "finish",
                {
                    "status": "blocked",
                    "summary": "Bounded integration check completed; task success was not evaluated.",
                },
            )
            if (
                finish.is_error
                or finish.data.get("_finish") is not True
                or finish.data.get("status") != "blocked"
            ):
                raise RuntimeError("finish(blocked) did not complete")
            if finish.data.get("official_success") is not None or toolkit.solved():
                raise RuntimeError("integration check must not claim benchmark success")
            result = {
                "status": "passed",
                "definition": "direct Toolkit -> real VLA -> borrowed Env -> observations -> finish",
                "cli_run": False,
                "vlm_planner_assessed": False,
                "official_task_success": None,
                "execution": execution,
                "prediction_matches_submitted_prefix": True,
                "camera_images": 3,
                "finish": finish.data,
                "state_manifest": "toolkit/states.json",
            }
            write_json_atomic(self.output_dir / "policy-chain.json", result)
            return result
        finally:
            toolkit.close()

    def close(self) -> None:
        """Close only locally owned RPC connections; never stop borrowed services."""
        connections, self._connections = self._connections, []
        with ExitStack() as cleanup:
            for client in connections:
                cleanup.callback(client.close)
