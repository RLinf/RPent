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

"""Run the existing adaptive tools beside a simulator, across an RPC boundary."""

from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

from pydantic_core import to_jsonable_python

from robots.rl.contracts import EPISODE_CONTRACT_VERSION
from rpent.session import EnvState
from rpent.tools import ToolResult
from rpent.tools.toolkit import Toolkit
from rpent.utils.config import get_repo_root
from rpent.utils.logging import get_logger
from rpent.utils.rpc import RpcFacade, make_rpc_client

logger = get_logger("rl_remote")

REMOTE_OPTION_DEFAULTS = {
    "workflow": "episode",
    "world_cord": "base",
    "position_unit": "m",
    "reasoning": True,
    "observation_mode": "adaptive",
    "prompt": "simple",
    "image_size": 1024,
    "action_mode": "batch",
    "max_actions_per_call": 3,
    "reasoning_history": -1,
    "image_history": -1,
    "perception_mode": "single",
    "max_observe_calls": 8,
    "observe_concurrency": 8,
}


def remote_options(task: dict) -> dict:
    """Resolve and validate the workflow configuration at the RPC boundary."""
    from robots.rl.contracts import (
        validate_coordinates,
        validate_image_size,
        validate_options,
        validate_perception_options,
    )
    from robots.rl.images import IMAGE_SIZES
    from robots.rl.prompt_bundle import PROMPT_VARIANTS

    options = {
        name: task.get(name, default)
        for name, default in REMOTE_OPTION_DEFAULTS.items()
    }
    for name, default in REMOTE_OPTION_DEFAULTS.items():
        if type(options[name]) is not type(default):
            raise ValueError(f"invalid remote workflow option {name}")
    if options["workflow"] != "episode" or options["observation_mode"] != "adaptive":
        raise ValueError("remote toolkit supports adaptive episode only")
    if options["perception_mode"] != "single":
        raise ValueError("remote toolkit supports single perception only")
    if (
        options["image_size"] not in IMAGE_SIZES
        or options["prompt"] not in PROMPT_VARIANTS
    ):
        raise ValueError("invalid remote image size or prompt variant")
    validate_coordinates(task["environment"], options["world_cord"])
    from robots.rl.units import PositionUnits

    PositionUnits(options["position_unit"])
    validate_image_size(
        task["environment"], options["world_cord"], options["image_size"]
    )
    from robots.rl.prompt_bundle import validate_workflow

    validate_workflow(task["environment"], options["workflow"])
    validate_options(
        task["environment"],
        options["workflow"],
        options["action_mode"],
        options["max_actions_per_call"],
        options["reasoning_history"],
        options["image_history"],
        world_cord=options["world_cord"],
    )
    validate_perception_options(
        options["workflow"],
        options["perception_mode"],
        options["max_observe_calls"],
        options["observe_concurrency"],
    )
    return options


def source_digest() -> str:
    """Identify both ends of the control, tool and artifact protocol."""
    root = get_repo_root()
    digest = hashlib.sha256()
    for folder in (
        "robots/rl",
        "robots/libero",
        "robots/robocasa",
        "robots/robotwin",
        "robots/robodojo",
        "rpent/tools",
        "rpent/session",
        "rpent/utils/rpc",
    ):
        for path in sorted((root / folder).rglob("*")):
            if not path.is_file() or path.suffix not in {".py", ".md", ".yaml"}:
                continue
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def declarations(toolkit: Toolkit) -> list[dict[str, Any]]:
    """Describe the original tools without changing their model-facing schemas."""
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema,
            "readonly": tool.readonly,
        }
        for tool in toolkit.list_tools()
    ]


class ArtifactExchange:
    """Send changed original files with their byte hashes, without shared storage."""

    def __init__(self, output: Path) -> None:
        self.output = output
        self._sent: dict[str, tuple[int, int]] = {}

    def collect(self) -> list[dict[str, Any]]:
        """Collect files named by the state manifest, publishing the manifest last."""
        manifest_path = self.output / "states.json"
        manifest = json.loads(manifest_path.read_text())
        paths = [self.output / name for name in manifest["run_artifacts"]]
        for row in manifest["steps"]:
            for name in row["artifacts"]:
                paths.append(
                    self.output / name / f"{row['step_idx']:02d}{Path(name).suffix}"
                )
        result = []
        for path in [*sorted(set(paths)), manifest_path]:
            stat = path.stat()
            name = str(path.relative_to(self.output))
            signature = stat.st_mtime_ns, stat.st_size
            if self._sent.get(name) == signature:
                continue
            data = path.read_bytes()
            result.append(
                {
                    "path": name,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "bytes": base64.b64encode(data).decode("ascii"),
                }
            )
            self._sent[name] = signature
        return result

    @staticmethod
    def receive(output: Path, files: list[dict[str, Any]]) -> int:
        """Verify the entire delta before writing any file below the owned output."""
        verified = []
        seen = set()
        for item in files:
            relative = Path(item["path"])
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or not relative.parts
                or item["path"] in seen
            ):
                raise ValueError("invalid or duplicate remote artifact path")
            destination = output / relative
            if not destination.resolve().is_relative_to(output.resolve()):
                raise ValueError("remote artifact escapes its output directory")
            data = base64.b64decode(item["bytes"], validate=True)
            if hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise ValueError(f"remote artifact checksum mismatch: {relative}")
            verified.append((destination, data))
            seen.add(item["path"])
        for destination, data in verified:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".remote-tmp")
            temporary.write_bytes(data)
            temporary.replace(destination)
        return sum(len(data) for _, data in verified)


class ToolkitWorker(RpcFacade):
    """Own one ordinary RLToolkit and its local simulator daemons."""

    def __init__(self, toolkit: Toolkit, release: contextlib.ExitStack, config: dict):
        super().__init__()
        self.toolkit, self.release, self.config = toolkit, release, config
        self.exchange = ArtifactExchange(Path(config["output_dir"]))
        self._next_call = 0
        self._final_outcome: dict[str, Any] | None = None
        self._initial = toolkit.initial_observation()
        self._rpc.update(
            {
                "rl.describe": self.describe,
                "rl.observe": self.observe,
                "rl.execute": self.execute,
                "rl.close": self.finish,
            }
        )

    def describe(self) -> dict:
        """Return a small identity handshake before transferring native artifacts."""
        return {
            "task": self.config["task"],
            "episode_contract_version": EPISODE_CONTRACT_VERSION,
            "max_episode_steps": self.config["max_episode_steps"],
            "workflow_options": {
                name: (
                    getattr(self.toolkit, name)
                    if name not in {"observation_mode", "prompt"}
                    else self.config["task"].get(name, default)
                )
                for name, default in REMOTE_OPTION_DEFAULTS.items()
            },
            "source_digest": source_digest(),
            "tools": declarations(self.toolkit),
            "coordinates": self.toolkit._primitives.coordinates.metadata()
            if self.toolkit.world_cord == "general"
            or self.toolkit.environment == "libero"
            else None,
            "pid": os.getpid(),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        }

    def observe(self) -> dict:
        """Transfer the initial observation under an already renewable lease."""
        if self._final_outcome is not None:
            raise ValueError("session is closed")
        return self._pack(self._initial)

    def _outcome(self) -> dict[str, Any]:
        return {
            "solved": self.toolkit.solved(),
            "completion_result": self.toolkit.completion_result(),
            "infrastructure_error": getattr(self.toolkit, "infrastructure_error", None),
        }

    def _pack(self, result: ToolResult | None = None) -> dict:
        result = result or ToolResult()
        return {
            "data": result.data,
            "model_text": result.model_text,
            "images": [
                base64.b64encode(image).decode("ascii") for image in result.images
            ],
            **(
                self._final_outcome
                if self._final_outcome is not None
                else self._outcome()
            ),
            "workflow_stats": self.toolkit.workflow_stats,
            "files": self.exchange.collect(),
        }

    def execute(self, call_id: int, name: str, arguments: dict) -> dict:
        """Execute each numbered call at most once; uncertain calls are not replayed."""
        if (
            self._final_outcome is not None
            or type(call_id) is not int
            or call_id != self._next_call
        ):
            raise ValueError("closed session or out-of-order/replayed tool call")
        self._next_call += 1
        return self._pack(self.toolkit.execute_tool(name, arguments))

    def finish(self) -> dict:
        """Read the native verdict before shutdown and send the flushed artifacts."""
        if self._final_outcome is None:
            outcome = self._outcome()
            self.release.close()
            self._final_outcome = outcome
        return self._pack()

    def _builtin_dispatch(self, method: str, args: tuple, kwargs: dict) -> Any:
        if method == "rl.cancel":
            self.toolkit.cancel_active_and_wait()
            return {"cancelled": True}
        return super()._builtin_dispatch(method, args, kwargs)

    def close(self) -> None:
        """Release only the services owned by this episode."""
        self.release.close()


def cancel_remote_owners(endpoint: str, owner_ids: list[str]) -> dict:
    """Release explicit episode attempts, including ones still in backend.open."""
    rpc = make_rpc_client(endpoint)
    try:
        result = rpc.call(
            "backend.cancel_owners", kwargs={"owner_ids": owner_ids}, timeout_s=180
        )
    finally:
        rpc.close()
    if (
        result["owner_ids"] != list(dict.fromkeys(owner_ids))
        or result["remaining"]
        or result["cleanup_complete"] is not True
    ):
        raise RuntimeError(f"remote attempt cleanup was not verified: {result}")
    return result


class RemoteToolkit(Toolkit):
    """Keep the planner local while the original toolkit executes on its backend."""

    planner_builtin_tools = False
    api_environment_context = False

    def __init__(
        self,
        *,
        endpoint: str,
        request: dict,
        config,
        dashboard_events,
        memory,
        owner_id: str | None = None,
    ):
        state = EnvState(config.output_dir)
        super().__init__(state=state, dashboard_events=dashboard_events, memory=memory)
        self._rpc = make_rpc_client(endpoint)
        self._output = config.output_dir
        self.output_dir = Path(config.output_dir)
        options = remote_options(request["task"])
        configured = remote_options(
            {**request["task"], **getattr(config, "prompt_vars", {})}
        )
        if configured != options:
            raise ValueError(
                "remote request differs from the planner workflow configuration"
            )
        for name, value in options.items():
            setattr(self, name, value)
        from robots.rl.units import PositionUnits

        self.units = PositionUnits(self.position_unit)
        self.workflow_stats = {}
        self._call_lock = threading.Lock()
        self._call_id = 0
        self._stopped = threading.Event()
        self._session = None
        self._solved = False
        self.infrastructure_error = None
        self._completion = None
        self._transport_error = None
        self._heartbeat = None
        if owner_id is None:
            owner_id = os.environ.get("RPENT_REMOTE_OWNER_ID")
        request = {**request, "episode_contract_version": EPISODE_CONTRACT_VERSION}
        open_kwargs = {"request": request}
        if owner_id is not None:
            open_kwargs["owner_id"] = owner_id
        metadata = self._rpc.call("backend.open", kwargs=open_kwargs, timeout_s=1800)
        self._session = metadata["session"]
        try:
            if owner_id is not None and metadata.get("owner_id") != owner_id:
                raise ValueError("remote toolkit attempt owner does not match")
            if metadata.get("episode_contract_version") != EPISODE_CONTRACT_VERSION:
                raise ValueError(
                    "remote episode contract version differs from the planner"
                )
            if metadata["source_digest"] != source_digest():
                raise ValueError(
                    "remote toolkit source differs from the planner checkout"
                )
            if (
                metadata["task"] != request["task"]
                or metadata["max_episode_steps"] != request["max_episode_steps"]
                or metadata["workflow_options"] != options
            ):
                raise ValueError("remote toolkit task or budget mismatch")
            if self.world_cord == "general":
                coordinates = metadata.get("coordinates") or {}
                height = (
                    "floor_height_above_base_m"
                    if coordinates.get("support_surface") == "floor"
                    else "table_height_above_base_m"
                )
                if (
                    coordinates.get("world_cord") != "general"
                    or coordinates.get("tcp")
                    not in {"grip_site", "native_grasp_tcp", "calibrated_grasp_tcp"}
                    or coordinates.get("contract_version") != "episode_coordinates_v2"
                    or coordinates.get(height) is None
                ):
                    raise ValueError(
                        "remote general toolkit lacks matching base, support surface and TCP calibration"
                    )
            from robots.rl.toolkit import episode_tools

            for definition in episode_tools(
                self,
                request["task"]["environment"],
                self.world_cord,
                self.units,
                self.max_actions_per_call,
            ):
                self.add_tool(definition.with_handler(self._handler(definition.name)))
            if metadata["tools"] != declarations(self):
                raise ValueError("remote toolkit declarations differ from the planner")
            (self._output / "remote-runtime.json").write_text(
                json.dumps(
                    {
                        key: metadata[key]
                        for key in (
                            "session",
                            "pid",
                            "boot_id",
                            "source_digest",
                            "episode_contract_version",
                            "task",
                            "max_episode_steps",
                            "workflow_options",
                        )
                    },
                    indent=2,
                )
                + "\n"
            )
            self._heartbeat = threading.Thread(target=self._renew, daemon=True)
            self._heartbeat.start()
            self._initial = self._receive(self._invoke("rl.observe", timeout_s=1800))
        except BaseException:
            self._stopped.set()
            if self._heartbeat is not None:
                self._heartbeat.join(timeout=16)
            try:
                self._rpc.call(
                    "backend.release", kwargs={"session": self._session}, timeout_s=60
                )
            except Exception:
                logger.exception("Failed to release a rejected remote episode")
            raise

    def _register_common_tools(self) -> None:
        """Expose only the same clean-workflow declarations as the local toolkit."""

    def _invoke(self, method: str, *, timeout_s: float = 600, **kwargs) -> dict:
        return self._rpc.call(
            "backend.call",
            kwargs={"session": self._session, "method": method, "arguments": kwargs},
            timeout_s=timeout_s,
        )

    def _receive(self, payload: dict) -> ToolResult:
        started = time.perf_counter()
        transferred = ArtifactExchange.receive(self._output, payload["files"])
        self.state.reload()
        self._solved = payload["solved"]
        self.infrastructure_error = payload.get("infrastructure_error")
        self._completion = payload.get("completion_result")
        self.workflow_stats = payload["workflow_stats"]
        trace = self._output / "remote-transfer.jsonl"
        with trace.open("a") as file:
            file.write(
                json.dumps(
                    {
                        "call": self._call_id,
                        "artifact_bytes": transferred,
                        "received_at": time.time(),
                        "verify_write_seconds": time.perf_counter() - started,
                        "files": [
                            {"path": item["path"], "sha256": item["sha256"]}
                            for item in payload["files"]
                        ],
                    }
                )
                + "\n"
            )
        return ToolResult(
            data=payload["data"],
            model_text=payload.get("model_text"),
            images=[
                base64.b64decode(image, validate=True) for image in payload["images"]
            ],
        )

    def _handler(self, name: str):
        def execute(**kwargs):
            call_id = self._call_id
            self._call_id += 1
            try:
                return self._receive(
                    self._invoke(
                        "rl.execute",
                        call_id=call_id,
                        name=name,
                        arguments=to_jsonable_python(kwargs),
                    )
                )
            except Exception as exc:
                self._transport_error = exc
                raise

        return execute

    def execute_tool(self, name: str, input_dict: dict[str, Any]) -> ToolResult:
        with self._call_lock:
            if self._transport_error is not None:
                raise RuntimeError(
                    "remote tool outcome is uncertain; episode must stop"
                ) from self._transport_error
            result = super().execute_tool(name, input_dict)
            if self._transport_error is not None:
                raise RuntimeError(
                    "remote tool outcome is uncertain; episode must stop"
                ) from self._transport_error
            return result

    def initial_observation(self) -> ToolResult:
        return self._initial

    def get_env_state(
        self, *, command: dict, result: dict, elapsed_s: float
    ) -> ToolResult:
        """The backend has already captured the original post-action state."""
        return ToolResult(data=result)

    def completion_result(self) -> dict | None:
        """Return the native terminal state received from the owning worker."""
        return self._completion

    def solved(self) -> bool:
        return self._solved

    def _renew(self) -> None:
        while not self._stopped.wait(30):
            try:
                self._rpc.call(
                    "backend.renew", kwargs={"session": self._session}, timeout_s=15
                )
            except Exception:
                logger.exception("Remote episode lease renewal failed")

    def cancel_active_and_wait(self) -> None:
        try:
            if self._session is not None and not self._stopped.is_set():
                self._invoke("rl.cancel")
        finally:
            super().cancel_active_and_wait()

    def close(self) -> None:
        if self._stopped.is_set():
            return
        try:
            super().close()
            self._receive(self._invoke("rl.close"))
        finally:
            self._stopped.set()
            if self._heartbeat is not None:
                self._heartbeat.join(timeout=16)
            failing = sys.exception() is not None
            try:
                self._rpc.call(
                    "backend.release", kwargs={"session": self._session}, timeout_s=60
                )
            except Exception:
                if not failing:
                    raise
                logger.exception("Failed to release remote episode during cleanup")


def main() -> None:
    """Start one backend-owned toolkit worker from its validated configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--parent-watch", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.configuration.read_text())
    from robots.rl.rl_episode import start_environment
    from robots.rl.robot_spec import get_robot_spec

    robot_parser = argparse.ArgumentParser()
    get_robot_spec().add_cli_args(robot_parser, False)
    robot_args = robot_parser.parse_args([])
    robot_args.output_dir = Path(config["output_dir"])
    from rpent.utils.logging import init_output_dir

    init_output_dir(robot_args.output_dir)
    robot_args.remote_toolkit_endpoint = None
    robot_args.cuda_device = config["gpu"]
    robot_args.sam3_endpoint = config["sam3_endpoint"]
    robot_args.max_episode_steps = config["max_episode_steps"]
    robot_args.prompt = config["task"].get("prompt", "simple")
    robot_args.libero_type = config["task"].get("libero_type")
    release = contextlib.ExitStack()
    try:
        toolkit, _ = start_environment(robot_args, config["task"], release)
        ToolkitWorker(toolkit, release, config).serve(
            transport="http", host="127.0.0.1", port=0, parent_watch=args.parent_watch
        )
    finally:
        release.close()


if __name__ == "__main__":
    main()
