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

"""RPent planner hosted by the official XPolicyLab policy server."""

from __future__ import annotations

import re
import time
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from queue import Empty
from threading import Thread
from typing import Any

from robots.robodojo.env_client import RoboDojoEnvClient
from robots.robodojo.primitives import RoboDojoPrimitives
from robots.robodojo.robot_spec import get_robot_spec
from robots.robodojo.toolkit import RoboDojoToolkit
from robots.robodojo.xpolicylab.bridge import Bridge, PublicPrimitives
from rpent.dashboard.events import NullDashboardEventSink
from rpent.evaluation import write_json_atomic
from rpent.memory import MemoryManager
from rpent.planner.base import build_planner
from rpent.utils.config import get_repo_root
from rpent.utils.logging import get_logger, init_output_dir
from rpent.utils.rpc import make_rpc_client

logger = get_logger(__name__)


class ExternalToolkit(RoboDojoToolkit):
    """Use public controller feedback while retaining RPent's existing tools."""

    def init_primitives(self, *, primitives_kwargs: dict[str, Any]) -> None:
        self._pending_perception = []
        self._state.reset()
        self._primitives = PublicPrimitives(
            **primitives_kwargs,
            check_cancelled=self.raise_if_cancelled,
        )

    def _register_robodojo_tools(self, vla_backend: str) -> None:
        super()._register_robodojo_tools(vla_backend)
        declaration = RoboDojoPrimitives.set_gripper.with_handler(
            self._primitives.set_gripper
        )
        self.add_tool(
            replace(
                declaration,
                description=(
                    "Move one gripper toward val (0 closed, 1 open) over steps native action targets. "
                    "Starts from the last requested/controller target, not measured opening. "
                    "gripper_target is the requested target; verify grasp through images. "
                    "After EE control, arm targets begin at measured joints."
                ),
            )
        )


class Model:
    """Single-environment XPolicyLab model; each episode gets a fresh planner."""

    def __init__(self, model_cfg: dict[str, Any]) -> None:
        if model_cfg.get("env_cfg_type") != "arx_x5":
            raise ValueError("RPent's XPolicyLab adapter requires dual ARX X5")
        self.cfg = model_cfg
        self.output_root = Path(model_cfg["output_dir"]).resolve()
        self.worker: Thread | None = None
        self.bridge: Bridge | None = None
        self.toolkit: ExternalToolkit | None = None
        self.context: dict | None = None
        self.report = {"status": "waiting"}

    def reset(self) -> None:
        """Discard per-episode observations, targets, action queues and history."""
        planner = None
        if self.bridge is not None:
            self.abort()
            planner = self.bridge.planner
        if self.worker is not None:
            self.worker.join(timeout=130)
            if self.worker.is_alive():
                raise RuntimeError(
                    "Previous planner did not stop; refusing to start another episode"
                )
        self.bridge = Bridge(planner=planner)
        self.worker = None
        self.toolkit = None
        self.context = None
        self.report = {"status": "waiting"}

    def update_obs(self, obs: dict) -> None:
        """Acknowledge an observation; the first observation starts the planner."""
        if self.bridge is None:
            raise RuntimeError("reset must precede observations")
        receipt = obs["episode_status"]
        context = obs["rpent_episode"]
        if self.context is not None and context != self.context:
            raise ValueError("Episode identity changed without reset")
        self.bridge.observe(
            obs, receipt["step"], receipt["step_limit"], receipt["ended"]
        )
        if self.worker is None:
            if receipt["step"] != 0:
                raise ValueError("A new planner must attach at action step zero")
            if not re.fullmatch(r"[A-Za-z0-9_]+", context["task"]):
                raise ValueError("Invalid task identifier")
            for field in ("seed", "layout"):
                if type(context[field]) is not int or context[field] < 0:
                    raise ValueError(f"Invalid {field}")
            self.context = dict(context)
            self.tag = f"{context['task']}_s{context['seed']}_l{context['layout']}"
            self.output = self.output_root / "episodes" / self.tag
            self.output.mkdir(parents=True, exist_ok=False)
            self.report = {
                "status": "running",
                "episode": self.context,
                "output_dir": str(self.output),
            }
            self.worker = Thread(target=self._run_episode, daemon=True)
            self.worker.start()

    def get_action(self) -> list[dict]:
        """Return one queued action, or no action while the planner is thinking."""
        if self.bridge is None:
            return []
        try:
            action = self.bridge.actions.get(timeout=1)
        except Empty:
            return []
        return [] if action is None else [action]

    def episode_report(self) -> dict:
        """Return execution status to the evaluator, without advancing simulation."""
        return {**deepcopy(self.report), "timeout_s": self.cfg.get("timeout", 4800)}

    def abort(self) -> None:
        """Wake pending action waits and cancel current toolkit execution."""
        if self.bridge is not None:
            self.bridge.stop()
        if self.toolkit is not None:
            self.toolkit.cancel_active_and_wait()

    def _run_episode(self) -> None:
        started = time.monotonic()
        env = None
        report = dict(self.report)
        try:
            init_output_dir(self.output)
            write_json_atomic(self.output / "episode.json", self.context)
            env = RoboDojoEnvClient(self.bridge, expected_meta={"mode": "xpolicylab"})
            from robots.robodojo.vpp2_client import VPP2Client

            kwargs = {
                "env": env,
                "vla_client": VPP2Client(make_rpc_client(self.cfg["am_endpoint"])),
            }
            if endpoint := self.cfg.get("sam3_endpoint"):
                from rpent.robots.components.sam3_client import Sam3Client

                kwargs["sam3_client"] = Sam3Client(make_rpc_client(endpoint))
            memory = Path(self.cfg["memory_dir"])
            variables = dict(
                **self.context,
                env_cfg_type="arx_x5",
                recipe_tag=self.tag,
                reference_tag=f"{self.context['task']}_s0_l0",
                task_summary={"task": self.context["task"]},
                memory_dir=str(memory),
                output_dir=str(self.output),
                mode="dev",
                vla_backend="vpp2",
            )
            bundle = get_robot_spec().prompts
            system = bundle.render("system", variables=variables)
            guide = get_repo_root() / "robots/robodojo/guides/GUIDE_RPENT.md"
            system = system.replace("robots/robodojo/guides/GUIDE_RPENT.md", str(guide))
            user = bundle.render("user", variables=variables)
            user += "\n\nExecution interface: XPolicyLab."
            write_json_atomic(
                self.output / "task-instruction.json",
                {
                    "instruction": self.bridge.obs["instruction"],
                    "source": "observation.instruction",
                },
            )
            (self.output / "system_prompt.txt").write_text(system)
            (self.output / "user_prompt.txt").write_text(user)
            events = NullDashboardEventSink()
            self.toolkit = ExternalToolkit(
                primitives_kwargs=kwargs,
                dashboard_events=events,
                memory=MemoryManager(memory, memory_access="read_only"),
                state_output_dir=self.output,
            )
            planner = build_planner(
                self.cfg.get("planner", "codex"),
                output_dir=self.output,
                recipe_tag=self.tag,
                robot_name="robodojo",
                memory_dir=memory,
                model=self.cfg.get("model"),
                reasoning_effort=self.cfg.get("reasoning_effort", "high"),
                planner_timeout_s=self.cfg.get("timeout", 4800),
                dashboard_events=events,
            )
            result = planner.solve(
                system_prompt=system,
                user_message=user,
                toolkit=self.toolkit,
                max_turns=10000,
            )
            report = dict(
                report,
                status="failed"
                if result.error or result.finish_result is None
                else "completed",
                finish=result.finish_result,
                error=result.error,
                stats=result.stats,
                native_steps=self.bridge.status["step"],
                history_steps=env._history_step,
                inference_count=env._inference_index,
                elapsed_s=time.monotonic() - started,
            )
        except Exception as exc:
            logger.exception("XPolicyLab planner episode failed")
            report = dict(report, status="failed", error=f"{type(exc).__name__}: {exc}")
        finally:
            try:
                if self.toolkit is not None:
                    self.toolkit.close()
                if env is not None:
                    env.close()
            except Exception as exc:
                report = dict(report, status="failed", cleanup_error=str(exc))
            try:
                write_json_atomic(self.output / "report.json", report)
            except Exception as exc:
                report = dict(
                    report,
                    status="failed",
                    error=f"Could not save episode report: {exc}",
                )
                logger.exception("Failed to save episode report")
            self.report = report
            self.bridge.actions.put(None)
