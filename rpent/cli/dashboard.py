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

"""CLI orchestration for one long-lived Dashboard Session."""

from __future__ import annotations

import argparse
import copy
import json
import shlex
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rpent.cli.main import _serialize_messages
from rpent.dashboard.events import RunStartedEvent
from rpent.memory import MemoryManager
from rpent.orchestration import (
    PlannerSessionContext,
    PlannerSessionRequest,
    PlannerSessionStopReason,
    PlannerSessionToolkit,
    PreparedPlannerSession,
    SynchronousPlannerSessionService,
    continuation_handoff_message,
)
from rpent.planner.base import PlannerResult, build_planner
from rpent.robots import get_toolkit
from rpent.utils.config import get_memory_dir
from rpent.utils.logging import get_logger, init_output_dir

if TYPE_CHECKING:
    from rpent.dashboard.state import ClaimedTask, DashboardState
    from rpent.robots.robot_spec import RobotSpec
    from rpent.utils.daemon import ProcessDaemon

logger = get_logger("agent")


class _DashboardPlannerSessionAdapter:
    """Bind Dashboard state and planner construction to the common service."""

    def __init__(
        self,
        *,
        args: argparse.Namespace,
        task_args: argparse.Namespace,
        robot_spec: RobotSpec,
        state: DashboardState,
        run_config: Any,
        output_dir: Path,
        primitives_kwargs: dict[str, Any],
        prompt_vars: dict[str, Any],
        recipe_tag: str,
    ) -> None:
        self._args = args
        self._task_args = task_args
        self._robot_spec = robot_spec
        self._state = state
        self._run_config = run_config
        self._output_dir = output_dir
        self._primitives_kwargs = primitives_kwargs
        self._prompt_vars = prompt_vars
        self._recipe_tag = recipe_tag

    def prepare_session(
        self,
        context: PlannerSessionContext,
        first_user_message: str,
    ) -> PreparedPlannerSession:
        session_message = first_user_message
        if context.session_number > 1:
            logger.info(
                "=== handing off to agent %d/%d ===",
                context.session_number,
                context.session_count,
            )
            session_message = continuation_handoff_message(
                self._output_dir,
                context.session_number,
                context.session_count,
                robot_name=self._args.robot_name,
            )
        system_prompt = self._robot_spec.prompts.render(
            "system",
            variables={
                **self._prompt_vars,
                "session_number": context.session_number,
                "session_max": context.session_count,
            },
        )
        return PreparedPlannerSession(
            context=context,
            system_prompt=system_prompt,
            user_message=session_message,
        )

    def create_toolkit(
        self,
        context: PlannerSessionContext,
    ) -> PlannerSessionToolkit:
        if context.exploration:
            self._state.begin_planner_session(
                video_path=context.state_output_dir / "episode.mp4",
            )
        if self._robot_spec.supports_exploration:
            return get_toolkit(
                self._args.robot_name,
                primitives_kwargs=self._primitives_kwargs,
                dashboard_events=self._state,
                config=self._run_config,
                mode="exploration" if self._task_args.explore else "evaluation",
                attempts_per_session=getattr(
                    self._task_args,
                    "explore_attempts_per_session",
                    0,
                ),
                state_output_dir=context.state_output_dir,
            )
        return get_toolkit(
            self._args.robot_name,
            primitives_kwargs=self._primitives_kwargs,
            dashboard_events=self._state,
            config=self._run_config,
        )

    def invoke_planner(
        self,
        prepared: PreparedPlannerSession,
        toolkit: PlannerSessionToolkit,
    ) -> PlannerResult:
        planner = build_planner(
            self._args.planner,
            output_dir=self._output_dir,
            recipe_tag=self._recipe_tag,
            robot_name=self._args.robot_name,
            base_url=self._args.base_url,
            model=self._args.model,
            max_tokens=self._args.max_tokens,
            planner_timeout_s=self._args.planner_timeout_s,
            reasoning_effort=self._args.reasoning_effort,
            claude_code_max_budget_usd=self._args.claude_code_max_budget_usd,
            dashboard_events=self._state,
            no_images=self._args.no_images,
        )
        return planner.solve(
            system_prompt=prepared.system_prompt,
            user_message=prepared.user_message,
            toolkit=toolkit,
            max_turns=self._args.max_turns,
            dashboard_interaction=self._state,
        )


def run_dashboard_session(
    args: argparse.Namespace,
    robot_spec: RobotSpec,
    *,
    parser: argparse.ArgumentParser,
) -> int:
    """Run one long-lived Dashboard Session with sequential fresh TaskRuns."""
    from rpent.dashboard.launcher import apply_to_args, defaults_from_args
    from rpent.dashboard.server import DashboardServer
    from rpent.dashboard.session import DashboardSessionController
    from rpent.dashboard.state import DashboardState
    from rpent.utils.config import get_repo_root

    dashboard_spec = robot_spec.dashboard
    if dashboard_spec is None:
        parser.error(f"robot {robot_spec.name!r} does not support Dashboard control")
    runtime_components = dashboard_spec["runtime_components"]
    shared_components = {
        component["name"]
        for component in runtime_components
        if component["scope"] == "shared"
    }
    unique_components = {
        component["name"]
        for component in runtime_components
        if component["scope"] == "unique"
    }

    dashboard_server = DashboardServer(
        host=args.dashboard_host,
        port=args.dashboard_port,
        language=args.dashboard_language,
        dashboard_spec=dashboard_spec,
    )
    dashboard_url = dashboard_server.start()
    print(
        f"Dashboard: {dashboard_url}. Open it, adjust the Session config, "
        "and click Start Session.",
        flush=True,
    )
    launch_config = dashboard_server.wait_for_launch(defaults=defaults_from_args(args))
    apply_to_args(args, launch_config)

    if args.env_endpoint is not None:
        parser.error(
            "Dashboard task control cannot use --env-endpoint because each "
            "TaskRun requires a fresh owned env_server"
        )

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H:%M:%S")
        session_root = get_repo_root() / "logs" / f"{timestamp}_dashboard_session"
    else:
        session_root = Path(args.output_dir)
    session_root = init_output_dir(session_root, verbose=args.verbose)
    logger.info("Dashboard: %s", dashboard_url)
    logger.info("launcher Session config applied: %s", launch_config)
    logger.info("physical agent cmd: %s", shlex.join([sys.executable, *sys.argv]))

    if (
        not getattr(args, "explore", False)
        and getattr(args, "memory_profile", "hf") == "hf"
    ):
        MemoryManager(get_memory_dir(robot_spec.name)).sync(
            remote_repo=robot_spec.memory_repo_id,
        )
    state = DashboardState(
        run_id=f"dashboard-session/{session_root.name}",
        output_dir=session_root,
        dashboard_spec=dashboard_spec,
    )
    dashboard_server.register(state)

    controller = DashboardSessionController(
        state=state,
        start_shared=lambda: robot_spec.init_runtime(
            args,
            session_root,
            state,
            shared_components,
        ),
        run_task=lambda claimed, shared: _run_dashboard_task(
            args=args,
            robot_spec=robot_spec,
            state=state,
            claimed=claimed,
            shared_primitives_kwargs=shared,
            unique_components=unique_components,
            session_root=session_root,
        ),
    )
    try:
        controller.run()
        if state.session_state == "fatal":
            logger.error(
                "Dashboard Session is fatal. Still serving at %s; "
                "press Ctrl+C to stop.",
                dashboard_url,
            )
            threading.Event().wait()
    except KeyboardInterrupt:
        state.request_shutdown()
    return 0


def _run_dashboard_task(
    *,
    args: argparse.Namespace,
    robot_spec: RobotSpec,
    state: DashboardState,
    claimed: ClaimedTask,
    shared_primitives_kwargs: dict[str, Any],
    unique_components: set[str],
    session_root: Path,
) -> str | None:
    """Execute one fresh Dashboard TaskRun against Session-owned services."""
    task_args = copy.copy(args)
    for name, value in claimed.request.items():
        setattr(task_args, name, value)
    task_args.output_dir = str(claimed.output_dir)
    run_config = robot_spec.parse_config(task_args)
    output_dir = init_output_dir(run_config.output_dir, verbose=args.verbose)

    recipe_tag = run_config.recipe_tag
    finish_result = None
    messages: list[dict] = []
    stats: dict = {}
    agent_error: str | None = None
    task_daemons: list[ProcessDaemon] = []
    recipe_path = ""
    started = time.time()
    solved = False
    memory_manager = None
    try:
        task_daemons, task_primitives_kwargs = robot_spec.init_runtime(
            task_args,
            output_dir,
            state,
            unique_components,
        )
        if not state.task_replacement_requested:
            primitives_kwargs = {
                **task_primitives_kwargs,
                **shared_primitives_kwargs,
            }
            prompt_vars = {**run_config.prompt_vars, "output_dir": output_dir}
            first_user_message = robot_spec.prompts.render(
                "user",
                variables=prompt_vars,
            )
            if not state.task_replacement_requested:
                state.emit(RunStartedEvent())
            adapter = _DashboardPlannerSessionAdapter(
                args=args,
                task_args=task_args,
                robot_spec=robot_spec,
                state=state,
                run_config=run_config,
                output_dir=output_dir,
                primitives_kwargs=primitives_kwargs,
                prompt_vars=prompt_vars,
                recipe_tag=recipe_tag,
            )
            check_solved = args.robot_name == "libero" or (
                task_args.explore and robot_spec.supports_exploration
            )
            session_service = SynchronousPlannerSessionService(
                prepare_session=adapter.prepare_session,
                create_toolkit=adapter.create_toolkit,
                invoke_planner=adapter.invoke_planner,
                probe_solved=(lambda toolkit: toolkit.solved())
                if check_solved
                else None,
                export_recipe=(lambda toolkit: toolkit.write_recipe(recipe_tag))
                if check_solved
                else None,
                cancellation_requested=lambda: state.task_replacement_requested,
                on_intermediate_timeout=lambda context, error: logger.warning(
                    "session %d/%d timed out; continuing with a fresh handoff",
                    context.session_number,
                    context.session_count,
                ),
                format_exception=str,
            )
            session_result = session_service.run(
                PlannerSessionRequest(
                    output_dir=output_dir,
                    exploration=getattr(task_args, "explore", False),
                    requested_session_count=getattr(
                        task_args,
                        "explore_sessions",
                        1,
                    ),
                    first_user_message=first_user_message,
                )
            )
            finish_result = session_result.finish_result
            messages = session_result.messages
            stats = session_result.stats
            agent_error = session_result.error
            solved = session_result.solved
            recipe_path = session_result.recipe_path
            memory_manager = session_result.memory_manager
            if session_result.stop_reason is PlannerSessionStopReason.EXCEPTION:
                logger.error(
                    "EXCEPTION in Dashboard TaskRun %04d: %s",
                    claimed.number,
                    agent_error,
                )
    except Exception as exc:
        logger.error("EXCEPTION in Dashboard TaskRun %04d: %s", claimed.number, exc)
        agent_error = str(exc)
    finally:
        cleanup_errors: list[str] = []
        if recipe_path:
            logger.info("recipe: %s", recipe_path)
        else:
            logger.info("recipe: not written (cell unsolved)")
        for daemon in reversed(task_daemons):
            try:
                daemon.stop()
            except Exception as exc:
                cleanup_errors.append(f"robot cleanup failed: {exc}")
        if cleanup_errors:
            cleanup_error = "; ".join(cleanup_errors)
            if agent_error is None:
                agent_error = cleanup_error
            else:
                logger.warning("%s", cleanup_error)

        transcript_path = output_dir / f"transcript_{run_config.recipe_tag}.json"
        record = {
            **run_config.task_desc,
            "model": args.model,
            "elapsed_s": round(time.time() - started, 1),
            "finish": finish_result,
            "stats": stats,
            "messages": _serialize_messages(messages),
        }
        try:
            with open(transcript_path, "a") as transcript_file:
                json.dump(record, transcript_file, indent=2, default=str)
        except Exception as exc:
            logger.warning(
                "failed to write TaskRun transcript %s: %s", transcript_path, exc
            )
        init_output_dir(session_root, verbose=args.verbose)

    if (
        getattr(task_args, "explore", False)
        and getattr(task_args, "auto_merge_memory", False)
        and not agent_error
        and not state.task_replacement_requested
        and memory_manager is not None
    ):
        try:
            merge_result = memory_manager.merge_memory(
                cell_tag=run_config.recipe_tag,
                run_state_dir=run_config.output_dir,
                solved=solved,
            )
            if merge_result:
                logger.info("run finalized: %s", merge_result)
        except Exception as exc:
            warning = f"memory finalization failed: {type(exc).__name__}: {exc}"
            logger.warning("%s", warning)
            state.report_task_warning(f"Task succeeded, but {warning}")

    return agent_error
