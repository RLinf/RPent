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

from rpent.cli.main import (
    _handoff_message,
    _serialize_messages,
)
from rpent.dashboard.events import RunStartedEvent
from rpent.planner.base import build_planner
from rpent.robots import get_toolkit
from rpent.utils.logging import get_logger, init_output_dir
from rpent.utils.resources import ensure_resources

if TYPE_CHECKING:
    from rpent.dashboard.state import ClaimedTask, DashboardState
    from rpent.robots.robot_spec import RobotSpec
    from rpent.utils.daemon import ProcessDaemon

logger = get_logger("agent")


def _resolve_dashboard_class(path: str) -> type[Any]:
    """Resolve ``module:ClassName`` dashboard class paths from robot specs."""

    module_name, separator, class_name = path.partition(":")
    if not separator or not module_name or not class_name:
        raise ValueError(f"invalid dashboard class path: {path!r}")
    import importlib

    module = importlib.import_module(module_name)
    dashboard_class = getattr(module, class_name)
    if not isinstance(dashboard_class, type):
        raise TypeError(f"dashboard class path did not resolve to a class: {path!r}")
    return dashboard_class


def _dashboard_server_and_state_classes(
    robot_spec: RobotSpec,
    dashboard_spec: dict[str, Any],
) -> tuple[type[Any], type[Any]]:
    """Return the Dashboard classes selected by the robot dashboard spec."""

    from rpent.dashboard.server import DashboardServer
    from rpent.dashboard.state import DashboardState

    classes = dashboard_spec.get("classes")
    if classes is not None:
        if not isinstance(classes, dict):
            raise TypeError(f"robot {robot_spec.name!r} dashboard classes must be a dict")
        server_path = classes.get("server")
        state_path = classes.get("state")
        if not isinstance(server_path, str) or not isinstance(state_path, str):
            raise TypeError(
                f"robot {robot_spec.name!r} dashboard classes require server/state paths"
            )
        return _resolve_dashboard_class(server_path), _resolve_dashboard_class(state_path)
    return DashboardServer, DashboardState


def run_dashboard_session(
    args: argparse.Namespace,
    robot_spec: RobotSpec,
    *,
    parser: argparse.ArgumentParser,
) -> int:
    """Run one long-lived Dashboard Session with sequential fresh TaskRuns."""
    from rpent.dashboard.launcher import apply_to_args, defaults_from_args
    from rpent.dashboard.session import DashboardSessionController
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

    dashboard_server_cls, dashboard_state_cls = _dashboard_server_and_state_classes(
        robot_spec,
        dashboard_spec,
    )

    dashboard_server = dashboard_server_cls(
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
        ensure_resources(robot_spec)
    state = dashboard_state_cls(
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
        _bind_behavior_dashboard_backend(state, task_primitives_kwargs)
        if not state.task_replacement_requested:
            primitives_kwargs = {
                **task_primitives_kwargs,
                **shared_primitives_kwargs,
            }
            prompt_vars = {**run_config.prompt_vars, "output_dir": output_dir}
            session_message = robot_spec.prompts.render("user", variables=prompt_vars)
            sessions = max(
                1,
                int(getattr(task_args, "explore_sessions", 1) or 1),
            )
            if not getattr(task_args, "explore", False):
                sessions = 1
            if not state.task_replacement_requested:
                state.emit(RunStartedEvent())
            for session_number in range(1, sessions + 1):
                if state.task_replacement_requested:
                    break
                if session_number > 1:
                    logger.info(
                        "=== handing off to agent %d/%d ===",
                        session_number,
                        sessions,
                    )
                    session_message = _handoff_message(
                        output_dir,
                        session_number,
                        sessions,
                    )
                system_prompt = robot_spec.prompts.render(
                    "system",
                    variables={
                        **prompt_vars,
                        "session_number": session_number,
                        "session_max": sessions,
                    },
                )
                state_output_dir = output_dir
                if getattr(task_args, "explore", False):
                    state_output_dir = (
                        output_dir / "sessions" / f"session_{session_number:03d}"
                    )
                    state.begin_planner_session(
                        video_path=state_output_dir / "episode.mp4",
                    )
                if args.robot_name == "libero":
                    toolkit = get_toolkit(
                        args.robot_name,
                        primitives_kwargs=primitives_kwargs,
                        dashboard_events=state,
                        config=run_config,
                        mode="exploration" if task_args.explore else "evaluation",
                        attempts_per_session=getattr(
                            task_args, "explore_attempts_per_session", 0
                        ),
                        state_output_dir=state_output_dir,
                    )
                else:
                    toolkit = get_toolkit(
                        args.robot_name,
                        primitives_kwargs=primitives_kwargs,
                        dashboard_events=state,
                        config=run_config,
                    )
                memory_manager = toolkit.memory
                try:
                    planner = build_planner(
                        args.planner,
                        output_dir=output_dir,
                        recipe_tag=recipe_tag,
                        robot_name=args.robot_name,
                        base_url=args.base_url,
                        model=args.model,
                        max_tokens=args.max_tokens,
                        planner_timeout_s=args.planner_timeout_s,
                        reasoning_effort=args.reasoning_effort,
                        claude_code_max_budget_usd=args.claude_code_max_budget_usd,
                        dashboard_events=state,
                        no_images=args.no_images,
                    )
                    result = planner.solve(
                        system_prompt=system_prompt,
                        user_message=session_message,
                        toolkit=toolkit,
                        max_turns=args.max_turns,
                        dashboard_interaction=state,
                    )
                    finish_result = result.finish_result
                    messages += result.messages
                    stats = result.stats
                    agent_error = result.error
                    if args.robot_name == "libero":
                        solved = toolkit.solved()
                        if solved:
                            recipe_path = toolkit.write_recipe(recipe_tag)
                finally:
                    toolkit.close()
                if solved or state.task_replacement_requested:
                    break
                if agent_error:
                    if session_number < sessions and "timed out" in agent_error.lower():
                        logger.warning(
                            "session %d/%d timed out; continuing with a fresh handoff",
                            session_number,
                            sessions,
                        )
                        continue
                    break
    except Exception as exc:
        logger.error("EXCEPTION in Dashboard TaskRun %04d: %s", claimed.number, exc)
        agent_error = str(exc)
    finally:
        cleanup_errors: list[str] = []
        if recipe_path:
            logger.info("recipe: %s", recipe_path)
        else:
            logger.info("recipe: not written (cell unsolved)")
        _unbind_behavior_dashboard_backend(state)
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


def _bind_behavior_dashboard_backend(
    state: DashboardState,
    primitives_kwargs: dict[str, Any],
) -> None:
    """Bind BEHAVIOR's env client to its optional Dashboard control routes."""

    backend = primitives_kwargs.get("env")
    if backend is None or not hasattr(state, "control_controller"):
        return
    controller = state.control_controller()
    if controller is None:
        try:
            from robots.behavior.dashboard import BehaviorControlController
        except Exception:
            return
        bind_controller = getattr(state, "bind_controller", None)
        if not callable(bind_controller):
            return
        controller = BehaviorControlController(state=state, backend=backend)
        bind_controller(controller)
        return
    bind_backend = getattr(controller, "bind_backend", None)
    if callable(bind_backend):
        bind_backend(backend)


def _unbind_behavior_dashboard_backend(state: DashboardState) -> None:
    controller_getter = getattr(state, "control_controller", None)
    if callable(controller_getter):
        controller = controller_getter()
        unbind_backend = getattr(controller, "unbind_backend", None)
        if callable(unbind_backend):
            unbind_backend()
    unbind = getattr(state, "unbind_controller", None)
    if callable(unbind):
        unbind()
