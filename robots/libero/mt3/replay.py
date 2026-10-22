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
"""Run one LIBERO cell with MT3 against the toolkit's live environment.

The memory directory holds ``mt3.json``: the execution settings
(:class:`~robots.libero.mt3.core.MT3Config`) and ``library``, the directory of
task libraries (``<library>/libero_<family>/task_XX.npz``, absolute or relative
to the memory directory). A perturbed suite reuses its base suite's library:
``_swap`` moves objects and fixtures, ``_lan`` only rewords the instruction.

Object poses are read from the simulator (``env.poses``), not perceived. Each
segment runs as one ``mt3_segment`` tool call, so every call ends with the
toolkit's usual state capture and success bookkeeping.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from robots.libero.mt3.core import (
    GRIP_SITE,
    EnvHooks,
    MT3Config,
    MT3Runner,
    TaskLibrary,
    pose_from_wxyz,
    reach_retrieval,
)
from rpent.utils.logging import get_output_dir

#: ``<family>[_<perturbation>]_t<task>_s<seed>``, the tag the CLI builds per cell.
_CELL = re.compile(r"^(10|goal|object|spatial)(?:_(swap|lan))?_t(\d+)_s(\d+)$")
CONFIG_NAME = "mt3.json"

_SEGMENT_SPEC = {
    "name": "mt3_segment",
    "description": "Align to and replay one segment of the retrieved demonstration.",
    "input_schema": {
        "type": "object",
        "properties": {"segment": {"type": "integer"}},
        "required": ["segment"],
    },
}
_HOLD_SPEC = {
    "name": "mt3_hold",
    "description": "Hold the last MT3 command while objects settle.",
    "input_schema": {"type": "object", "properties": {}},
}
_REPLAY_SPEC = {
    "name": "mt3_replay_actions",
    "description": "Send the retrieved demonstration's raw actions.",
    "input_schema": {"type": "object", "properties": {}},
}


def parse_cell(cell_tag: str) -> tuple[str, int, int]:
    """Base suite, task and seed of a cell tag MT3 can run."""
    match = _CELL.match(cell_tag)
    if not match:
        raise ValueError(
            f"MT3 cannot run cell {cell_tag!r}: it needs a base, _swap or _lan "
            "suite, whose task i is the base suite's task i"
        )
    family, _, task, seed = match.groups()
    return f"libero_{family}", int(task), int(seed)


def load_config(memory_root: Path) -> tuple[MT3Config, Path]:
    """Settings and library directory named by ``<memory_root>/mt3.json``."""
    data = json.loads((memory_root / CONFIG_NAME).read_text())
    library = Path(data["library"])
    if not library.is_absolute():
        library = memory_root / library
    return MT3Config.from_json(data), library


def _hooks(toolkit: Any) -> EnvHooks:
    primitives = toolkit.primitives
    env = primitives.env

    def body_poses(bodies: list[str]) -> dict[str, np.ndarray]:
        return {k: pose_from_wxyz(v) for k, v in env.poses(bodies).items()}

    return EnvHooks(
        step=primitives._step_env,
        eef_pose=lambda: pose_from_wxyz(env.poses([], [GRIP_SITE])[GRIP_SITE]),
        body_poses=body_poses,
        done=lambda: bool(env.terminated or env.truncated),
        check_paths=lambda paths, options=None: env.check_paths(paths, options),
        plan_path=lambda goal: env.plan_path(goal),
    )


def execute(toolkit: Any, name: str, arguments: dict[str, Any]) -> None:
    """Execute one registered MT3 tool; raise on its error result.

    The tool's own diagnostics stay on the runner; the toolkit returns the
    state it captured after the call.
    """
    result = toolkit.execute_tool(name, arguments).result
    if error := result.get("error"):
        raise RuntimeError(f"{name} failed: {error}\n{result.get('traceback', '')}")


def select_demo(lib: TaskLibrary, scene: dict[str, np.ndarray], cfg: MT3Config):
    """Select the configured distance rank; rank zero is the usual nearest demo."""
    demo, retrieval = lib.nearest(scene, cfg)
    rank = int(cfg.retrieval_rank)
    if rank:
        ranked = lib.ranked(scene, cfg)
        if rank < 0 or rank >= len(ranked):
            raise ValueError(
                f"retrieval_rank {rank} is outside the {len(ranked)} kept demonstrations"
            )
        demo = ranked[rank]
        retrieval = {**retrieval, "demo": demo, "rank": rank}
    return demo, retrieval


def run_mt3(
    toolkit: Any, cell_tag: str, note: Callable[[str], None] = lambda _: None
) -> dict:
    """The ``RobotSpec`` hook behind ``--planner mt3``: one episode, no retries."""
    # Avoid numba's source cache locator failure when LIBERO is launched from
    # worker processes. The simulator does not depend on a persistent cache.
    try:
        import robosuite.macros as robosuite_macros

        robosuite_macros.CACHE_NUMBA = False
    except ImportError:
        pass
    suite, task, seed = parse_cell(cell_tag)
    cfg, library_dir = load_config(Path(toolkit.memory.root))
    if cfg.force_recorded_trace:
        trace = np.load(cfg.force_recorded_trace)
        (Path(get_output_dir()) / "exact_trace_progress.json").write_text(
            json.dumps({"phase": "loaded", "actions": int(len(trace["actions"]))})
        )
        def replay_trace() -> dict[str, Any]:
            executed = 0
            for action in trace["actions"]:
                toolkit.primitives._step_env(np.asarray(action, dtype=np.float64))
                executed += 1
                if toolkit.primitives.env.terminated or toolkit.primitives.env.truncated:
                    break
            return {"steps": executed, "success": bool(toolkit.primitives.env.terminated)}

        toolkit.add_tool(
            "mt3_exact_trace",
            {"name": "mt3_exact_trace", "description": "Replay the recorded raw action trace verbatim.",
             "input_schema": {"type": "object", "properties": {}}},
            partial(toolkit._execute_primitive, "mt3_exact_trace", replay_trace),
        )
        (Path(get_output_dir()) / "exact_trace_progress.json").write_text(
            json.dumps({"phase": "tool_registered", "actions": int(len(trace["actions"]))})
        )
        execute(toolkit, "mt3_exact_trace", {})
        episode = {
            "cell": cell_tag, "suite": suite, "task": task, "seed": seed,
            "mode": "exact_recorded_trace", "success": bool(toolkit.solved()),
            "actions": int(len(trace["actions"])),
            "terminated": bool(toolkit.primitives.env.terminated),
            "truncated": bool(toolkit.primitives.env.truncated),
        }
        (Path(get_output_dir()) / "mt3_episode.json").write_text(json.dumps(episode, indent=1))
        return {"done": episode["success"], "plan": 1, "anchors": 0, "program": "mt3/exact_recorded_trace"}
    lib = TaskLibrary(library_dir / suite / f"task_{task:02d}.npz")
    hooks = _hooks(toolkit)
    trace: dict[str, list] = {"states": [], "actions": []}
    if cfg.record_trace:
        env = toolkit.primitives.env
        step = hooks.step

        def recorded_step(action: np.ndarray) -> None:
            trace["states"].append(env.sim_state())
            trace["actions"].append(np.asarray(action, dtype=np.float64))
            step(action)

        hooks.step = recorded_step
    # Fixture roots are not contained in the saved qpos init-state vector.
    # Preserve their actual placement when evaluating native RPent seeding.
    (Path(get_output_dir()) / "initial_body_poses.json").write_text(
        json.dumps(toolkit.primitives.env.poses(), indent=1)
    )
    scene = hooks.body_poses(lib.ref_bodies)
    demo, retrieval = select_demo(lib, scene, cfg)
    if cfg.retrieval == "reach" and hooks.check_paths is not None:
        demo, reach = reach_retrieval(lib, scene, cfg, hooks)
        retrieval = {**retrieval, "demo": demo, "reach": reach}
    runner = MT3Runner(lib, demo, cfg, hooks)
    note(
        f"MT3 {cfg.mode}: {suite} task {task}, demo {demo} of {len(lib)} "
        f"(scene distance {retrieval['dist']:.3f}), "
        f"{len(runner.segments)} segments"
    )

    def segment(segment: int) -> dict[str, Any]:
        return runner.run_segment(int(segment))

    for spec, handler in (
        (_SEGMENT_SPEC, segment),
        (_HOLD_SPEC, runner.hold),
        (_REPLAY_SPEC, runner.replay_actions),
    ):
        toolkit.add_tool(
            spec["name"],
            spec,
            partial(toolkit._execute_primitive, spec["name"], handler),
        )

    calls = 0
    if cfg.mode == "replay":
        execute(toolkit, "mt3_replay_actions", {})
        calls += 1
    else:
        for k in range(len(runner.segments)):
            if hooks.done():
                break
            execute(toolkit, "mt3_segment", {"segment": k})
            calls += 1
            note(
                f"  segment {k} ({runner.logs[-1]['ref']}): {runner.logs[-1]['status']}"
            )
            if runner.logs[-1].get("blocked"):
                break
        if not hooks.done() and not any(c.get("blocked") for c in runner.logs):
            execute(toolkit, "mt3_hold", {})
            calls += 1

    episode = {
        "cell": cell_tag,
        "suite": suite,
        "task": task,
        "seed": seed,
        "mode": cfg.mode,
        "fixture_seed_policy": os.environ.get(
            "RPENT_MT3_FIXTURE_SEED_POLICY", "deterministic"
        ),
        "config": cfg.__dict__,
        "library": str(lib.path),
        "retrieval": retrieval,
        "scene": {
            body: [round(float(v), 5) for v in pose[:3, 3]]
            + [round(float(np.arctan2(pose[1, 0], pose[0, 0])), 5)]
            for body, pose in scene.items()
        },
        "calls": runner.logs,
        "steps": runner.steps,
        "success": bool(toolkit.solved()),
        "terminated": bool(toolkit.primitives.env.terminated),
        "truncated": bool(toolkit.primitives.env.truncated),
    }
    if cfg.record_trace and trace["actions"]:
        np.savez_compressed(
            Path(get_output_dir()) / "mt3_trace.npz",
            states=np.asarray(trace["states"]),
            actions=np.asarray(trace["actions"]),
            success=bool(toolkit.solved()),
        )
    (Path(get_output_dir()) / "mt3_episode.json").write_text(
        json.dumps(episode, indent=1)
    )
    return {
        "done": episode["success"],
        "plan": calls,
        "anchors": len(scene),
        "program": f"mt3/{cfg.mode}/demo{demo}",
    }
