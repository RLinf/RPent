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

"""Expose recorded interaction segments to an ordinary RPent planner."""

import json
from functools import partial
from pathlib import Path

import numpy as np

from robots.libero.mt3.core import MT3Runner, TaskLibrary
from robots.libero.mt3.replay import _hooks, load_config, parse_cell
from rpent.tools.toolkit import readonly
from rpent.utils.logging import get_output_dir


def register_agent_tools(toolkit, cell_tag: str, config_path: str) -> None:
    """Register catalog and guarded segment replay beside the ordinary tools."""
    cfg, library_dir = load_config(Path(config_path).parent)
    if Path(config_path).name != "mt3.json":
        raise ValueError("replay config must be named mt3.json")
    if cfg.mode != "mt3" or cfg.alignment_guard_pos <= 0:
        raise ValueError("agent replay requires guarded mt3 pose tracking")
    suite, task, _ = parse_cell(cell_tag)
    lib = TaskLibrary(library_dir / suite / f"task_{task:02d}.npz")
    hooks = _hooks(toolkit)
    runners = {}
    calls = []

    @readonly
    def trajectory_catalog():
        demo, retrieval = lib.nearest(hooks.body_poses(lib.ref_bodies), cfg)
        return {
            "nearest_demo": demo,
            "retrieval": retrieval,
            "available_demos": np.flatnonzero(lib.keep).tolist(),
            "segments": [
                {"segment": k, "reference": s["ref"], "events": s["events"]}
                for k, s in enumerate(lib.segments[demo])
            ],
            "notice": "Replay internally uses simulator object poses. Inspect current images after every segment. Alignment failure stops before the source interaction.",
        }

    def trajectory_replay(segment: int, demo: int | None = None):
        if hooks.done():
            return {
                "terminated": bool(toolkit.primitives.env.terminated),
                "truncated": bool(toolkit.primitives.env.truncated),
            }
        if demo is None:
            demo, _ = lib.nearest(hooks.body_poses(lib.ref_bodies), cfg)
        if (
            not isinstance(demo, int)
            or demo < 0
            or demo >= len(lib)
            or not lib.keep[demo]
        ):
            raise ValueError("demo must be a kept demo from trajectory_catalog")
        if (
            not isinstance(segment, int)
            or segment < 0
            or segment >= len(lib.segments[demo])
        ):
            raise ValueError("segment is outside the selected demonstration")
        runner = runners.setdefault(demo, MT3Runner(lib, demo, cfg, hooks))
        result = runner.run_segment(segment)
        calls.append({"demo": demo, **result})
        (Path(get_output_dir()) / "trajectory_calls.json").write_text(
            json.dumps(calls, indent=2)
        )
        return {"demo": demo, **result}

    toolkit.add_tool(
        "trajectory_catalog",
        {
            "name": "trajectory_catalog",
            "description": "List a nearest task demonstration and its object-relative interaction segments without moving.",
            "input_schema": {"type": "object", "properties": {}},
        },
        trajectory_catalog,
    )
    toolkit.add_tool(
        "trajectory_replay",
        {
            "name": "trajectory_replay",
            "description": "Align and replay one demonstrated interaction. Stops before contact if alignment remains blocked. Inspect the returned images and residuals, then recover with ordinary robot tools when needed. Never resets.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "segment": {"type": "integer"},
                    "demo": {"type": "integer"},
                },
                "required": ["segment"],
            },
        },
        partial(toolkit._execute_primitive, "trajectory_replay", trajectory_replay),
    )
