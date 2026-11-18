# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.
"""Compare rendered prompt characters for two source checkouts, without robots.

Run with the project's test environment:
    python experiments/prompt_character_counts/count_prompts.py \
        --before /path/to/baseline --after /path/to/change

Counts are Unicode code points, including whitespace, for system + first user
messages. Tool schemas, images, later turns and planner wrappers are excluded.
Both revisions use identical synthetic fixtures and actual registered task text
for physical robots. The temporary memory root is normalized to /memory before
counting, so counts do not depend on the machine's temporary directory.
"""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def render(root: Path, memory: Path) -> list[dict]:
    """Render each supported prompt selection with fixed task variables."""
    sys.path.insert(0, str(root.resolve()))
    from robots.dual_franka.tasks import DUAL_FRANKA_TASKS
    from robots.franka.tasks import FRANKA_TASKS
    from robots.yam.tasks import TASK_INSTRUCTIONS
    from rpent.prompt.utils import format_prompt

    common = {
        "seed": 0,
        "recipe_tag": "sample_s0",
        "reference_tag": "sample_s0",
        "memory_dir": str(memory),
        "memory_inbox": str(memory / "_internal/inbox/sample_s0"),
        "output_dir": "/output/sample_s0",
        "session_number": 1,
        "session_max": 3,
        "explore_attempts_per_session": 3,
        "memory_profile": "hf",
        "enable_direct_action": False,
    }
    cases = []

    def add(robot: str, label: str, **values: object) -> None:
        cases.append((robot, label, common | values))

    for mode in ("eval", "explore"):
        for profile in ("hf", "local"):
            for direct in (False, True):
                add(
                    "libero",
                    f"pi05/{mode}/{profile}/direct={direct}",
                    mode=mode,
                    suite="libero_10_task",
                    task=0,
                    memory_profile=profile,
                    enable_direct_action=direct,
                    policy_kind="vla",
                    policy_backend="pi05",
                )
        add(
            "robocasa",
            f"rldx/{mode}",
            mode=mode,
            task_name="OpenDrawer",
            split="target",
        )
        for backend in ("lingbot", "uniwam"):
            add(
                "robotwin",
                f"{backend}/{mode}",
                mode=mode,
                task_name="beat_block_hammer",
                task_config="hd_clean",
                policy_backend=backend,
                policy_kind="wam" if backend == "uniwam" else "vla",
            )
        for backend in ("vpp2", "awomo05"):
            add(
                "robodojo",
                f"{backend}/{mode}",
                mode="dev" if mode == "eval" else mode,
                task="stack_blocks",
                task_summary="Stack the blocks.",
                layout=0,
                env_cfg_type="arx_x5",
                policy_backend=backend,
                vla_backend=backend,
            )
        task = DUAL_FRANKA_TASKS[1]
        add(
            "dual_franka",
            mode,
            mode=mode,
            task_id=1,
            task_name=task.name,
            instruction=task.instruction,
            setup=task.setup,
            success_criteria=task.success_criteria,
            constraints="\n".join(task.constraints),
        )
        for backend in ("primitives-only", "pi05"):
            add(
                "yam",
                f"{backend}/{mode}",
                mode=mode,
                task_id=1,
                instruction=TASK_INSTRUCTIONS[1],
                policy_backend=backend,
                vla_enabled=backend == "pi05",
            )
    for robot, backends in (
        ("libero", ("cosmos_policy", "cosmos3", "fastwam")),
        ("robotwin", ("cosmos-policy", "cosmos3", "fast-wam")),
    ):
        for backend in backends:
            add(
                robot,
                f"{backend}/eval",
                mode="eval",
                policy_kind="wam",
                policy_backend=backend,
                suite="libero_10_task",
                task=0,
                task_name="beat_block_hammer",
                task_config="hd_clean",
            )
    for backend in ("vpp2", "awomo05"):
        add(
            "robodojo",
            f"{backend}/eval-fair",
            mode="eval-fair",
            policy_backend=backend,
            vla_backend=backend,
            task="stack_blocks",
            layout=0,
        )
    task = FRANKA_TASKS[1]
    add(
        "franka",
        "eval",
        mode="eval",
        task_name=task.name,
        instruction=task.instruction,
        success_criteria=task.success_criteria,
        constraints="\n".join(task.constraints),
    )
    rows = []
    for robot, label, values in cases:
        module = importlib.import_module(f"robots.{robot}.prompt_bundle")
        counts = {}
        for role in ("system", "user"):
            result = getattr(module, f"{role}_prompt")(values)
            if not isinstance(result, str):
                result = format_prompt(result, variables=values)
            result = result.replace(str(memory), "/memory")
            if "{{" in result:
                raise ValueError(f"Unbound variable in {robot}/{label}/{role}")
            counts[role] = len(result)
        rows.append(
            {"robot": robot, "case": label, **counts, "total": sum(counts.values())}
        )
    return rows


def main() -> None:
    """Compare separate interpreter imports using the same memory fixture."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path)
    parser.add_argument("--after", type=Path)
    parser.add_argument("--render", type=Path)
    parser.add_argument("--memory", type=Path)
    args = parser.parse_args()
    if args.render:
        print(json.dumps(render(args.render, args.memory), indent=2))
        return
    if not args.before or not args.after:
        parser.error("--before and --after are required")
    with tempfile.TemporaryDirectory(prefix="rpent-prompt-count-") as temporary:
        memory = Path(temporary).resolve() / "memory"
        files = {
            "global/GLOBAL_MEMORY.md": "# Synthetic global memory\n",
            "task-specific/OpenDrawer_s0.json": '{"success": true}\n',
            "task-specific/OpenDrawer_s0_recipe.jsonl": '{"action": "vla_act"}\n',
            "task-specific/OpenDrawer.md": "# Synthetic task memory\n",
        }
        for name, content in files.items():
            path = memory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        results = []
        for root in (args.before, args.after):
            output = subprocess.check_output(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--render",
                    str(root.resolve()),
                    "--memory",
                    str(memory),
                ],
                cwd=temporary,
                text=True,
            )
            results.append(json.loads(output))
    comparison = []
    for before, after in zip(*results, strict=True):
        assert (before["robot"], before["case"]) == (after["robot"], after["case"])
        comparison.append(
            {
                "robot": before["robot"],
                "case": before["case"],
                "before": {k: before[k] for k in ("system", "user", "total")},
                "after": {k: after[k] for k in ("system", "user", "total")},
                "delta": after["total"] - before["total"],
            }
        )
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
