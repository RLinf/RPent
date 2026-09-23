#!/usr/bin/env python3
# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS.
"""Install a pinned RoboDawn controller and the RPent bridge in an isolated experiment."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import yaml

POLICY = "RoboDojo_EmbodiedAgent"
ROBODAWN_COMMIT = "9247f366cd31f278e10f2fbe5fe8469b5f1b5b94"


def install(source: Path, experiment: Path) -> Path:
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != ROBODAWN_COMMIT:
        raise ValueError(
            f"RoboDawn checkout must be pinned to {ROBODAWN_COMMIT}; got {revision}"
        )
    workspace = experiment / "workspace"
    target = workspace / "XPolicyLab/policy" / POLICY
    if target.exists():
        raise FileExistsError(target)
    if not (workspace / "RoboDojo-eval/scripts/eval_policy.sh").is_file():
        raise FileNotFoundError("prepare an isolated RoboDojo experiment first")
    source_policy = source / "evaluation/policies/vlm_agent"
    shutil.copytree(
        source_policy, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    examples = Path(__file__).resolve().parent
    shutil.copy2(examples / "model.py", target / "model.py")
    demos = workspace / "RoboDawn-demos"
    shutil.copytree(source / "demos/robodojo", demos)
    shutil.copy2(source / "LICENSE", workspace / "RoboDawn-LICENSE")
    cfg_path = target / "deploy.yml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["policy_name"] = POLICY
    cfg["policy_uv_env_path"] = "../.."
    cfg["result_dir"] = str(experiment / "results" / POLICY)
    cfg["vlm_agent"]["icl"]["demo_bank"] = str(demos)
    cfg["vlm_agent"]["log_dir"] = str(experiment / "runtime" / "vlm_logs")
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    # The prepared workspace supplies Isaac-compatible Python shims and the
    # XPolicyLab server/client launch scripts. RoboDawn's conda wrappers do not.
    template = workspace / "XPolicyLab/policy/RoboDojo_Agent_L3_Inspect"
    for name in ("setup_eval_policy_server.sh", "setup_eval_env_client.sh"):
        script = (template / name).read_text(encoding="utf-8")
        if name == "setup_eval_policy_server.sh":
            script = script.replace(
                'PYTHONPATH="${BENCH_ROOT}"',
                'PYTHONPATH="${BENCH_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"',
            )
            if 'PYTHONPATH="${BENCH_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"' not in script:
                raise RuntimeError(
                    "XPolicyLab server script changed its PYTHONPATH contract"
                )
        (target / name).write_text(script, encoding="utf-8")
        (target / name).chmod(0o755)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    print(install(args.source.resolve(), args.experiment.resolve()))
