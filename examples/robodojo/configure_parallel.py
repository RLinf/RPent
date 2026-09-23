#!/usr/bin/env python3
# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS.
"""Adapt an isolated RoboProbe once-run launcher to RoboDojo EmbodiedAgent."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

POLICY = "RoboDojo_EmbodiedAgent"


def replace_once(path: Path, before: str, after: str) -> None:
    content = path.read_text(encoding="utf-8")
    if content.count(before) != 1:
        raise RuntimeError(f"expected one launcher anchor in {path}: {before[:60]}")
    path.write_text(content.replace(before, after), encoding="utf-8")


def configure(
    experiment: Path,
    repo: Path,
    *,
    key_file: Path | None = None,
    decision_limit: int | None = None,
    standard_tasks_only: bool = False,
) -> None:
    launcher = experiment / "manage.py"
    worker = experiment / "worker.sh"
    manifest_path = experiment / "manifest.json"
    for path in (launcher, worker, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    replace_once(
        launcher, 'POLICY = "RoboDojo_Agent_L3_Inspect_EEF"', f'POLICY = "{POLICY}"'
    )
    replace_once(
        launcher,
        '"L3_INSPECT_ADDITIONAL_INFO": "ckpt_name=tokenhub-once,action_type=joint",',
        '"L3_INSPECT_ADDITIONAL_INFO": "ckpt_name=tokenhub-once,action_type=joint",\n'
        '        "ROBODOJO_LLM_MODEL": "gpt-6-astra/azure_L/qwb",\n'
        '        "ROBODOJO_LLM_BASE_URL": "https://tokenhub.sensetime.com/v1",\n'
        '        "VLM_AGENT_RUN_TAG": run_id,\n'
        '        "VLM_AGENT_OVERRIDES": json.dumps({"max_decisions": '
        + str(decision_limit)
        + "}) if "
        + str(decision_limit is not None)
        + ' else "{}",',
    )
    # The prepared worker loads the key from a private file. Import the new
    # policy before launching Isaac, and expose RPent plus extra Python deps.
    replace_once(
        worker,
        'export PYTHONPATH="${RP_ONCE_DEPS}:${WORKSPACE}',
        f'export PYTHONPATH="${{RP_ONCE_DEPS}}:${{EXPERIMENT}}/runtime/python:{repo}:${{WORKSPACE}}',
    )
    replace_once(
        worker,
        "from XPolicyLab.policy.RoboDojo_Agent_L3_Inspect_EEF.model import Model",
        f"from XPolicyLab.policy.{POLICY}.model import Model",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if standard_tasks_only:
        manifest["tasks"] = [
            task for task in manifest["tasks"] if not task.endswith("_random")
        ]
        if len(manifest["tasks"]) != 42:
            raise RuntimeError("expected 42 standard RoboDojo tasks")
        replace_once(
            launcher, 'len(manifest["tasks"]) != 54', 'len(manifest["tasks"]) != 42'
        )
        replace_once(
            launcher,
            'len(set(manifest["tasks"])) != 54',
            'len(set(manifest["tasks"])) != 42',
        )
        replace_once(launcher, "exactly 54 unique tasks", "exactly 42 unique tasks")
        replace_once(launcher, '"total": 54', '"total": 42')
        replace_once(launcher, 'status["scored"] == 54', 'status["scored"] == 42')
        replace_once(launcher, "scored={}/54", "scored={}/42")
        manager = launcher.read_text(encoding="utf-8")
        before = manager.index("def validate(exp):\n")
        after = manager.index("\n\ndef plan(", before)
        if "chain_evidence(exp, general)" not in manager[before:after]:
            raise RuntimeError("unknown RoboProbe validation contract")
        manager = (
            manager[:before]
            + """def validate(exp):
    status = report(exp, write=True)
    evidence = {}
    for task in load_manifest(exp)["tasks"]:
        scored = next((row for row in status["results"] if row["task"] == task), None)
        if scored is None:
            evidence[task] = False
            continue
        root = exp / "runtime/vlm_logs" / scored["run_id"]
        evidence[task] = any(root.glob("**/decision_*.json"))
    complete = (status["scored"] == 42 and not status["conflicts"]
                and not status["invalid_files"] and all(evidence.values()))
    payload = {"complete": bool(complete), "status": status,
               "decision_logs_found": evidence}
    path = exp / "results" / ("once-validation.json" if complete else "once-validation-incomplete.json")
    tmp = path.with_suffix(".json.pending")
    tmp.write_text(json.dumps(payload, indent=2) + "\\n")
    tmp.replace(path)
    return bool(complete)
"""
            + manager[after:]
        )
        launcher.write_text(manager, encoding="utf-8")
        manifest["protocol"] = (
            "42 standard RoboDojo tasks, seed=0, layout=0, one native scored episode per task"
        )
    manifest["policy"] = POLICY
    manifest["robodawn_commit"] = "9247f366cd31f278e10f2fbe5fe8469b5f1b5b94"
    manifest["rpent_repo"] = str(repo)
    manifest["decision_limit"] = decision_limit
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    key_target = experiment / "runtime/openai_api_key"
    if key_file is not None and not key_target.exists():
        shutil.copyfile(key_file, key_target)
        key_target.chmod(0o600)
    if not key_target.is_file():
        raise FileNotFoundError("provide --key-file or create runtime/openai_api_key")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--decision-limit", type=int)
    parser.add_argument("--standard-tasks-only", action="store_true")
    args = parser.parse_args()
    configure(
        args.experiment.resolve(),
        args.repo.resolve(),
        key_file=args.key_file.resolve() if args.key_file else None,
        decision_limit=args.decision_limit,
        standard_tasks_only=args.standard_tasks_only,
    )
