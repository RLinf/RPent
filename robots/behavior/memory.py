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

"""Attach BEHAVIOR image evidence to canonical MemoryManager task records."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from robots.behavior.dino_v2.index import DinoIndexError, _read_evidence
from robots.behavior.terminal_success import validate_official_success_receipt
from rpent.memory import MemoryManager


class BehaviorMemoryManager(MemoryManager):
    task_artifacts: Path | None = None

    def _publish_task_artifacts(
        self,
        audit: Path,
        recipe: Path,
        task_dir: Path,
        result: dict[str, Any],
    ) -> None:
        """Make DINO evidence visible before the canonical audit is published."""
        if self.task_artifacts is None:
            return super()._publish_task_artifacts(audit, recipe, task_dir, result)

        cell_tag = audit.stem
        audit_target = task_dir / audit.name
        recipe_target = task_dir / recipe.name
        if audit_target.exists() != recipe_target.exists():
            # Recipe-first publication can leave this state after process death.
            if (
                audit_target.exists()
                or recipe_target.read_bytes() != recipe.read_bytes()
            ):
                result["skipped"].append("incomplete existing task audit/recipe pair")
                return
        paired = audit_target.exists() and recipe_target.exists()
        destination = task_dir / "artifacts" / cell_tag
        if destination.is_symlink():
            raise ValueError(f"task evidence path must not be a symlink: {destination}")

        if paired:
            published_audit = json.loads(audit_target.read_text())
            published_receipt = validate_official_success_receipt(
                published_audit.get("official_success_receipt")
            )
            if published_receipt is None:
                raise ValueError(
                    "canonical task audit has no valid official success receipt"
                )
            if destination.exists():
                _read_evidence(
                    self.root,
                    destination / "manifest.json",
                    published_receipt["receipt_sha256"],
                )
                return

        source_audit = json.loads(audit.read_text())
        receipt = validate_official_success_receipt(
            source_audit.get("official_success_receipt")
        )
        if receipt is None:
            raise ValueError("task audit has no valid official success receipt")
        if paired and published_receipt["receipt_sha256"] != receipt["receipt_sha256"]:
            raise ValueError("run evidence does not match the canonical task audit")
        source = self.task_artifacts.resolve()
        _read_evidence(
            audit.parent, source / "manifest.json", receipt["receipt_sha256"]
        )

        evidence_ready = False
        if destination.exists():
            try:
                _read_evidence(
                    self.root,
                    destination / "manifest.json",
                    receipt["receipt_sha256"],
                )
            except DinoIndexError:
                if paired:
                    raise
            else:
                evidence_ready = True

        with tempfile.TemporaryDirectory(prefix=f".{cell_tag}.", dir=task_dir) as temp:
            staging = Path(temp)
            if not evidence_ready:
                staged_evidence = staging / "evidence"
                shutil.copytree(source, staged_evidence)
                _read_evidence(
                    staging,
                    staged_evidence / "manifest.json",
                    receipt["receipt_sha256"],
                )
            if not paired:
                shutil.copy2(audit, staging / audit.name)
                if not recipe_target.exists():
                    shutil.copy2(recipe, staging / recipe.name)

            destination.parent.mkdir(parents=True, exist_ok=True)
            backup = staging / "previous_evidence"
            backed_up = installed_evidence = installed_recipe = installed_audit = False
            try:
                if not evidence_ready:
                    if destination.exists():
                        os.replace(destination, backup)
                        backed_up = True
                    os.replace(staged_evidence, destination)
                    installed_evidence = True
                if not paired:
                    if not recipe_target.exists():
                        os.replace(staging / recipe.name, recipe_target)
                        installed_recipe = True
                    # DINO reads only cells with an audit, recipe and evidence.
                    os.replace(staging / audit.name, audit_target)
                    installed_audit = True
                    result["task"] = 1
            except BaseException:
                if installed_audit:
                    audit_target.unlink()
                if installed_recipe:
                    recipe_target.unlink()
                if installed_evidence:
                    shutil.rmtree(destination)
                if backed_up:
                    os.replace(backup, destination)
                raise
