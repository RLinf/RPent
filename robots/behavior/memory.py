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

from robots.behavior.dino_v2.index import DinoIndexError, _read_evidence, _writer_lock
from robots.behavior.terminal_success import validate_official_success_receipt
from rpent.memory import MemoryManager


class BehaviorMemoryManager(MemoryManager):
    task_artifacts: Path | None = None

    def merge_memory(
        self, *, cell_tag: str, run_state_dir: str | Path, solved: bool
    ) -> dict[str, Any]:
        result = super().merge_memory(
            cell_tag=cell_tag, run_state_dir=run_state_dir, solved=solved
        )
        if not solved or self.task_artifacts is None:
            return result
        audit_path = self.root / "task-specific" / f"{cell_tag}.json"
        recipe_path = audit_path.with_name(f"{cell_tag}_recipe.jsonl")
        if not audit_path.is_file() or not recipe_path.is_file():
            return result
        audit = json.loads(audit_path.read_text())
        receipt = validate_official_success_receipt(
            audit.get("official_success_receipt")
        )
        if receipt is None:
            raise ValueError(
                "canonical task audit has no valid official success receipt"
            )
        # The main manager may retain an older task pair. Never attach evidence
        # from another episode to that pair, or undo canonical publication.
        artifacts = self.root / "task-specific" / "artifacts"
        with _writer_lock(artifacts):
            destination = artifacts / cell_tag
            if destination.exists():
                try:
                    _read_evidence(
                        self.root,
                        destination / "manifest.json",
                        receipt["receipt_sha256"],
                    )
                except DinoIndexError:
                    if not result["task"]:
                        raise
                else:
                    return result
            _read_evidence(
                Path(run_state_dir).resolve(),
                self.task_artifacts / "manifest.json",
                receipt["receipt_sha256"],
            )
            with tempfile.TemporaryDirectory(
                prefix=f".{cell_tag}.", dir=artifacts
            ) as temporary:
                staging = Path(temporary) / "evidence"
                shutil.copytree(self.task_artifacts, staging)
                if destination.exists():
                    shutil.rmtree(destination)
                os.replace(staging, destination)
        return result
