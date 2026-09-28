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

"""Small, offline RoboCasa memory corpora for contract tests."""

from __future__ import annotations

import json

import pytest


@pytest.fixture
def make_corpus():
    def create(
        root,
        *,
        tasks=("OpenDrawer", "StirVegetables"),
        markdown=None,
        global_memory=True,
    ):
        from robots.robocasa.memory import GLOBAL_FILE, task_files

        files = {}
        markdown = tasks if markdown is None else markdown
        for task in tasks:
            names = task_files(task)
            files[names[0]] = (
                json.dumps({"task": task, "seed": 0, "success": True}) + "\n"
            )
            files[names[1]] = '{"action":"vla_act"}\n'
            if task in markdown:
                files[names[2]] = f"# {task}\nRe-ground from live observations.\n"
        if global_memory:
            files[GLOBAL_FILE] = (
                "# Reviewed global prior\nPreserve productive contact.\n"
            )
        for name, text in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        root.mkdir(parents=True, exist_ok=True)
        return root

    return create


@pytest.fixture
def make_native_corpus(tmp_path):
    """Exercise the real merge API with explicitly synthetic exploration evidence."""

    def create(root, *, task="OpenDrawer", split="target"):
        from rpent.memory import MemoryManager

        tag = f"{task}_{split}_s0"
        run = tmp_path / f"synthetic-export-{tag}"
        run.mkdir(exist_ok=True)
        (run / f"{tag}.json").write_text('{"fixture": "synthetic export"}\n')
        (run / f"{tag}_recipe.jsonl").write_text(
            '{"action":"move_to","fixture":true}\n'
        )
        inbox = root / "_internal/inbox" / tag
        inbox.mkdir(parents=True)
        common = f"evidence:\n  cells: [{tag}]\nconfidence: single-shot\n"
        (inbox / "family.md").write_text(
            f"---\nscope: task-family\nsuite: robocasa\nregime: {split}\n"
            f"task_id: {task}\ntask_language: Synthetic offline test task\n"
            + common
            + "---\nSynthetic task-family note; no robot was run.\n"
        )
        (inbox / "global.md").write_text(
            "---\nscope: global\nkind: strategy\ntitle: Synthetic test prior\n"
            "applies_when: Offline test only\n"
            + common
            + "---\nSynthetic global note; no robot was run.\n"
        )
        result = MemoryManager(root).merge_memory(
            cell_tag=tag, run_state_dir=run, solved=True
        )
        assert result["task"] == result["task-family"] == result["global"] == 1
        assert not result["skipped"]
        return root

    return create
