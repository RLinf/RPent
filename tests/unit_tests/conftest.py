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

import hashlib
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
        manifest = {
            "schema_version": 1,
            "files": {
                name: hashlib.sha256(text.encode()).hexdigest()
                for name, text in files.items()
            },
        }
        (root / "CORPUS.json").write_text(json.dumps(manifest))
        return root

    return create
