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

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from robots.behavior.memory import BehaviorMemoryManager


@pytest.fixture
def prepared_corpus(tmp_path):
    from robots.behavior.dino_v2.index import prepare_evidence_pack
    from robots.behavior.terminal_success import make_raw_success_receipt
    from rpent.session import EnvState

    cell = "turning_on_radio_s0"
    root = tmp_path / "memory"
    run = tmp_path / "run"
    state = EnvState(run / "sessions" / "session_001")
    receipt = make_raw_success_receipt({"done": {"success": True}}, env_step=32)
    terminal = {
        "_finish": True,
        "task_success": True,
        "official_success_receipt": receipt,
    }
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    with state.record_step(
        state={"total_env_steps": 32},
        terminated=True,
        command={"action": "pi0_nav_pick", "chunks": 1},
        result={"official_success": True},
    ):
        for name in ("head_rgb.png", "left_wrist_rgb.png", "right_wrist_rgb.png"):
            state.save(name, image)
    state.save("terminal_receipt.json", terminal, step=None)
    run_state = EnvState(run)
    run_state.save(
        f"{cell}.json",
        {
            "task_name": "turning_on_radio",
            "public_seed": 0,
            "official_success_receipt": receipt,
        },
        step=None,
    )
    run_state.save(
        f"{cell}_recipe.jsonl",
        [{"action": "pi0_nav_pick", "instruction": "turn on radio", "chunks": 1}],
        step=None,
    )
    prepared = prepare_evidence_pack(run, cell, state, terminal)
    assert prepared is not None
    return SimpleNamespace(
        root=root,
        run=run,
        cell=cell,
        prepared=prepared,
        image=image,
        state=state,
        terminal=terminal,
    )


@pytest.fixture
def official_corpus(prepared_corpus):
    manager = BehaviorMemoryManager(prepared_corpus.root)
    manager.task_artifacts = prepared_corpus.prepared
    result = manager.merge_memory(
        cell_tag=prepared_corpus.cell,
        run_state_dir=prepared_corpus.run,
        solved=True,
    )
    assert result["task"] == 1
    return (
        prepared_corpus.root,
        prepared_corpus.cell,
        prepared_corpus.image,
        prepared_corpus.state,
        prepared_corpus.terminal,
    )
