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

from rpent.reproduce.robocasa.protocol import (
    ALLOWED_ACTIONS,
    CELLS,
    EMPTY_MEMORY_TASKS,
    EXPECTED_ROLLOUTS,
    MEMORY_PAIR_TASKS,
    PAPER_REFERENCES,
    PROTOCOL_ID,
    SPLITS,
    Split,
    cell_for,
)


def test_frozen_matrix_and_timeouts():
    assert len(CELLS) == EXPECTED_ROLLOUTS == 340
    assert len(ALLOWED_ACTIONS) == 8
    assert len(MEMORY_PAIR_TASKS) == 43
    assert len(EMPTY_MEMORY_TASKS) == 7
    assert SPLITS[Split.ATOMIC].timeout_seconds == 1800
    assert SPLITS[Split.COMPOSITE_SEEN].timeout_seconds == 3600
    assert SPLITS[Split.COMPOSITE_UNSEEN].timeout_seconds == 3600
    assert PROTOCOL_ID == "robocasa-harness-vla-v1"
    assert set(PAPER_REFERENCES) == {
        "RLDX-1",
        "Harness VLA (Codex)",
        "Harness VLA (CC)",
    }


def test_cell_for_rejects_non_protocol_cells():
    assert cell_for("atomic", "OpenDrawer", 1).tag == "OpenDrawer_s1"
    for args in (("atomic", "OpenDrawer", 0), ("atomic", "ArrangeTea", 1)):
        try:
            cell_for(*args)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid cell accepted")
