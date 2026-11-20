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

"""Select requested layouts before running the unchanged official evaluator."""

from __future__ import annotations

import json
import os
import runpy
from typing import Any


def select_seed_queue(manager: Any, layouts: list[int]) -> None:
    """Filter the official queue, retaining its original ID-to-layout mapping."""
    missing = set(layouts) - set(manager.seed_info)
    if missing:
        raise ValueError(
            f"Requested layouts unavailable in this seed group: {sorted(missing)}"
        )
    available = set(manager.seed_list)
    manager.seed_list = [layout for layout in layouts if layout in available]
    manager.ed_idx = len(manager.seed_list)
    manager.idx = 0
    manager._current_batch_seeds = None


def main() -> None:
    """Install only the per-process layout selector; retain official execution/scoring."""
    from env.seed_manager.seed_manager import SeedManager

    layouts = json.loads(os.environ["RPENT_XPL_LAYOUTS"])
    original_init = SeedManager.init_eval

    def init_selected(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        select_seed_queue(self, layouts)

    SeedManager.init_eval = init_selected
    runpy.run_module("src.eval_client.main", run_name="__main__")


if __name__ == "__main__":
    main()
