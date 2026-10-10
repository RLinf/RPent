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

"""Offline functional check for the continuous manual policy entry point."""

import threading

from tests.manual.dual_franka_full_vla import run_full_vla
from tests.unit_tests.robots.dual_franka.test_vla_test import make_session


def test_policy_continues_across_chunks_until_operator_stop(tmp_path):
    session, calls, _ = make_session(tmp_path)
    stopped = threading.Event()
    actions = []

    def step(action):
        actions.append(action.copy())
        if len(actions) == 21:
            stopped.set()
        return {"ok": True}

    session.env.chunk_step = step
    result = run_full_vla(
        session.env,
        session.model,
        prompt=session.prompt,
        workspace=session.workspace,
        expected_steps=session.expected_steps,
        stopped=stopped.is_set,
        emit=lambda event: None,
    )
    assert len(actions) == result["steps"] == 21
    assert calls.count("predict") == result["chunks"] == 2
    assert result["reason"] == "operator_stop"
