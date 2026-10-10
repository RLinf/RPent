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

"""Continuous manual VLA cancellation tests without hardware or network."""

import threading

import numpy as np
import pytest

from tests.manual.dual_franka_full_vla import MANUAL_SKILLS, run_full_vla
from tests.unit_tests.robots.dual_franka.test_vla_test import make_session


def run(session, stop, events):
    return run_full_vla(
        session.env,
        session.model,
        prompt=session.prompt,
        workspace=session.workspace,
        expected_steps=session.expected_steps,
        stopped=stop,
        emit=events.append,
    )


def test_continues_past_twenty_chunks_without_reset_or_semantic_stops(tmp_path):
    session, calls, _ = make_session(tmp_path)
    count = [0]
    stopped = threading.Event()

    def step(actions):
        assert actions.shape == (1, 20)
        count[0] += 1
        if count[0] == 501:
            stopped.set()
        return {"ok": True}

    session.env.chunk_step = step
    events = []
    result = run(session, stopped.is_set, events)
    assert count[0] == 501
    assert result["chunks"] == 26
    assert calls.count("predict") == 26
    assert result["reason"] == "operator_stop"
    assert events[-1]["steps"] == 501


@pytest.mark.parametrize("stage", ["before_start", "observation", "prediction"])
def test_stop_before_action_never_sends_prediction(tmp_path, stage):
    session, calls, _ = make_session(tmp_path)
    stop = threading.Event()
    if stage == "before_start":
        stop.set()
    else:
        owner, name = (
            (session.env, "get_observation")
            if stage == "observation"
            else (session.model, "predict")
        )
        original = getattr(owner, name)

        def interrupted(*args, **kwargs):
            result = original(*args, **kwargs)
            stop.set()
            return result

        setattr(owner, name, interrupted)
    result = run(session, stop.is_set, [])
    assert "execute" not in calls
    assert result["steps"] == 0


def test_external_stop_file_discards_remaining_chunk(tmp_path):
    session, _, _ = make_session(tmp_path)
    stop_file = tmp_path / "STOP"
    calls = []

    def step(actions):
        calls.append(actions)
        stop_file.touch()
        return {"ok": True}

    session.env.chunk_step = step
    result = run(session, stop_file.exists, [])
    assert len(calls) == result["steps"] == 1


@pytest.mark.parametrize("result", [{"ok": False}, {"error": "controller fault"}])
def test_reported_error_stops_without_retry(tmp_path, result):
    session, _, _ = make_session(tmp_path)
    calls = []
    session.env.chunk_step = lambda actions: calls.append(actions) or result
    events = []
    with pytest.raises(RuntimeError, match="environment action failed"):
        run(session, lambda: False, events)
    assert len(calls) == 1
    assert events[-1]["status"] == "failed"


def test_rpc_exception_stops_without_retry(tmp_path):
    session, calls, _ = make_session(tmp_path, fail=True)
    with pytest.raises(RuntimeError, match="RPC timeout"):
        run(session, lambda: False, [])
    assert calls.count("execute") == 1


def test_invalid_prediction_never_moves(tmp_path):
    session, calls, _ = make_session(tmp_path)
    session.model.predict = lambda *args, **kwargs: np.full((20, 20), np.nan)
    with pytest.raises(ValueError, match="finite actions"):
        run(session, lambda: False, [])
    assert "execute" not in calls


@pytest.mark.parametrize("field", ["terminated", "truncated"])
def test_environment_stop_is_not_automatically_reset(tmp_path, field):
    session, _, _ = make_session(tmp_path)
    calls = []
    session.env.chunk_step = lambda actions: calls.append(actions) or {field: True}
    result = run(session, lambda: False, [])
    assert len(calls) == 1
    assert result["status"] == "environment_stopped"


def test_manual_registry_is_not_agent_schema():
    from robots.dual_franka.toolkit import DualFrankaToolkit
    from robots.dual_franka.tools import DualFrankaPrimitives

    assert "vla_full_rollout" in MANUAL_SKILLS
    assert "vla_full_rollout" not in {
        tool.name for tool in DualFrankaToolkit.declared_tools()
    }
    assert not hasattr(DualFrankaPrimitives, "vla_full_rollout")
