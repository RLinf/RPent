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

"""Native worker requests and teardown must not overlap on its Pipe."""

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from threading import Event
from types import SimpleNamespace

import pytest

from robots.libero.env_client import LiberoEnvClient
from robots.libero.env_server import LiberoEnvFacade
from rpent.robots.components.env_client_base import BaseEnvClient


def test_libero_client_allows_bounded_cold_reset_without_relaxing_actions():
    calls = []

    def call(method, **kwargs):
        calls.append((method, kwargs["timeout_s"]))
        if method == "env.step":
            return {}, 0, False, False, {}
        return {} if method == "env.get_env_meta" else ({}, {})

    client = LiberoEnvClient(SimpleNamespace(call=call), expected_meta={})
    client.step([0] * 7)
    assert calls == [
        ("env.get_env_meta", 30.0),
        ("env.reset", 900.0),
        ("env.step", 60.0),
    ]
    calls.clear()
    BaseEnvClient(SimpleNamespace(call=call), expected_meta={})
    assert calls == [("env.get_env_meta", 30.0), ("env.reset", 120.0)]


@pytest.mark.parametrize(
    "second_method", ["env.get_camera_meta", "env.render_camera", "env.get_action_spec"]
)
def test_queries_do_not_interleave_native_pipe_replies(second_method):
    entered, release = Event(), Event()
    second_entered = Event()

    def query(*, hold=False):
        if hold:
            entered.set()
            assert release.wait(5)
            return ["frame"]
        second_entered.set()

    facade = LiberoEnvFacade(SimpleNamespace(), meta={})
    # Replace native work, retaining each method's registered locking policy.
    facade._rpc["env.render_camera"] = query
    facade._rpc[second_method] = query
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(facade._dispatch, "env.render_camera", (), {"hold": True})
        assert entered.wait(5)
        second = pool.submit(facade._dispatch, second_method, (), {})
        try:
            with pytest.raises(TimeoutError):
                second.result(timeout=0.1)
            assert not second_entered.is_set()
        finally:
            release.set()
        assert first.result(timeout=5) == ["frame"]
        second.result(timeout=5)
    assert second_entered.is_set()


def test_close_drains_reset_then_releases_worker_once():
    entered, release = Event(), Event()
    closed = []

    def reset():
        entered.set()
        assert release.wait(5)
        assert not closed
        return {"task_descriptions": ["task"]}, {}

    env = SimpleNamespace(
        reset=reset,
        task_ids=[0],
        trial_ids=[16],
        env=SimpleNamespace(close=lambda: closed.append(True)),
    )
    facade = LiberoEnvFacade(env, meta={})
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = pool.submit(facade._dispatch, "env.reset", (), {})
        assert entered.wait(5)
        closing = pool.submit(facade.close)
        try:
            with pytest.raises(TimeoutError):
                closing.result(timeout=0.1)
            assert not closed
        finally:
            release.set()
        assert pending.result(timeout=5)[1]["init_state_id"] == 16
        closing.result(timeout=5)
    facade.close()
    assert closed == [True]
