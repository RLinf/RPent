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

"""Pool fairness, complete request forwarding and finite queue waits."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from rpent.robots.components.sam3_pool import Sam3Pool


class Worker:
    def __init__(self, index, delay=0.01, fail=False):
        self.index, self.delay, self.fail = index, delay, fail
        self.active = 0

    def call(self, method, *, args=(), kwargs=None, timeout_s=None):
        assert self.active == 0
        self.active += 1
        try:
            time.sleep(self.delay)
            if self.fail:
                raise RuntimeError("worker failed")
            return {
                "worker": self.index,
                "method": method,
                "args": args,
                "kwargs": kwargs,
            }
        finally:
            self.active -= 1

    def close(self):
        pass


def test_round_robin_keeps_query_group_whole_and_workers_exclusive(tmp_path):
    pool = Sam3Pool(
        [Worker(i) for i in range(4)], metrics_path=tmp_path / "metrics.jsonl"
    )
    payload = {"image_base64": "same-frame", "queries": ["cube", "bowl"]}
    with ThreadPoolExecutor(max_workers=20) as executor:
        results = list(
            executor.map(
                lambda _: pool.forward("sam3.segment_queries", **payload), range(20)
            )
        )
    assert all(row["kwargs"] == payload for row in results)
    assert pool.stats()["requests"] == [5, 5, 5, 5]
    assert len((tmp_path / "metrics.jsonl").read_text().splitlines()) == 20


def test_worker_error_is_propagated_and_releases_slot():
    pool = Sam3Pool([Worker(0, fail=True)])
    with pytest.raises(RuntimeError, match="worker failed"):
        pool.forward("sam3.segment", image_base64="frame")
    assert pool.stats()["busy"] == [False]
    assert pool.stats()["errors"] == [1]


def test_queue_wait_has_deadline_and_removes_ticket():
    entered, release = threading.Event(), threading.Event()

    class BlockingWorker(Worker):
        def call(self, *args, **kwargs):
            entered.set()
            release.wait(1)

    pool = Sam3Pool([BlockingWorker(0)], timeout_s=0.05)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(pool.forward, "sam3.segment")
        assert entered.wait(1)
        try:
            with pytest.raises(TimeoutError, match="deadline"):
                pool.forward("sam3.segment")
            assert pool.stats()["pending"] == 0
        finally:
            release.set()
        first.result()


def test_optional_backend_admission_limit_is_observable_and_bounded():
    entered = threading.Barrier(3)
    release = threading.Event()
    lock = threading.Lock()
    running = 0

    class OverlappingWorker:
        def call(self, method, **kwargs):
            nonlocal running
            with lock:
                running += 1
                assert running <= 2
            try:
                entered.wait(timeout=2)
                assert release.wait(timeout=2)
                return method
            finally:
                with lock:
                    running -= 1

        def close(self):
            pass

    pool = Sam3Pool([OverlappingWorker()], per_backend_concurrency=2)
    with ThreadPoolExecutor(max_workers=2) as executor:
        calls = [executor.submit(pool.forward, "sam3.segment") for _ in range(2)]
        entered.wait(timeout=2)
        try:
            assert pool.stats()["inflight"] == [2]
            assert pool.stats()["busy"] == [True]
        finally:
            release.set()
        assert [call.result() for call in calls] == ["sam3.segment"] * 2
    assert pool.stats()["inflight"] == [0]
    assert pool.stats()["requests"] == [2]


def test_backend_admission_limit_must_be_positive():
    with pytest.raises(ValueError):
        Sam3Pool([Worker(0)], per_backend_concurrency=0)
