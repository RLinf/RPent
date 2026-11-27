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

"""Unit tests for host-local flock GPU and install slot allocation."""

from __future__ import annotations

import concurrent.futures
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.tooling.gpu_slots import acquire_slots, parse_slot_pool, run_with_slots


def test_parse_slot_pool_gpu_and_install() -> None:
    assert parse_slot_pool("gpu", environ={"RPENT_GPU_POOL": "0, 1, 2, 1"}) == [
        "0",
        "1",
        "2",
    ]
    assert parse_slot_pool("gpu", environ={"CUDA_VISIBLE_DEVICES": "4,5"}) == [
        "4",
        "5",
    ]
    assert parse_slot_pool("install", environ={"RPENT_INSTALL_SLOTS": "3"}) == [
        "0",
        "1",
        "2",
    ]
    with pytest.raises(ValueError, match="RPENT_GPU_POOL"):
        parse_slot_pool("gpu", environ={})
    with pytest.raises(ValueError, match="RPENT_INSTALL_SLOTS"):
        parse_slot_pool("install", environ={"RPENT_INSTALL_SLOTS": "0"})


def test_concurrent_single_and_dual_gpu_slots_no_overlap_no_deadlock(
    tmp_path: Path,
) -> None:
    lock_dir = tmp_path / "slots"
    events_dir = tmp_path / "events"
    events_dir.mkdir()
    pool = ["0", "1", "2", "3"]
    # Mix of 1-GPU and 2-GPU jobs exceeding total pool capacity (2+2+1+1+2 = 8 > 4).
    requests = [(2, "a"), (2, "b"), (1, "c"), (1, "d"), (2, "e")]

    def _run_job(req: tuple[int, str]) -> dict[str, object]:
        slots, tag = req
        out_file = events_dir / f"{tag}.json"
        code = (
            "import json, os, time; "
            "t0 = time.monotonic(); "
            "time.sleep(0.15); "
            "t1 = time.monotonic(); "
            f"open({str(out_file)!r}, 'w').write(json.dumps("
            "{'gpus': os.environ['CUDA_VISIBLE_DEVICES'].split(','), "
            "'start': t0, 'end': t1}))"
        )
        rc = run_with_slots(
            [sys.executable, "-c", code],
            kind="gpu",
            count=slots,
            pool=pool,
            lock_dir=lock_dir,
            timeout_s=15.0,
            poll_interval_s=0.05,
        )
        assert rc == 0
        return json.loads(out_file.read_text(encoding="utf-8"))

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(requests)) as ex:
        results = list(ex.map(_run_job, requests))

    for (slots, _), res in zip(requests, results, strict=True):
        assert len(res["gpus"]) == slots
        assert len(set(res["gpus"])) == slots

    # Verify no two overlapping time intervals ever shared a physical GPU.
    for i, a in enumerate(results):
        for b in results[i + 1 :]:
            overlaps = a["start"] < b["end"] and b["start"] < a["end"]
            if overlaps:
                assert not (set(a["gpus"]) & set(b["gpus"]))


def test_slots_released_automatically_on_sigkill(tmp_path: Path) -> None:
    lock_dir = tmp_path / "slots"
    ready_file = tmp_path / "ready"
    script = Path(__file__).resolve().parents[2] / "tooling" / "gpu_slots.py"

    holder = subprocess.Popen(
        [
            sys.executable,
            str(script),
            "--kind",
            "gpu",
            "--slots",
            "2",
            "--pool",
            "0,1",
            "--lock-dir",
            str(lock_dir),
            "--",
            sys.executable,
            "-c",
            f"open({str(ready_file)!r}, 'w').write('ok'); import time; time.sleep(30)",
        ],
    )
    try:
        deadline = time.monotonic() + 5.0
        while not ready_file.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert ready_file.exists()

        # Pool is fully occupied; immediate acquisition must time out.
        with pytest.raises(TimeoutError):
            acquire_slots(
                kind="gpu",
                count=1,
                pool=["0", "1"],
                lock_dir=lock_dir,
                timeout_s=0.1,
                poll_interval_s=0.05,
            )

        # Hard-kill holder process; OS kernel must release both flock fds.
        os.kill(holder.pid, signal.SIGKILL)
        holder.wait(timeout=5)
    finally:
        if holder.poll() is None:
            holder.kill()
            holder.wait(timeout=5)

    # Now both slots must be immediately acquirable.
    ids, fds = acquire_slots(
        kind="gpu",
        count=2,
        pool=["0", "1"],
        lock_dir=lock_dir,
        timeout_s=2.0,
        poll_interval_s=0.05,
    )
    for fd in fds:
        os.close(fd)
    assert ids == ["0", "1"]
