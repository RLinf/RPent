#!/usr/bin/env python3
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

"""Host-local flock slot allocator for GPU and install CI jobs.

Multiple homogeneous GitHub Actions runners on a single GPU host share a
local lock directory (default ``/tmp/rpent-ci-slots``). Each job acquires
``--slots N`` exclusive slots atomically under a short-lived ``meta.lock``
so multi-GPU jobs (for example ``--slots 2``) never deadlock against each
other, and the OS kernel releases all locks automatically if a job exits
or is killed.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import random
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from pathlib import Path

DEFAULT_LOCK_DIR = "/tmp/rpent-ci-slots"
DEFAULT_INSTALL_SLOTS = 4


def parse_slot_pool(
    kind: str,
    raw_pool: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> list[str]:
    """Return ordered slot identifiers for the requested pool kind."""
    env = os.environ if environ is None else environ
    if raw_pool is not None and raw_pool.strip():
        tokens = [t.strip() for t in raw_pool.split(",") if t.strip()]
    elif kind == "gpu":
        source = (
            env.get("RPENT_GPU_POOL", "").strip()
            or env.get("CUDA_VISIBLE_DEVICES", "").strip()
        )
        tokens = [t.strip() for t in source.split(",") if t.strip().isdecimal()]
        if not tokens:
            raise ValueError(
                "RPENT_GPU_POOL or CUDA_VISIBLE_DEVICES must list at least one "
                f"numeric GPU ordinal (got {source!r})"
            )
    elif kind == "install":
        raw_count = env.get("RPENT_INSTALL_SLOTS", str(DEFAULT_INSTALL_SLOTS)).strip()
        if not raw_count.isdecimal() or int(raw_count) < 1:
            raise ValueError(
                f"RPENT_INSTALL_SLOTS must be a positive integer, got {raw_count!r}"
            )
        tokens = [str(i) for i in range(int(raw_count))]
    else:
        raise ValueError(f"Unsupported slot kind: {kind!r}")

    deduped: list[str] = []
    for token in tokens:
        if token not in deduped:
            deduped.append(token)
    return deduped


def _release_fds(fds: list[int]) -> None:
    for fd in fds:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            os.close(fd)
        except OSError:
            pass


def acquire_slots(
    *,
    kind: str,
    count: int,
    pool: list[str],
    lock_dir: Path,
    timeout_s: float = 7200.0,
    poll_interval_s: float = 2.0,
) -> tuple[list[str], list[int]]:
    """Atomically acquire ``count`` distinct slots from ``pool``.

    Uses a per-kind ``meta.lock`` to serialize probing so two multi-slot
    waiters cannot hold partial sets and deadlock.
    """
    if count < 1:
        raise ValueError(f"Slot count must be >= 1, got {count}")
    if count > len(pool):
        raise ValueError(
            f"Requested {count} {kind} slot(s), but pool only has {len(pool)}: {pool}"
        )

    lock_dir.mkdir(parents=True, exist_ok=True)
    meta_path = lock_dir / f"{kind}-meta.lock"
    deadline = time.monotonic() + timeout_s

    while True:
        meta_fd = os.open(meta_path, os.O_CREAT | os.O_RDWR, 0o666)
        try:
            fcntl.flock(meta_fd, fcntl.LOCK_EX)
            acquired_ids: list[str] = []
            acquired_fds: list[int] = []
            for slot_id in pool:
                slot_path = lock_dir / f"{kind}-{slot_id}.lock"
                fd = os.open(slot_path, os.O_CREAT | os.O_RDWR, 0o666)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    os.close(fd)
                    continue
                acquired_ids.append(slot_id)
                acquired_fds.append(fd)
                if len(acquired_ids) == count:
                    for held_fd in acquired_fds:
                        os.ftruncate(held_fd, 0)
                        os.lseek(held_fd, 0, os.SEEK_SET)
                        os.write(held_fd, f"pid={os.getpid()}\n".encode("ascii"))
                    return acquired_ids, acquired_fds
            _release_fds(acquired_fds)
        finally:
            try:
                fcntl.flock(meta_fd, fcntl.LOCK_UN)
            finally:
                os.close(meta_fd)

        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Timed out after {timeout_s:.0f}s waiting for {count} {kind} "
                f"slot(s) from pool {pool}"
            )
        jitter = random.uniform(0.0, min(0.5, poll_interval_s * 0.25))
        time.sleep(poll_interval_s + jitter)


def run_with_slots(
    cmd: list[str],
    *,
    kind: str,
    count: int,
    pool: list[str],
    lock_dir: Path,
    timeout_s: float = 7200.0,
    poll_interval_s: float = 2.0,
    environ: Mapping[str, str] | None = None,
) -> int:
    """Acquire slots, run ``cmd`` with slot env vars, and release on exit."""
    t0 = time.monotonic()
    slot_ids, held_fds = acquire_slots(
        kind=kind,
        count=count,
        pool=pool,
        lock_dir=lock_dir,
        timeout_s=timeout_s,
        poll_interval_s=poll_interval_s,
    )
    waited = time.monotonic() - t0
    joined = ",".join(slot_ids)
    print(
        f"[slots:{kind}] acquired {count} slot(s) [{joined}] "
        f"(waited {waited:.1f}s, pool={','.join(pool)})",
        flush=True,
    )

    child_env = dict(os.environ if environ is None else environ)
    child_env["RPENT_ALLOCATED_SLOTS"] = joined
    if kind == "gpu":
        child_env["CUDA_VISIBLE_DEVICES"] = joined

    proc: subprocess.Popen[bytes] | None = None

    def _forward_signal(signum: int, _frame: object) -> None:
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(proc.pid, signum)
            except ProcessLookupError:
                pass

    is_main = threading.current_thread() is threading.main_thread()
    prev_term = signal.signal(signal.SIGTERM, _forward_signal) if is_main else None
    prev_int = signal.signal(signal.SIGINT, _forward_signal) if is_main else None
    try:
        proc = subprocess.Popen(cmd, env=child_env, start_new_session=True)
        return proc.wait()
    finally:
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
        if is_main and prev_term is not None:
            signal.signal(signal.SIGTERM, prev_term)
        if is_main and prev_int is not None:
            signal.signal(signal.SIGINT, prev_int)
        _release_fds(held_fds)
        print(f"[slots:{kind}] released slot(s) [{joined}]", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("gpu", "install"), required=True)
    parser.add_argument("--slots", type=int, default=1)
    parser.add_argument("--pool", default=None)
    parser.add_argument(
        "--lock-dir",
        type=Path,
        default=Path(os.environ.get("RPENT_SLOT_LOCK_DIR", DEFAULT_LOCK_DIR)),
    )
    parser.add_argument("--timeout-s", type=float, default=7200.0)
    parser.add_argument("--poll-interval-s", type=float, default=2.0)
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cmd = list(args.cmd)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        print("Error: command after '--' is required", file=sys.stderr)
        return 2
    pool = parse_slot_pool(args.kind, args.pool)
    return run_with_slots(
        cmd,
        kind=args.kind,
        count=args.slots,
        pool=pool,
        lock_dir=args.lock_dir,
        timeout_s=args.timeout_s,
        poll_interval_s=args.poll_interval_s,
    )


if __name__ == "__main__":
    raise SystemExit(main())
