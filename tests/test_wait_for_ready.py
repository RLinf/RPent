"""Tests for wait_for_ready fail-fast / timeout behavior (stdlib unittest)."""
from __future__ import annotations

import unittest

from rpent.utils.rpc import wait_for_ready


class _FailingClient:
    """RpcClient whose healthz never succeeds."""

    def call(self, method, args=(), kwargs=None, *, timeout_s=None):
        raise ConnectionError("connection refused")

    def close(self):
        pass


class _DeadDaemon:
    """Daemon stub whose subprocess has already exited with ``rc``."""

    name = "env_server"

    def __init__(self, rc: int, tail: str = "") -> None:
        self._rc = rc
        self._tail = tail

    def poll(self):
        return self._rc

    def tail_log(self, max_lines: int = 20) -> str:
        return self._tail


class WaitForReadyTest(unittest.TestCase):
    def test_fail_fast_on_crashed_daemon(self):
        with self.assertRaises(RuntimeError) as cm:
            wait_for_ready(_FailingClient(), daemon=_DeadDaemon(1))
        msg = str(cm.exception)
        self.assertIn("env_server exited with code 1", msg)
        # Crashed before any healthz round-trip -> no bare "None" in message.
        self.assertNotIn("error: None", msg)

    def test_crash_log_tail_included(self):
        daemon = _DeadDaemon(1, tail="ModuleNotFoundError: No module named 'torch'")
        with self.assertRaises(RuntimeError) as cm:
            wait_for_ready(_FailingClient(), daemon=daemon)
        self.assertIn("No module named 'torch'", str(cm.exception))

    def test_timeout_without_daemon(self):
        with self.assertRaises(TimeoutError):
            wait_for_ready(_FailingClient(), timeout_s=0.1, poll_interval_s=0.01)


if __name__ == "__main__":
    unittest.main()
