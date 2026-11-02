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

"""Sign in to the DeepSeek Harness ChatGPT route using a device code."""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import TextIO

_MARKER = "RPENT_DSH_LOGIN "


def _read_events(stream: TextIO, events: queue.Queue[str | None]) -> None:
    try:
        for line in stream:
            if line.startswith(_MARKER):
                events.put(line[len(_MARKER) :])
    finally:
        events.put(None)


def login(dsh_home: Path, *, timeout_s: float = 960) -> bool:
    """Authorize ChatGPT access and retain the grant in the DSH credential store.

    Args:
        dsh_home: Explicit Harness home shared with subsequent planner runs.
        timeout_s: Maximum elapsed seconds for runtime startup and device login.

    Returns:
        Whether the Harness confirmed that it stored the new grant.

    Raises:
        ValueError: If the timeout is not positive.
        RuntimeError: If the runtime exits or cannot complete authorization.
        TimeoutError: If authorization does not finish before the deadline.
    """
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("--timeout-s must be a finite positive number")
    from deepseek_harness_runtime import resolve_bundled_launch_args

    dsh_home = dsh_home.expanduser().resolve()
    dsh_home.mkdir(parents=True, mode=0o700, exist_ok=True)
    plugin = Path(__file__).resolve().parents[1] / "planner" / "deepseek_login.mjs"
    env = os.environ.copy()
    env["DSH_HOME"] = str(dsh_home)
    env["DSH_PRIMARY_RUNTIME"] = ""
    with tempfile.TemporaryDirectory(prefix="rpent-dsh-login-") as temporary:
        patch = Path(temporary) / "login.patch.json"
        patch.write_text(
            json.dumps([{"insert": [{"id": "rpent-login", "name": str(plugin)}]}])
        )
        process = subprocess.Popen(
            [*resolve_bundled_launch_args(), "--profile", "sdk", "--patch", str(patch)],
            cwd=temporary,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        events: queue.Queue[str | None] = queue.Queue()
        assert process.stderr is not None
        reader = threading.Thread(
            target=_read_events,
            args=(process.stderr, events),
            name="dsh-login-stderr",
            daemon=True,
        )
        reader.start()
        deadline = time.monotonic() + timeout_s
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("DeepSeek Harness login timed out")
                try:
                    line = events.get(timeout=min(remaining, 0.5))
                except queue.Empty:
                    continue
                if line is None:
                    raise RuntimeError("DeepSeek Harness exited before authorization")
                event = json.loads(line)
                if event.get("event") == "device_code":
                    print(f"Open {event['url']}", flush=True)
                    print(f"Enter code: {event['code']}", flush=True)
                elif event.get("event") == "settled":
                    return event.get("status") == "authorized"
                elif event.get("event") == "error":
                    raise RuntimeError(
                        "DeepSeek Harness authorization failed "
                        f"({event.get('name')}: {event.get('code')})"
                    )
        finally:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            if process.stdin is not None:
                process.stdin.close()
            reader.join(timeout=1)
            process.stderr.close()


def main() -> int:
    """Run the ``rpent-dsh-login`` console command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dsh-home",
        default=os.environ.get("DSH_HOME"),
        help="Harness credential home; defaults to DSH_HOME and must be explicit",
    )
    parser.add_argument("--timeout-s", type=float, default=960)
    args = parser.parse_args()
    if not args.dsh_home or not args.dsh_home.strip():
        parser.error("pass --dsh-home or set DSH_HOME")
    try:
        authorized = login(Path(args.dsh_home), timeout_s=args.timeout_s)
    except ImportError:
        parser.exit(
            1,
            'Install the Harness extra first: uv pip install -e ".[deepseek-harness]"\n',
        )
    except KeyboardInterrupt:
        parser.exit(130, "Login interrupted.\n")
    except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
    if not authorized:
        parser.exit(1, "Authorization cancelled.\n")
    print("Authorized. Use the same --dsh-home for RPent planner runs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
