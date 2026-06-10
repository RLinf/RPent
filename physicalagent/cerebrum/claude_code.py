"""Claude Code cerebrum — delegates the agent loop to `claude -p`.

Claude Code interacts directly with the REPL workdir filesystem (Bash,
Read, Write, Grep, Glob).  This cerebrum writes a combined task prompt,
spawns ``claude -p`` with directory access, and waits for completion.
"""
from __future__ import annotations

import os
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable

from physicalagent.cerebrum.base import CerebrumResult
from physicalagent.config import get_repo_root


class ClaudeCodeCerebrum:
    """Cerebrum backed by the ``claude`` CLI (Claude Code subscription).

    Constructor parameters
    ----------------------
    workdir:
        REPL working directory (granted to Claude Code via ``--add-dir``).
    repo_root:
        Repository root (used as ``cwd`` for the subprocess so relative
        paths in the prompt resolve correctly).
    model:
        Claude model id passed to ``--model`` (default ``"sonnet"``).
    allowed_tools:
        Space-separated tool list for ``--allowedTools``.
    timeout_s:
        Hard wall-clock cap on the ``claude -p`` subprocess.
    max_budget_usd:
        Passed to ``--max-budget-usd``.
    extra_dirs:
        Additional ``--add-dir`` paths (e.g. the memory snapshot).
    output_path:
        Optional path for full ``claude -p`` stdout/stderr, matching the
        legacy ``claude_<tag>.txt`` artifact.
    driver_pid:
        Optional REPL driver PID.  If it dies while Claude is still polling
        for done flags, Claude is terminated early instead of waiting for the
        full wall-clock timeout.
    """

    def __init__(
        self,
        *,
        workdir: str,
        repo_root: str | Path | None = None,
        model: str = "sonnet",
        allowed_tools: str = "Bash Read Write Glob Grep",
        timeout_s: int = 600,
        max_budget_usd: float = 10.0,
        extra_dirs: list[str] | None = None,
        output_path: str | Path | None = None,
        driver_pid: int | None = None,
    ):
        self._workdir = str(workdir)
        self._repo_root = str(repo_root) if repo_root else str(get_repo_root())
        self._model = model
        self._allowed_tools = allowed_tools
        self._timeout_s = timeout_s
        self._max_budget_usd = max_budget_usd
        self._extra_dirs = extra_dirs or []
        self._output_path = Path(output_path) if output_path else None
        self._driver_pid = driver_pid
        self._driver_proc: subprocess.Popen | None = None

    def set_driver_pid(self, pid: int | None) -> None:
        """Attach the REPL driver PID after the runner starts it."""
        self._driver_pid = pid

    def set_driver_process(self, proc: subprocess.Popen | None) -> None:
        """Attach the REPL driver process for reliable death detection."""
        self._driver_proc = proc
        self._driver_pid = proc.pid if proc is not None else None

    # ------------------------------------------------------------------
    # Cerebrum protocol
    # ------------------------------------------------------------------

    def solve(
        self,
        *,
        system_prompt: str,
        user_message: str,
        tools_spec: list[dict[str, Any]] | None = None,
        tool_handler: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
        tool_result_formatter: Callable[[dict[str, Any]], list[dict[str, Any]]] | None = None,
        max_turns: int = 80,
        verbose: bool = True,
    ) -> CerebrumResult:
        """Run ``claude -p`` with the combined system+user prompt.

        ``tools_spec``, ``tool_handler``, and ``tool_result_formatter`` are
        accepted for protocol compatibility but **ignored** — Claude Code
        uses its own built-in tool set (Bash, Read, Write, Grep, Glob).
        """
        # Build the combined task prompt.  The Claude Code prompt is often a
        # self-contained legacy prompt, so avoid a leading blank when there is
        # no separate system prompt.
        full_prompt = (
            f"{system_prompt}\n\n{user_message}" if system_prompt else user_message
        )

        # Write prompt to a temp file so it can be passed to claude -p.
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", prefix="cc_task_", delete=False
        ) as f:
            f.write(full_prompt)
            prompt_file = f.name

        try:
            if verbose:
                print(f"[cc-cerebrum] prompt: {len(full_prompt)} chars → {prompt_file}")
                print(f"[cc-cerebrum] workdir: {self._workdir}")
                print(f"[cc-cerebrum] invoking claude -p --model {self._model} "
                      f"(timeout={self._timeout_s}s, budget=${self._max_budget_usd})")

            cmd = [
                "claude", "-p",
                full_prompt,               # stdin works too, but explicit is clearer
                "--model", self._model,
                "--output-format", "text",
                "--add-dir", self._workdir,
                "--allowedTools", self._allowed_tools,
                "--max-budget-usd", str(self._max_budget_usd),
            ]
            for d in self._extra_dirs:
                cmd += ["--add-dir", d]

            output_path = self._output_path or Path(prompt_file).with_suffix(".out")
            output_path.parent.mkdir(parents=True, exist_ok=True)

            t0 = time.time()
            timed_out = False
            killed_by_watchdog = False
            with open(output_path, "w") as out_f:
                proc = subprocess.Popen(
                    cmd,
                    stdout=out_f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=self._repo_root,
                    env={**os.environ},  # inherit API key / base URL from env
                    start_new_session=True,
                )

                stop_watch = threading.Event()

                def _watch_driver() -> None:
                    nonlocal killed_by_watchdog
                    if self._driver_pid is None and self._driver_proc is None:
                        return
                    while not stop_watch.wait(5.0):
                        if proc.poll() is not None:
                            return
                        driver_alive = (
                            self._driver_proc.poll() is None
                            if self._driver_proc is not None
                            else _pid_alive(self._driver_pid)
                        )
                        if driver_alive:
                            continue
                        # Give the driver a moment to flush its traceback, then
                        # kill Claude and any Bash poll loop it spawned.
                        stop_watch.wait(10.0)
                        if proc.poll() is None:
                            killed_by_watchdog = True
                            msg = (
                                f"\n[cc-cerebrum] WATCHDOG: driver pid "
                                f"{self._driver_pid} died mid-run; killing "
                                "claude -p worker.\n"
                            )
                            out_f.write(msg)
                            out_f.flush()
                            _terminate_process_group(proc)
                        return

                watch_thread = threading.Thread(target=_watch_driver, daemon=True)
                watch_thread.start()
                try:
                    proc.wait(timeout=self._timeout_s)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    out_f.write(
                        f"\n[cc-cerebrum] TIMEOUT after {self._timeout_s}s; "
                        "killing claude -p worker.\n"
                    )
                    out_f.flush()
                    _terminate_process_group(proc)
                    proc.wait(timeout=15)
                finally:
                    stop_watch.set()
                    watch_thread.join(timeout=1.0)

            elapsed = time.time() - t0
            stdout_text = output_path.read_text(errors="replace")
            returncode = proc.returncode

            if verbose:
                print(f"[cc-cerebrum] claude -p finished in {elapsed:.1f}s "
                      f"rc={returncode}")
                print(f"[cc-cerebrum] output: {output_path}")
                # Print last few lines of output for diagnostics
                tail = stdout_text.strip().splitlines()[-10:]
                if tail:
                    print("[cc-cerebrum] last 10 lines of claude output:")
                    for line in tail:
                        print(f"  | {line}")

            error = None
            if timed_out:
                error = f"claude -p timed out after {self._timeout_s}s"
            elif killed_by_watchdog:
                error = "claude -p killed because the REPL driver died"
            elif returncode != 0:
                error = f"claude -p exited with rc={returncode}: {stdout_text[-500:]}"

            return CerebrumResult(
                finish_result=None,  # Can't easily parse from Claude Code output
                messages=[{"role": "claude_code", "content": stdout_text}],
                stats={
                    "elapsed_s": round(elapsed, 1),
                    "returncode": returncode,
                    "output_chars": len(stdout_text),
                    "output_path": str(output_path),
                },
                error=error,
            )
        except subprocess.TimeoutExpired:
            return CerebrumResult(
                error=f"claude -p timed out after {self._timeout_s}s",
                stats={"elapsed_s": self._timeout_s},
            )
        finally:
            # Clean up temp prompt file.
            try:
                os.unlink(prompt_file)
            except OSError:
                pass


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _terminate_process_group(proc: subprocess.Popen) -> None:
    """Terminate Claude and tool subprocesses spawned in its process group."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError:
        proc.terminate()
        return

    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            proc.kill()