"""Claude Code cerebrum — delegates the agent loop to `claude -p`.

Claude Code interacts directly with the REPL workdir filesystem (Bash,
Read, Write, Grep, Glob).  This cerebrum writes a combined task prompt,
spawns ``claude -p`` with directory access, and waits for completion.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
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
    ):
        self._workdir = str(workdir)
        self._repo_root = str(repo_root) if repo_root else str(get_repo_root())
        self._model = model
        self._allowed_tools = allowed_tools
        self._timeout_s = timeout_s
        self._max_budget_usd = max_budget_usd
        self._extra_dirs = extra_dirs or []

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
        # Build the combined task prompt.
        full_prompt = system_prompt + "\n\n" + user_message

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

            t0 = time.time()
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._timeout_s,
                cwd=self._repo_root,
                env={**os.environ},  # inherit API key / base URL from env
            )
            elapsed = time.time() - t0

            if verbose:
                print(f"[cc-cerebrum] claude -p finished in {elapsed:.1f}s "
                      f"rc={proc.returncode}")
                # Print last few lines of output for diagnostics
                tail = proc.stdout.strip().splitlines()[-10:]
                if tail:
                    print("[cc-cerebrum] last 10 lines of claude output:")
                    for line in tail:
                        print(f"  | {line}")

            error = None
            if proc.returncode == 124:
                error = f"claude -p timed out after {self._timeout_s}s"
            elif proc.returncode != 0:
                error = f"claude -p exited with rc={proc.returncode}: {proc.stderr[-500:]}"

            return CerebrumResult(
                finish_result=None,  # Can't easily parse from Claude Code output
                messages=[{"role": "claude_code", "content": proc.stdout}],
                stats={
                    "elapsed_s": round(elapsed, 1),
                    "returncode": proc.returncode,
                    "output_chars": len(proc.stdout),
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