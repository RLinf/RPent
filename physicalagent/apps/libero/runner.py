"""Hybrid LLM-in-the-loop agent — Anthropic API runner.

Drives interactive_driver.py through tool calls. Supports:
- Starting the driver as a subprocess (or attaching to an existing one)
- Multi-turn Claude conversation with vision + tool use
- Per-turn token usage logging
- Conversation transcript persisted at the end

Use as a script (see __main__ at the bottom) or import `run_one_cell`.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

# Auto-detect project paths: this file is at
# <repo>/physicalagent/apps/libero/runner.py
from physicalagent.config import (
    get_repo_root,
    get_python_bin,
    get_repl_driver_script,
    get_default_workdir_prefix,
    get_cuda_device,
    get_libero_type,
    get_anthropic_api_key,
    get_anthropic_base_url,
    get_anthropic_model,
    get_memory_dir,
)

REPO_ROOT = get_repo_root()
DEFAULT_DRIVER_CMD = get_python_bin()
DEFAULT_DRIVER_SCRIPT = str(get_repl_driver_script())

from physicalagent.tools.repl import (  # noqa: E402
    execute_tool,
    get_tools_spec,
    tool_result_to_content_blocks,
    set_workdir as tools_set_workdir,
)
from physicalagent.context.libero_prompts import (  # noqa: E402
    CLAUDE_CODE_PERCEPTION_PROMPT_TEMPLATE,
    CLAUDE_CODE_PROMPT_TEMPLATE,
    INITIAL_USER_TEMPLATE,
    PERCEPTION_PREFIX,
    PERCEPTION_USER_TEMPLATE,
    SYSTEM_PROMPT,
    format_claude_code_prompt,
)
from physicalagent.logging import make_log_dir  # noqa: E402
from physicalagent.cerebrum.anthropic import AnthropicCerebrum  # noqa: E402
from physicalagent.cerebrum.claude_code import ClaudeCodeCerebrum  # noqa: E402


def start_driver(
    suite: str,
    task: int,
    seed: int,
    workdir: str | None = None,
    max_episode_steps: int = 600,
    libero_type: str | None = None,
    cuda_device: str | None = None,
    log_path: str | None = None,
    python_bin: str | None = None,
    driver_script: str | None = None,
    ready_timeout_s: float = 300.0,
    perception: bool = False,
) -> subprocess.Popen:
    """Clear workdir and launch the REPL driver in background.

    Returns the Popen handle; waits until state_00.json appears.
    """
    wd = Path(workdir or get_default_workdir_prefix())
    if wd.exists():
        shutil.rmtree(wd)
    wd.mkdir(parents=True, exist_ok=True)

    if log_path is None:
        log_path = str(wd.parent / f"{wd.name}_driver.log")

    env = os.environ.copy()
    env["LIBERO_TYPE"] = libero_type or get_libero_type()
    env["CUDA_VISIBLE_DEVICES"] = str(cuda_device or get_cuda_device())
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("ROBOT_PLATFORM", "LIBERO")

    cmd = [
        python_bin or get_python_bin(),
        driver_script or str(get_repl_driver_script()),
        "--suite", suite,
        "--task", str(task),
        "--seed", str(seed),
        "--max_episode_steps", str(max_episode_steps),
        "--workdir", str(wd),
    ]
    if perception:
        cmd += ["--hide_object_coords", "--always_render"]
        cmd += ["--video_path", str(wd / "episode.mp4")]
    print(f"[agent] driver cmd: {' '.join(cmd)}")
    print(f"[agent] driver log: {log_path}")
    print(f"[agent] CUDA_VISIBLE_DEVICES={cuda_device}  workdir={wd}")
    log_f = open(log_path, "w")
    proc = subprocess.Popen(
        cmd,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=str(REPO_ROOT),
    )

    print(f"[agent] waiting for state_00.json (Pi0 load ~80s)...")
    t0 = time.time()
    while not (wd / "state_00.json").exists():
        time.sleep(2)
        if proc.poll() is not None:
            print(f"[agent] driver EXITED before becoming ready. Last log:")
            print(Path(log_path).read_text()[-2000:])
            raise RuntimeError("driver exited prematurely")
        if time.time() - t0 > ready_timeout_s:
            proc.terminate()
            raise RuntimeError(f"driver not ready after {ready_timeout_s}s")
    print(f"[agent] driver ready in {time.time()-t0:.1f}s")
    return proc


def stop_driver(proc: subprocess.Popen, workdir: str | None = None, timeout: float = 15.0) -> None:
    if proc.poll() is not None:
        return
    cmd_path = Path(workdir or get_default_workdir_prefix()) / "command.json"
    try:
        with open(cmd_path, "w") as f:
            json.dump({"action": "exit"}, f)
    except Exception:
        pass
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


# ---------------------------------------------------------------------------
# Anthropic API agent loop
# ---------------------------------------------------------------------------


def _serialize_messages(messages: list[dict]) -> list[dict]:
    """Convert SDK objects in `content` to plain dicts so they're JSON-safe."""
    out = []
    for m in messages:
        c = m["content"]
        if isinstance(c, str):
            out.append({"role": m["role"], "content": c})
            continue
        new_blocks = []
        for b in c:
            if isinstance(b, dict):
                # Don't write large base64 images to disk — replace with a stub
                if b.get("type") == "image":
                    new_blocks.append({"type": "image", "source": {"_omitted_for_transcript": True}})
                else:
                    new_blocks.append(b)
                continue
            bd: dict = {"type": getattr(b, "type", "?")}
            for attr in ("text", "name", "input", "id"):
                if hasattr(b, attr):
                    bd[attr] = getattr(b, attr)
            new_blocks.append(bd)
        out.append({"role": m["role"], "content": new_blocks})
    return out


# ---------------------------------------------------------------------------
# Emergency save (so a successful sim run isn't lost on agent crash)
# ---------------------------------------------------------------------------


def _emergency_save(workdir, output_dir, suite, task, seed, recipe_tag,
                    agent_error, regime="strict", verbose=True):
    """If the workdir has libero_terminated=True in any state file and the
    output_dir is missing the recipe.jsonl / audit.json, stitch them now
    from logs in the workdir. Idempotent — won't overwrite existing files.
    """
    wd = Path(workdir)
    out = Path(output_dir)
    if not wd.exists():
        return
    recipe_path = out / f"recipe_{recipe_tag}.jsonl"
    audit_path = out / f"{recipe_tag}.json"
    if recipe_path.exists() and audit_path.exists():
        return  # agent already saved

    # Did the sim ever fire libero_terminated=True?
    sim_terminated = False
    states = {}
    for sp in sorted(wd.glob("state_*.json")):
        try:
            sn = int(sp.stem.split("_")[1])
            d = json.load(open(sp))
            states[sn] = d
            if d.get("libero_terminated"):
                sim_terminated = True
        except Exception:
            continue
    logs = {}
    for lp in sorted(wd.glob("log_*.json")):
        try:
            ln = int(lp.stem.split("_")[1])
            logs[ln] = json.load(open(lp))
        except Exception:
            continue

    if not states and not logs:
        return  # nothing to salvage

    # Always rebuild a recipe.jsonl from the commands actually executed
    if not recipe_path.exists() and logs:
        recipe_lines = []
        for ln in sorted(logs.keys()):
            cmd = logs[ln].get("command") or {}
            if cmd.get("action") in ("exit", "reset"):
                continue
            recipe_lines.append(json.dumps(cmd))
        if recipe_lines:
            out.mkdir(parents=True, exist_ok=True)
            recipe_path.write_text("\n".join(recipe_lines) + "\n")
            if verbose:
                print(f"[agent] [emergency_save] wrote {recipe_path} ({len(recipe_lines)} cmds)")

    # Build a minimal audit (PRO schema) if missing
    if not audit_path.exists():
        pick_step = next(
            (n for n in sorted(logs) if (logs[n].get("command") or {}).get("action") == "pi0_pick"),
            None,
        )
        release_step = next(
            (n for n in reversed(sorted(logs)) if (logs[n].get("command") or {}).get("action") == "release"),
            None,
        )
        move_steps = [n for n in sorted(logs) if (logs[n].get("command") or {}).get("action") == "move_to"]
        last_state = states[max(states)] if states else {}
        record = {
            "suite": suite,
            "task_id": task,
            "seed": seed,
            "regime": regime,
            "strategy_notes": (
                f"emergency-saved by runner after agent error: {agent_error}"
                if agent_error else "emergency-saved by runner (agent did not call finish)"
            ),
            "pick_result":     logs[pick_step]["result"]   if pick_step    is not None else None,
            "post_pick_state": states[pick_step]["state"]  if pick_step    is not None and pick_step in states else None,
            "move_results":    [logs[n]["result"] for n in move_steps],
            "release_result":  logs[release_step]["result"] if release_step is not None else None,
            "final_state":     last_state.get("state"),
            "libero_terminated": bool(last_state.get("libero_terminated")),
            "sim_reached_terminated": sim_terminated,
            "agent_error": agent_error,
        }
        out.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(json.dumps(record, indent=2, default=str))
        if verbose:
            print(f"[agent] [emergency_save] wrote {audit_path} "
                  f"(libero_terminated={record['libero_terminated']})")


# ---------------------------------------------------------------------------
# High-level entrypoint
# ---------------------------------------------------------------------------


def run_one_cell(
    suite: str,
    task: int,
    seed: int,
    *,
    api_key: str | None = None,
    model: str | None = None,
    max_turns: int = 80,
    max_tokens: int = 4096,
    max_episode_steps: int = 600,
    cuda_device: str | None = None,
    output_dir: str | None = None,
    no_driver: bool = False,
    verbose: bool = True,
    base_url: str | None = None,
    workdir: str | None = None,
    perception: bool = False,
    libero_type: str | None = None,
    cerebrum_type: str = "anthropic",
    claude_code_timeout_s: int | None = None,
    claude_code_max_budget_usd: float | None = None,
) -> dict:
    """Solve one (suite, task, seed) cell end-to-end.

    By default the REPL workdir is placed inside the log directory
    (``output_dir/repl/``) so images, depth arrays, state snapshots,
    and the episode video land there directly — no post-hoc copy.
    Pass an explicit ``workdir`` to override (e.g. for parallel runs
    that share a single output directory).

    ``cerebrum_type`` selects the LLM backend:
    - ``"anthropic"`` — Anthropic Messages API with tool-use (default).
    - ``"claude_code"`` — delegates to ``claude -p`` (Claude Code).
    """
    import anthropic

    if max_episode_steps == 600 and "libero_10" in suite:
        max_episode_steps = 5000
        if verbose:
            print("[agent] auto-bumped max_episode_steps to 5000 for libero_10")

    # ---- resolve output directory early so the workdir can live inside it ----
    if output_dir is None:
        output_dir = str(make_log_dir(suite=suite, task=task, seed=seed, repo_root=REPO_ROOT))
    else:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ---- resolve workdir ----
    if workdir is None:
        workdir = str(Path(output_dir) / "repl")
        Path(workdir).mkdir(parents=True, exist_ok=True)
    else:
        Path(workdir).mkdir(parents=True, exist_ok=True)

    # Point the agent's tools at the per-run workdir BEFORE the loop starts.
    tools_set_workdir(workdir)

    if cerebrum_type == "anthropic":
        api_key = api_key or get_anthropic_api_key()
        if not api_key:
            raise ValueError("api_key is required for anthropic cerebrum")
        import anthropic
        client = anthropic.Anthropic(
            api_key=api_key,
            max_retries=8,
            timeout=120.0,
            **({"base_url": base_url} if base_url else {}),
        )
        cerebrum = AnthropicCerebrum(
            client=client,
            model=model or get_anthropic_model(),
            max_tokens=max_tokens,
        )
    elif cerebrum_type == "claude_code":
        cc_timeout_s = claude_code_timeout_s
        if cc_timeout_s is None:
            cc_timeout_s = int(os.environ.get("CELL_TIMEOUT_S", "1200" if perception else "600"))
        cc_budget = claude_code_max_budget_usd
        if cc_budget is None:
            cc_budget = float(os.environ.get("MAX_BUDGET_USD", "10"))
        cc_output_path = Path(output_dir) / f"claude_{suite.replace('libero_', '')}_t{task}_s{seed}.txt"
        cerebrum = ClaudeCodeCerebrum(
            workdir=workdir,
            repo_root=REPO_ROOT,
            model=model or "sonnet",
            timeout_s=cc_timeout_s,
            max_budget_usd=cc_budget,
            extra_dirs=[str(get_memory_dir())],
            output_path=cc_output_path,
        )
    else:
        raise ValueError(f"unknown cerebrum_type: {cerebrum_type}")

    # Auto-route LIBERO_TYPE if not set
    if libero_type is None:
        if any(suite.endswith(s) for s in ("_swap", "_task", "_lan")):
            libero_type = "pro"
        else:
            libero_type = "standard"

    recipe_tag = f"{suite.replace('libero_', '')}_t{task}_s{seed}"
    regime = "strict_perception" if perception else "strict"

    if cerebrum_type == "claude_code":
        # Claude Code gets the full legacy single-shot prompt.  It interacts
        # through Bash/Read/Write directly, not Anthropic API tool schemas.
        system_prompt = ""
        template = (
            CLAUDE_CODE_PERCEPTION_PROMPT_TEMPLATE
            if perception
            else CLAUDE_CODE_PROMPT_TEMPLATE
        )
        user_msg = format_claude_code_prompt(
            template,
            suite=suite, task=task, seed=seed,
            output_dir=output_dir, recipe_tag=recipe_tag,
            workdir=workdir,
        )
    elif perception:
        user_msg = PERCEPTION_USER_TEMPLATE.format(
            suite=suite, task=task, seed=seed,
            output_dir=output_dir, recipe_tag=recipe_tag,
            workdir=workdir,
        )
        system_prompt = PERCEPTION_PREFIX + SYSTEM_PROMPT
    else:
        user_msg = INITIAL_USER_TEMPLATE.format(
            suite=suite, task=task, seed=seed,
            output_dir=output_dir, recipe_tag=recipe_tag,
            workdir=workdir,
        )
        system_prompt = SYSTEM_PROMPT

    proc = None
    if not no_driver:
        proc = start_driver(
            suite=suite, task=task, seed=seed,
            workdir=workdir,
            max_episode_steps=max_episode_steps,
            cuda_device=cuda_device,
            libero_type=libero_type,
            perception=perception,
        )
        if cerebrum_type == "claude_code":
            cerebrum.set_driver_process(proc)
    else:
        if not (Path(workdir) / "state_00.json").exists():
            raise RuntimeError(f"--no_driver but {workdir}/state_00.json missing")

    t0 = time.time()
    finish_result, messages, agent_error = None, [], None
    stats: dict = {}
    try:
        result = cerebrum.solve(
            system_prompt=system_prompt,
            user_message=user_msg,
            tools_spec=get_tools_spec(),
            tool_handler=execute_tool,
            tool_result_formatter=tool_result_to_content_blocks,
            max_turns=max_turns,
            verbose=verbose,
        )
        finish_result = result.finish_result
        messages = result.messages
        stats = result.stats
        agent_error = result.error
    except Exception as e:
        agent_error = f"{type(e).__name__}: {e}"
        if verbose:
            print(f"[agent] EXCEPTION in agent loop: {agent_error}")
    finally:
        # Salvage: if the sim reached libero_terminated=True before the
        # agent crashed (or before it called finish), still write a
        # minimal recipe + audit so the run isn't lost.
        try:
            _emergency_save(workdir, output_dir, suite, task, seed, recipe_tag,
                            agent_error, regime=regime, verbose=verbose)
        except Exception as e:
            if verbose:
                print(f"[agent] emergency save failed: {e}")
        if proc is not None:
            stop_driver(proc, workdir=workdir)

    elapsed = time.time() - t0

    transcript_path = Path(output_dir) / f"transcript_{recipe_tag}.json"
    record = {
        "suite": suite, "task": task, "seed": seed, "model": model,
        "elapsed_s": round(elapsed, 1),
        "finish": finish_result,
        "stats": stats,
        "messages": _serialize_messages(messages),
    }
    with open(transcript_path, "w") as f:
        json.dump(record, f, indent=2, default=str)
    if verbose:
        print(f"\n[agent] elapsed: {elapsed:.1f}s")
        print(f"[agent] usage: in={stats.get('total_input_tokens', '?')} "
              f"out={stats.get('total_output_tokens', '?')} "
              f"tool_calls={stats.get('tool_calls', '?')}")
        print(f"[agent] transcript: {transcript_path}")
        if agent_error:
            print(f"[agent] error: {agent_error}")
    return record


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Standalone hybrid LLM-in-the-loop agent for LIBERO PRO",
    )
    ap.add_argument("--suite", required=True,
                    help="e.g. libero_object_task, libero_spatial_swap")
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=None,
                    help="Model id. Defaults to ANTHROPIC_MODEL env or claude-sonnet-4-5.")
    ap.add_argument("--max_turns", type=int, default=80)
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--max_episode_steps", type=int, default=600)
    ap.add_argument("--cuda_device", default=None,
                    help="GPU device. Defaults to CUDA_DEVICE env or 0.")
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--api_key", default=None,
                    help="Anthropic API key (required for --cerebrum anthropic). "
                         "Defaults to ANTHROPIC_API_KEY env var.")
    ap.add_argument("--base_url", default=None,
                    help="Override Anthropic base URL. "
                         "Defaults to ANTHROPIC_BASE_URL env var.")
    ap.add_argument("--cerebrum", default="anthropic",
                    choices=["anthropic", "claude_code"],
                    help="LLM backend: anthropic (API tool-use) | claude_code (claude -p CLI).")
    ap.add_argument("--claude_code_timeout_s", type=int, default=None,
                    help="Wall-clock cap for claude -p. Defaults to CELL_TIMEOUT_S, "
                         "or 1200 in --perception mode / 600 otherwise.")
    ap.add_argument("--claude_code_max_budget_usd", type=float, default=None,
                    help="Budget passed to claude -p --max-budget-usd. "
                         "Defaults to MAX_BUDGET_USD env or 10.")
    ap.add_argument("--no_driver", action="store_true",
                    help="Don't spawn driver; attach to existing workdir")
    ap.add_argument("--workdir", default=None,
                    help="REPL working directory. Default: <output_dir>/repl/. "
                         "Override for parallel runs that share an output dir.")
    ap.add_argument("--perception", action="store_true",
                    help="PERCEPTION-ISOLATED mode: hide object coords, "
                         "use camera+depth+back_project for localization.")
    ap.add_argument("--libero_type", default=None,
                    choices=["standard", "pro", "plus"],
                    help="LIBERO variant (auto-routed from suite suffix if not set).")
    ap.add_argument("--quiet", action="store_true")
    return ap


def main() -> int:
    ap = _build_argparser()
    args = ap.parse_args()

    api_key = args.api_key or get_anthropic_api_key()
    if args.cerebrum == "anthropic" and not api_key:
        print("ERROR: set ANTHROPIC_API_KEY env var or pass --api_key", file=sys.stderr)
        return 2

    run_one_cell(
        suite=args.suite, task=args.task, seed=args.seed,
        api_key=api_key,
        model=args.model,
        max_turns=args.max_turns,
        max_tokens=args.max_tokens,
        max_episode_steps=args.max_episode_steps,
        cuda_device=args.cuda_device,
        output_dir=args.output_dir,
        no_driver=args.no_driver,
        verbose=not args.quiet,
        base_url=args.base_url or get_anthropic_base_url(),
        workdir=args.workdir,
        perception=args.perception,
        libero_type=args.libero_type,
        cerebrum_type=args.cerebrum,
        claude_code_timeout_s=args.claude_code_timeout_s,
        claude_code_max_budget_usd=args.claude_code_max_budget_usd,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
