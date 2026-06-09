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
# <repo>/physicalagent/primitives/workspace_pro/hybrid_agent/runner.py
PROJECT_ROOT = Path(__file__).resolve().parents[4]
PRIMITIVES_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORKDIR = "/tmp/hybrid_repl"

# Make tools importable when run as a script
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import tools as tools_mod  # noqa: E402
from tools import (  # noqa: E402
    TOOLS_SPEC,
    execute_tool,
    tool_result_to_content_blocks,
)
from prompts import (  # noqa: E402
    SYSTEM_PROMPT, INITIAL_USER_TEMPLATE,
    PERCEPTION_PREFIX, PERCEPTION_USER_TEMPLATE,
)


# ---------------------------------------------------------------------------
# Driver lifecycle
# ---------------------------------------------------------------------------

DEFAULT_DRIVER_CMD = os.environ.get("PYTHON_BIN", "/opt/venv/openpi/bin/python")
DEFAULT_DRIVER_SCRIPT = str(PRIMITIVES_ROOT / "interactive_driver.py")


def start_driver(
    suite: str,
    task: int,
    seed: int,
    workdir: str = DEFAULT_WORKDIR,
    max_episode_steps: int = 600,
    libero_type: str = "pro",
    cuda_device: str = "0",
    log_path: str | None = None,
    python_bin: str = DEFAULT_DRIVER_CMD,
    driver_script: str = DEFAULT_DRIVER_SCRIPT,
    ready_timeout_s: float = 300.0,
    perception: bool = False,
) -> subprocess.Popen:
    """Clear workdir and launch interactive_driver.py in background.

    Returns the Popen handle; waits until state_00.json appears.
    """
    wd = Path(workdir)
    if wd.exists():
        shutil.rmtree(wd)
    wd.mkdir(parents=True, exist_ok=True)

    if log_path is None:
        log_path = str(wd.parent / f"{wd.name}_driver.log")

    env = os.environ.copy()
    env["LIBERO_TYPE"] = libero_type
    env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("ROBOT_PLATFORM", "LIBERO")

    cmd = [
        python_bin,
        driver_script,
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
        cwd=str(PROJECT_ROOT),
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


def stop_driver(proc: subprocess.Popen, workdir: str = DEFAULT_WORKDIR, timeout: float = 15.0) -> None:
    if proc.poll() is not None:
        return
    cmd_path = Path(workdir) / "command.json"
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


def _short_repr(obj, maxlen=200):
    s = json.dumps(obj, default=str) if not isinstance(obj, str) else obj
    return s if len(s) <= maxlen else s[:maxlen] + "...(+%d)" % (len(s) - maxlen)


def run_agent_loop(
    client,
    model: str,
    system: str,
    user_msg: str,
    max_turns: int = 80,
    max_tokens: int = 4096,
    verbose: bool = True,
):
    """Single-cell LLM-in-the-loop. Returns (finish_result_or_None, messages, stats)."""
    messages: list[dict] = [{"role": "user", "content": user_msg}]
    finish_result = None
    total_in = total_out = 0
    n_tool_calls = 0

    import anthropic  # for retryable exception classes
    for turn in range(1, max_turns + 1):
        if verbose:
            print(f"\n[agent] === turn {turn}/{max_turns} ===")

        # Outer retry on top of SDK's built-in: handles longer outages
        # (proxy down for 30+s) by sleeping then retrying once more.
        response = None
        last_err = None
        for outer in range(3):
            try:
                response = client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    system=system,
                    tools=TOOLS_SPEC,
                    messages=messages,
                )
                break
            except (anthropic.APIConnectionError, anthropic.APITimeoutError, anthropic.InternalServerError, anthropic.RateLimitError) as e:
                last_err = e
                wait = 10 * (outer + 1)
                if verbose:
                    print(f"[agent] API error '{type(e).__name__}: {e}' — sleeping {wait}s and retrying (outer {outer+1}/3)")
                time.sleep(wait)
        if response is None:
            if verbose:
                print(f"[agent] giving up after 3 outer retries; last error: {last_err}")
            break
        u = response.usage
        total_in += u.input_tokens
        total_out += u.output_tokens

        if verbose:
            for block in response.content:
                if block.type == "text" and block.text.strip():
                    print(f"[claude] {block.text.strip()}")
                elif block.type == "tool_use":
                    print(f"[tool→] {block.name}({_short_repr(block.input, 250)})")
            print(f"[usage] in={u.input_tokens}  out={u.output_tokens}  "
                  f"stop={response.stop_reason}  total_in={total_in}  total_out={total_out}")

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                n_tool_calls += 1
                result = execute_tool(block.name, block.input)
                if isinstance(result, dict) and result.get("_finish"):
                    finish_result = result
                if verbose:
                    summary = {k: v for k, v in result.items() if k not in ("state", "content", "log", "_image_path")} \
                        if isinstance(result, dict) else result
                    if isinstance(result, dict) and "state" in result:
                        s = result["state"]
                        summary["state_summary"] = {
                            "eef": [round(x, 3) for x in s.get("robot0_eef_pos", [])][:3],
                            "libero_terminated": result.get("libero_terminated"),
                        }
                    if isinstance(result, dict) and "log" in result:
                        lg = result["log"]
                        if isinstance(lg, dict) and "result" in lg:
                            summary["log_result_keys"] = list(lg["result"].keys())
                    print(f"[tool←] {block.name}: {_short_repr(summary, 350)}")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": tool_result_to_content_blocks(result),
                })
            messages.append({"role": "user", "content": tool_results})

            if finish_result is not None:
                if verbose:
                    print(f"\n[agent] FINISH called: {finish_result}")
                break
        elif response.stop_reason == "end_turn":
            if verbose:
                print("[agent] Claude ended turn without a tool call. Stopping.")
            break
        else:
            if verbose:
                print(f"[agent] unexpected stop_reason: {response.stop_reason}")
            break

    stats = {
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "turns_used": turn,
        "tool_calls": n_tool_calls,
    }
    return finish_result, messages, stats


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
    api_key: str,
    model: str = "claude-sonnet-4-5",
    max_turns: int = 80,
    max_tokens: int = 4096,
    max_episode_steps: int = 600,
    cuda_device: str = "0",
    output_dir: str | None = None,
    no_driver: bool = False,
    verbose: bool = True,
    base_url: str | None = None,
    workdir: str = DEFAULT_WORKDIR,
    perception: bool = False,
    libero_type: str | None = None,
) -> dict:
    """Solve one (suite, task, seed) cell end-to-end.

    `workdir` is the REPL working directory used by both the driver
    subprocess and the agent's tools. Override to run multiple cells in
    parallel (e.g. /tmp/hybrid_repl_s0, /tmp/hybrid_repl_s1, ...).
    """
    import anthropic
    kwargs = {
        "api_key": api_key,
        "max_retries": 8,    # SDK does exp-backoff on 429 / 5xx / ConnectionError
        "timeout": 120.0,    # per-request socket timeout
    }
    if base_url:
        kwargs["base_url"] = base_url
    client = anthropic.Anthropic(**kwargs)

    # Point the agent's tools at the per-run workdir BEFORE the loop starts.
    tools_mod.set_workdir(workdir)

    if output_dir is None:
        output_dir = str(PRIMITIVES_ROOT / "workspace_pro" / "results_agent_runs")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Auto-route LIBERO_TYPE if not set
    if libero_type is None:
        if any(suite.endswith(s) for s in ("_swap", "_task", "_lan")):
            libero_type = "pro"
        else:
            libero_type = "standard"

    recipe_tag = f"{suite.replace('libero_', '')}_t{task}_s{seed}"
    regime = "strict_perception" if perception else "strict"
    if perception:
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
    else:
        if not (Path(workdir) / "state_00.json").exists():
            raise RuntimeError(f"--no_driver but {workdir}/state_00.json missing")

    t0 = time.time()
    finish_result, messages, agent_error = None, [], None
    stats = {"total_input_tokens": 0, "total_output_tokens": 0, "turns_used": 0, "tool_calls": 0}
    try:
        finish_result, messages, stats = run_agent_loop(
            client, model, system_prompt, user_msg,
            max_turns=max_turns, max_tokens=max_tokens, verbose=verbose,
        )
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
    ap.add_argument("--model", default="claude-sonnet-4-5",
                    help="Anthropic model id (e.g. claude-sonnet-4-5, claude-opus-4-5)")
    ap.add_argument("--max_turns", type=int, default=80)
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--max_episode_steps", type=int, default=600)
    ap.add_argument("--cuda_device", default=os.environ.get("CUDA_DEVICE", "0"))
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--api_key", default=None,
                    help="Defaults to ANTHROPIC_API_KEY env var.")
    ap.add_argument("--base_url", default=None,
                    help="Override Anthropic base URL (for proxy endpoints). "
                         "Defaults to ANTHROPIC_BASE_URL env var or anthropic.com.")
    ap.add_argument("--no_driver", action="store_true",
                    help="Don't spawn driver; attach to existing workdir")
    ap.add_argument("--workdir", default=DEFAULT_WORKDIR,
                    help="REPL working directory (default /tmp/hybrid_repl). "
                         "Override for parallel runs.")
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
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: set ANTHROPIC_API_KEY env var or pass --api_key", file=sys.stderr)
        return 2
    base_url = args.base_url or os.environ.get("ANTHROPIC_BASE_URL")
    run_one_cell(
        suite=args.suite, task=args.task, seed=args.seed,
        api_key=api_key, model=args.model,
        max_turns=args.max_turns, max_tokens=args.max_tokens,
        max_episode_steps=args.max_episode_steps,
        cuda_device=args.cuda_device,
        output_dir=args.output_dir,
        no_driver=args.no_driver,
        verbose=not args.quiet,
        base_url=base_url,
        workdir=args.workdir,
        perception=args.perception,
        libero_type=args.libero_type,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
