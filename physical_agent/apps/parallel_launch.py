"""Launch several hybrid agents in parallel, one per GPU.

Each cell gets:
  - its own CUDA device (--cuda_device passed through)
  - its own output dir (<output_dir>/<tag>/) holding everything for the
    run: driver artifacts (images/, depths/, states.json, episode.mp4)
    plus the agent's transcript / recipe / audit / run.log
  - its own subprocess stdout/stderr capture file in --log_dir

Usage:
    export ANTHROPIC_API_KEY=sk-...
    export ANTHROPIC_BASE_URL=https://...

    # Or for OpenAI-compatible providers:
    export OPENAI_COMPAT_API_KEY=sk-...
    export OPENAI_COMPAT_BASE_URL=https://provider.example/v1

    # 4 seeds of libero_spatial_lan t0 on GPUs 0..3:
    python parallel_launch.py \\
        --suite libero_spatial_lan --task 0 \\
        --seeds 0 1 2 3 --cuda_devices 0 1 2 3 \\
        --model claude-sonnet-4-5 \\
        --output_dir /path/to/results_agent_runs

The launcher spawns N runner.py subprocesses (each starts its own
interactive_driver), waits for all to finish, then prints a summary.

Notes:
  - Each subprocess does its own Pi0 model load (~80 s); they run
    concurrently so total wall time is roughly max(per-cell) not sum.
    - API calls are concurrent — make sure your endpoint can
    handle the parallel rate. If you hit rate limits, drop the
    concurrency by passing fewer cuda_devices than seeds (the extras
    queue up).
  - This launcher only does single-task-many-seeds OR
    explicit-(suite,task,seed) tuples (see --tuples).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from physical_agent.utils.config import (
    get_anthropic_api_key,
    get_anthropic_base_url,
    get_openai_compat_api_key,
    get_openai_compat_base_url,
)
from physical_agent.utils.logging import get_logger, init_output_dir

logger = get_logger("parallel")

_THIS_DIR = Path(__file__).resolve().parent


def _summarize_process(proc, log_path: str, tag: str, cell: tuple) -> dict:
    try:
        log_text = Path(log_path).read_text()
    except Exception:
        log_text = ""
    finish_line = next(
        (line for line in reversed(log_text.splitlines()) if "FINISH called" in line),
        "",
    )
    usage_line = next(
        (line for line in reversed(log_text.splitlines()) if line.startswith("[agent] usage:")),
        "",
    )
    return {
        "tag": tag,
        "cell": cell,
        "rc": proc.returncode,
        "finish": finish_line,
        "usage": usage_line,
    }


def _runner_cmd(
    *,
    suite: str,
    task: int,
    seed: int,
    cuda_device: str,
    model: str | None,
    max_turns: int,
    max_tokens: int,
    max_episode_steps: int,
    output_dir: str,
    base_url: str | None,
    api_key: str | None,
    cerebrum: str,
    perception: bool,
    env_name: str | None,
    libero_type: str | None,
    claude_code_timeout_s: int | None,
    claude_code_max_budget_usd: float | None,
    codex_timeout_s: int | None,
    openai_compat_no_images: bool,
    thinking: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        str(_THIS_DIR / "runner.py"),
        "--suite", suite,
        "--task", str(task),
        "--seed", str(seed),
        "--cuda_device", str(cuda_device),
        "--max_turns", str(max_turns),
        "--max_tokens", str(max_tokens),
        "--max_episode_steps", str(max_episode_steps),
        "--output_dir", output_dir,
        "--cerebrum", cerebrum,
    ]
    if model:
        cmd += ["--model", model]
    if perception:
        cmd += ["--perception"]
    if env_name:
        cmd += ["--env", env_name]
    if libero_type:
        cmd += ["--libero_type", libero_type]
    if claude_code_timeout_s is not None:
        cmd += ["--claude_code_timeout_s", str(claude_code_timeout_s)]
    if claude_code_max_budget_usd is not None:
        cmd += ["--claude_code_max_budget_usd", str(claude_code_max_budget_usd)]
    if codex_timeout_s is not None:
        cmd += ["--codex_timeout_s", str(codex_timeout_s)]
    if base_url:
        cmd += ["--base_url", base_url]
    # Do not pass API keys on argv; main() injects them through the child env.
    if openai_compat_no_images:
        cmd += ["--openai_compat_no_images"]
    if thinking:
        cmd += ["--thinking"]
    return cmd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--task", type=int, required=True,
                    help="Used when --seeds is given (single task, multi seed)")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="List of seeds (one cell per seed on the given --task)")
    ap.add_argument("--tuples", type=str, nargs="+", default=None,
                    help="Alternative to --seeds: explicit (suite,task,seed) "
                         "triples e.g. libero_spatial_lan:0:0 libero_spatial_lan:0:1")
    ap.add_argument("--cuda_devices", type=str, nargs="+", default=["0", "1", "2", "3"])
    ap.add_argument("--model", default=None,
                    help="Model id. Defaults to the selected backend's model env var.")
    ap.add_argument("--cerebrum", default="anthropic", choices=["anthropic", "openai_compat", "claude_code", "codex"],
                    help="LLM backend passed to runner.py.")
    ap.add_argument("--max_turns", type=int, default=40)
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--max_episode_steps", type=int, default=600)
    ap.add_argument("--output_dir", required=True,
                    help="Parent directory; each cell writes to <output_dir>/<tag>/.")
    ap.add_argument("--log_dir", default="/tmp")
    ap.add_argument("--api_key", default=None,
                    help="defaults to the selected backend's API key env var")
    ap.add_argument("--base_url", default=None,
                    help="defaults to the selected backend's base URL env var")
    ap.add_argument("--stagger_s", type=float, default=20.0,
                    help="Seconds to wait between launching successive agents "
                         "(helps avoid hammering the API + spreads Pi0 load IO).")
    ap.add_argument("--perception", action="store_true",
                    help="Run perception-isolated cells (--hide_object_coords, full perception prompt).")
    ap.add_argument("--env", dest="env_name", default=None,
                    help="Environment backend passed to runner.py. Defaults to runner inference/libero.")
    ap.add_argument("--libero_type", default=None, choices=["standard", "pro", "plus"],
                    help="LIBERO variant to pass through. Default is runner.py auto-routing.")
    ap.add_argument("--skip_existing", action="store_true",
                    help="Skip cells whose audit JSON already exists at "
                         "<output_dir>/<tag>/<tag>.json.")
    ap.add_argument("--claude_code_timeout_s", type=int, default=None,
                    help="Wall-clock cap for claude -p cells. Defaults in runner.py.")
    ap.add_argument("--claude_code_max_budget_usd", type=float, default=None,
                    help="Budget passed to claude -p --max-budget-usd.")
    ap.add_argument("--codex_timeout_s", type=int, default=None,
                    help="Wall-clock cap for codex exec cells. Defaults in runner.py.")
    ap.add_argument("--openai_compat_no_images", action="store_true",
                    help="Do not send tool-result images to an openai_compat model.")
    ap.add_argument("--thinking", action="store_true",
                    help="Enable extended thinking / reasoning for anthropic and "
                         "openai_compat backends (no-op for claude_code/codex).")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    # Build cell list
    cells: list[tuple[str, int, int]] = []
    if args.seeds is not None:
        for s in args.seeds:
            cells.append((args.suite, args.task, s))
    if args.tuples:
        for t in args.tuples:
            parts = t.split(":")
            if len(parts) != 3:
                logger.error("bad tuple: %r (expected suite:task:seed)", t)
                return 2
            cells.append((parts[0], int(parts[1]), int(parts[2])))
    if not cells:
        logger.error("nothing to do (pass --seeds or --tuples)")
        return 2

    if args.cerebrum == "openai_compat":
        api_key = args.api_key or get_openai_compat_api_key()
        base_url = args.base_url or get_openai_compat_base_url()
    elif args.cerebrum == "anthropic":
        api_key = args.api_key or get_anthropic_api_key()
        base_url = args.base_url or get_anthropic_base_url()
    else:
        api_key = args.api_key
        base_url = args.base_url
    if args.cerebrum == "anthropic" and not api_key:
        logger.error("ANTHROPIC_API_KEY missing")
        return 2
    if args.cerebrum == "openai_compat" and not api_key:
        logger.error("OPENAI_COMPAT_API_KEY or OPENAI_API_KEY missing")
        return 2

    n_devices = len(args.cuda_devices)
    # Initialise unified logging for this run
    init_output_dir(args.log_dir)

    procs: list[tuple[subprocess.Popen, str, str, tuple, int]] = []
    active: list[tuple[subprocess.Popen, str, str, tuple, int]] = []
    all_results: list[dict] = []

    def collect_finished(*, block: bool = False) -> None:
        """Collect finished cells, optionally waiting for one free GPU slot."""
        while True:
            for item in list(active):
                proc, log_path, tag, cell, _slot = item
                rc = proc.poll()
                if rc is None:
                    continue
                active.remove(item)
                all_results.append(_summarize_process(proc, log_path, tag, cell))
            if not block or len(active) < n_devices:
                return
            time.sleep(2)

    logger.info("launching %d cells across %d GPUs", len(cells), n_devices)
    launched = 0
    for i, (suite, task, seed) in enumerate(cells):
        tag = f"{suite.replace('libero_','')}_t{task}_s{seed}"
        cell_output_dir = str(Path(args.output_dir) / tag)
        if args.skip_existing and (Path(cell_output_dir) / f"{tag}.json").exists():
            logger.info("  [%d] %s  SKIP existing audit", i, tag)
            continue
        if args.dry_run:
            slot = i % n_devices
        else:
            collect_finished(block=len(active) >= n_devices)
            used_slots = {item[4] for item in active}
            slot = next((idx for idx in range(n_devices) if idx not in used_slots), 0)
        cuda = args.cuda_devices[slot]
        log_path = f"{args.log_dir}/agent_{tag}.log"

        cmd = _runner_cmd(
            suite=suite, task=task, seed=seed,
            cuda_device=cuda,
            model=args.model,
            max_turns=args.max_turns,
            max_tokens=args.max_tokens,
            max_episode_steps=args.max_episode_steps,
            output_dir=cell_output_dir,
            base_url=base_url,
            api_key=api_key,
            cerebrum=args.cerebrum,
            perception=args.perception,
            env_name=args.env_name,
            libero_type=args.libero_type,
            claude_code_timeout_s=args.claude_code_timeout_s,
            claude_code_max_budget_usd=args.claude_code_max_budget_usd,
            codex_timeout_s=args.codex_timeout_s,
            openai_compat_no_images=args.openai_compat_no_images,
            thinking=args.thinking,
        )

        logger.info("  [%d] %s  GPU=%s  output_dir=%s  log=%s",
                    i, tag, cuda, cell_output_dir, log_path)
        if args.dry_run:
            logger.info("      cmd: %s", ' '.join(cmd))
            continue

        log_f = open(log_path, "w")
        env = os.environ.copy()
        # API key/base also via env as a fallback for the selected API backend.
        if args.cerebrum == "openai_compat":
            if api_key:
                env["OPENAI_COMPAT_API_KEY"] = api_key
            if base_url:
                env["OPENAI_COMPAT_BASE_URL"] = base_url
        elif args.cerebrum == "anthropic":
            if api_key:
                env["ANTHROPIC_API_KEY"] = api_key
            if base_url:
                env["ANTHROPIC_BASE_URL"] = base_url
        # Stagger launches so 4 agents don't hammer the API at the same
        # instant during initial guide reading (proxy may refuse parallel
        # TLS handshakes). Also spreads Pi0 model-load disk IO.
        if launched > 0:
            time.sleep(args.stagger_s)
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)
        item = (proc, log_path, tag, (suite, task, seed, cuda), slot)
        procs.append(item)
        active.append(item)
        launched += 1

    if args.dry_run:
        return 0

    logger.info("waiting for %d cells...", len(procs))
    t0 = time.time()
    for proc, log_path, tag, cell, _slot in list(active):
        rc = proc.wait()
        result = _summarize_process(proc, log_path, tag, cell)
        all_results.append(result)
        elapsed = time.time() - t0
        logger.info("[%s] rc=%d  elapsed=%.1fs", tag, rc, elapsed)
        if result["finish"]:
            # show head (where 'status' lives) and a tail if it's long
            logger.info("           %s", result['finish'][:280])
            if len(result["finish"]) > 480:
                logger.info("           ...%s", result['finish'][-200:])
        if result["usage"]:
            logger.info("           %s", result['usage'])

    results = all_results

    logger.info("all done in %.1fs", time.time()-t0)
    logger.info("summary:")
    n_ok = n_sim_ok = 0
    for r in results:
        ok = ("status': 'success'" in r["finish"]) or ('"status": "success"' in r["finish"])
        # Sim-side success fallback: read states.json for any step with
        # libero_terminated=true. This catches cases where the agent
        # crashed (e.g. API error) AFTER the sim already terminated.
        tag = r["tag"]
        cell_dir = Path(args.output_dir) / tag
        sim_ok = False
        if cell_dir.exists():
            import json as _json
            sp = cell_dir / "states.json"
            if sp.exists():
                try:
                    arr = _json.load(open(sp))
                    if isinstance(arr, list):
                        for entry in arr:
                            if isinstance(entry, dict) and entry.get("libero_terminated"):
                                sim_ok = True
                                break
                except Exception:
                    pass
        if ok:
            n_ok += 1
        if sim_ok:
            n_sim_ok += 1
        label = "AGENT-SUCCESS" if ok else ("SIM-SUCCESS-but-agent-crashed" if sim_ok else "FAILED/STUCK")
        logger.info("  %s: rc=%d  %s", tag, r['rc'], label)
    logger.info("%d/%d agents reported success "
          f"(+{n_sim_ok - n_ok} more reached libero_terminated=true but agent didn't finish cleanly)")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
