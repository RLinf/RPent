"""Launch several hybrid agents in parallel, one per GPU.

Each cell gets:
  - its own CUDA device (--cuda_device passed through)
    - its own workdir (<output_dir>/repl/hybrid_repl_<tag> by default)
  - its own log file
  - its own transcript / recipe / audit in --output_dir

Usage:
    export ANTHROPIC_API_KEY=sk-...
    export ANTHROPIC_BASE_URL=https://...

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
  - Anthropic API calls are concurrent — make sure your endpoint can
    handle the parallel rate. If you hit rate limits, drop the
    concurrency by passing fewer cuda_devices than seeds (the extras
    queue up).
  - This launcher only does single-task-many-seeds OR
    explicit-(suite,task,seed) tuples (see --tuples).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from physicalagent.config import get_python_bin, get_anthropic_api_key, get_anthropic_base_url

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
    python_bin: str,
    suite: str,
    task: int,
    seed: int,
    cuda_device: str,
    workdir: str,
    model: str,
    max_turns: int,
    max_tokens: int,
    max_episode_steps: int,
    output_dir: str,
    base_url: str | None,
    api_key: str | None,
    cerebrum: str,
    perception: bool,
    libero_type: str | None,
    claude_code_timeout_s: int | None,
    claude_code_max_budget_usd: float | None,
) -> list[str]:
    cmd = [
        python_bin,
        str(_THIS_DIR / "runner.py"),
        "--suite", suite,
        "--task", str(task),
        "--seed", str(seed),
        "--cuda_device", str(cuda_device),
        "--workdir", workdir,
        "--model", model,
        "--max_turns", str(max_turns),
        "--max_tokens", str(max_tokens),
        "--max_episode_steps", str(max_episode_steps),
        "--output_dir", output_dir,
        "--cerebrum", cerebrum,
    ]
    if perception:
        cmd += ["--perception"]
    if libero_type:
        cmd += ["--libero_type", libero_type]
    if claude_code_timeout_s is not None:
        cmd += ["--claude_code_timeout_s", str(claude_code_timeout_s)]
    if claude_code_max_budget_usd is not None:
        cmd += ["--claude_code_max_budget_usd", str(claude_code_max_budget_usd)]
    if base_url:
        cmd += ["--base_url", base_url]
    if api_key and cerebrum == "anthropic":
        cmd += ["--api_key", api_key]
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
    ap.add_argument("--model", default="claude-sonnet-4-5")
    ap.add_argument("--cerebrum", default="anthropic", choices=["anthropic", "claude_code"],
                    help="LLM backend passed to runner.py.")
    ap.add_argument("--max_turns", type=int, default=40)
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--max_episode_steps", type=int, default=600)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--log_dir", default="/tmp")
    ap.add_argument("--api_key", default=None,
                    help="defaults to ANTHROPIC_API_KEY env var")
    ap.add_argument("--base_url", default=None,
                    help="defaults to ANTHROPIC_BASE_URL env var")
    ap.add_argument("--python_bin", default=get_python_bin())
    ap.add_argument("--workdir_root", default=None,
                    help="Each cell gets a fresh <root>/hybrid_repl_<tag>/ directory. "
                         "Default: <output_dir>/repl, or PHYSICALAGENT_WORKDIR_PREFIX.")
    ap.add_argument("--stagger_s", type=float, default=20.0,
                    help="Seconds to wait between launching successive agents "
                         "(helps avoid hammering the API + spreads Pi0 load IO).")
    ap.add_argument("--perception", action="store_true",
                    help="Run perception-isolated cells (--hide_object_coords, full perception prompt).")
    ap.add_argument("--libero_type", default=None, choices=["standard", "pro", "plus"],
                    help="LIBERO variant to pass through. Default is runner.py auto-routing.")
    ap.add_argument("--skip_existing", action="store_true",
                    help="Skip cells whose audit JSON already exists in --output_dir.")
    ap.add_argument("--claude_code_timeout_s", type=int, default=None,
                    help="Wall-clock cap for claude -p cells. Defaults in runner.py.")
    ap.add_argument("--claude_code_max_budget_usd", type=float, default=None,
                    help="Budget passed to claude -p --max-budget-usd.")
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
                print(f"bad tuple: {t!r} (expected suite:task:seed)", file=sys.stderr)
                return 2
            cells.append((parts[0], int(parts[1]), int(parts[2])))
    if not cells:
        print("nothing to do (pass --seeds or --tuples)", file=sys.stderr)
        return 2

    api_key = args.api_key or get_anthropic_api_key()
    if args.cerebrum == "anthropic" and not api_key:
        print("ERROR: ANTHROPIC_API_KEY missing", file=sys.stderr)
        return 2
    base_url = args.base_url or get_anthropic_base_url()

    n_devices = len(args.cuda_devices)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    workdir_root = args.workdir_root or os.environ.get("PHYSICALAGENT_WORKDIR_PREFIX")
    if workdir_root is None:
        workdir_root = str(Path(args.output_dir) / "repl")

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

    print(f"[parallel] launching {len(cells)} cells across {n_devices} GPUs")
    launched = 0
    for i, (suite, task, seed) in enumerate(cells):
        tag = f"{suite.replace('libero_','')}_t{task}_s{seed}"
        if args.skip_existing and (Path(args.output_dir) / f"{tag}.json").exists():
            print(f"  [{i}] {tag}  SKIP existing audit")
            continue
        if args.dry_run:
            slot = i % n_devices
        else:
            collect_finished(block=len(active) >= n_devices)
            used_slots = {item[4] for item in active}
            slot = next((idx for idx in range(n_devices) if idx not in used_slots), 0)
        cuda = args.cuda_devices[slot]
        workdir = str(Path(workdir_root) / f"hybrid_repl_{tag}")
        log_path = f"{args.log_dir}/agent_{tag}.log"

        cmd = _runner_cmd(
            python_bin=args.python_bin,
            suite=suite, task=task, seed=seed,
            cuda_device=cuda, workdir=workdir,
            model=args.model,
            max_turns=args.max_turns,
            max_tokens=args.max_tokens,
            max_episode_steps=args.max_episode_steps,
            output_dir=args.output_dir,
            base_url=base_url,
            api_key=api_key,
            cerebrum=args.cerebrum,
            perception=args.perception,
            libero_type=args.libero_type,
            claude_code_timeout_s=args.claude_code_timeout_s,
            claude_code_max_budget_usd=args.claude_code_max_budget_usd,
        )

        print(f"  [{i}] {tag}  GPU={cuda}  workdir={workdir}  log={log_path}")
        if args.dry_run:
            print(f"      cmd: {' '.join(cmd)}")
            continue

        log_f = open(log_path, "w")
        env = os.environ.copy()
        # API key/base also via env as a fallback for the Anthropic backend.
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

    print(f"\n[parallel] waiting for {len(procs)} cells...")
    t0 = time.time()
    for proc, log_path, tag, cell, _slot in list(active):
        rc = proc.wait()
        result = _summarize_process(proc, log_path, tag, cell)
        all_results.append(result)
        elapsed = time.time() - t0
        print(f"[parallel] [{tag}] rc={rc}  elapsed={elapsed:.1f}s")
        if result["finish"]:
            # show head (where 'status' lives) and a tail if it's long
            print(f"           {result['finish'][:280]}")
            if len(result["finish"]) > 480:
                print(f"           ...{result['finish'][-200:]}")
        if result["usage"]:
            print(f"           {result['usage']}")

    results = all_results

    print(f"\n[parallel] all done in {time.time()-t0:.1f}s")
    print("[parallel] summary:")
    n_ok = n_sim_ok = 0
    for r in results:
        ok = ("status': 'success'" in r["finish"]) or ('"status": "success"' in r["finish"])
        # Sim-side success fallback: scan the workdir state_*.json for
        # libero_terminated=true. This catches cases where the agent
        # crashed (e.g. API error) AFTER the sim already terminated.
        tag = r["tag"]
        wd = Path(workdir_root) / f"hybrid_repl_{tag}"
        sim_ok = False
        if wd.exists():
            import json as _json
            for sp in sorted(wd.glob("state_*.json"), reverse=True):
                try:
                    if _json.load(open(sp)).get("libero_terminated"):
                        sim_ok = True
                        break
                except Exception:
                    continue
        if ok:
            n_ok += 1
        if sim_ok:
            n_sim_ok += 1
        label = "AGENT-SUCCESS" if ok else ("SIM-SUCCESS-but-agent-crashed" if sim_ok else "FAILED/STUCK")
        print(f"  {tag}: rc={r['rc']}  {label}")
    print(f"[parallel] {n_ok}/{len(results)} agents reported success "
          f"(+{n_sim_ok - n_ok} more reached libero_terminated=true but agent didn't finish cleanly)")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
