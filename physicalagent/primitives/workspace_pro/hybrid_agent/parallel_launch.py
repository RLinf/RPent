"""Launch several hybrid agents in parallel, one per GPU.

Each cell gets:
  - its own CUDA device (--cuda_device passed through)
  - its own workdir (/tmp/hybrid_repl_<tag>)
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

_THIS_DIR = Path(__file__).resolve().parent


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
    ]
    if base_url:
        cmd += ["--base_url", base_url]
    if api_key:
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
    ap.add_argument("--max_turns", type=int, default=40)
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--max_episode_steps", type=int, default=600)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--log_dir", default="/tmp")
    ap.add_argument("--api_key", default=None,
                    help="defaults to ANTHROPIC_API_KEY env var")
    ap.add_argument("--base_url", default=None,
                    help="defaults to ANTHROPIC_BASE_URL env var")
    ap.add_argument("--python_bin", default=os.environ.get("PYTHON_BIN", "/opt/venv/openpi/bin/python"))
    ap.add_argument("--workdir_root", default="/tmp",
                    help="Each cell gets a fresh /<root>/hybrid_repl_<tag>/ directory")
    ap.add_argument("--stagger_s", type=float, default=20.0,
                    help="Seconds to wait between launching successive agents "
                         "(helps avoid hammering the API + spreads Pi0 load IO).")
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

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY missing", file=sys.stderr)
        return 2
    base_url = args.base_url or os.environ.get("ANTHROPIC_BASE_URL")

    n_devices = len(args.cuda_devices)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    procs: list[tuple[subprocess.Popen, str, str, tuple]] = []
    print(f"[parallel] launching {len(cells)} cells across {n_devices} GPUs")
    for i, (suite, task, seed) in enumerate(cells):
        cuda = args.cuda_devices[i % n_devices]
        tag = f"{suite.replace('libero_','')}_t{task}_s{seed}"
        workdir = f"{args.workdir_root}/hybrid_repl_{tag}"
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
        )

        print(f"  [{i}] {tag}  GPU={cuda}  workdir={workdir}  log={log_path}")
        if args.dry_run:
            print(f"      cmd: {' '.join(cmd)}")
            continue

        log_f = open(log_path, "w")
        env = os.environ.copy()
        # API key/base also via env as a fallback
        env["ANTHROPIC_API_KEY"] = api_key
        if base_url:
            env["ANTHROPIC_BASE_URL"] = base_url
        # Stagger launches so 4 agents don't hammer the API at the same
        # instant during initial guide reading (proxy may refuse parallel
        # TLS handshakes). Also spreads Pi0 model-load disk IO.
        if i > 0:
            time.sleep(args.stagger_s)
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)
        procs.append((proc, log_path, tag, (suite, task, seed)))

    if args.dry_run:
        return 0

    print(f"\n[parallel] waiting for {len(procs)} cells...")
    t0 = time.time()
    results = []
    for proc, log_path, tag, cell in procs:
        rc = proc.wait()
        elapsed = time.time() - t0
        # Extract the FINISH line from the agent log if present
        try:
            log_text = Path(log_path).read_text()
        except Exception:
            log_text = ""
        finish_line = next(
            (ln for ln in reversed(log_text.splitlines()) if "FINISH called" in ln),
            "",
        )
        usage_line = next(
            (ln for ln in reversed(log_text.splitlines()) if ln.startswith("[agent] usage:")),
            "",
        )
        results.append({
            "tag": tag, "cell": cell, "rc": rc, "elapsed_s": round(elapsed, 1),
            "finish": finish_line,  # keep FULL line — 'status: success' is at the start
            "usage": usage_line,
        })
        print(f"[parallel] [{tag}] rc={rc}  elapsed={elapsed:.1f}s")
        if finish_line:
            # show head (where 'status' lives) and a tail if it's long
            print(f"           {finish_line[:280]}")
            if len(finish_line) > 480:
                print(f"           ...{finish_line[-200:]}")
        if usage_line:
            print(f"           {usage_line}")

    print(f"\n[parallel] all done in {time.time()-t0:.1f}s")
    print("[parallel] summary:")
    n_ok = n_sim_ok = 0
    for r in results:
        ok = ("status': 'success'" in r["finish"]) or ('"status": "success"' in r["finish"])
        # Sim-side success fallback: scan the workdir state_*.json for
        # libero_terminated=true. This catches cases where the agent
        # crashed (e.g. API error) AFTER the sim already terminated.
        tag = r["tag"]
        wd = Path(f"{args.workdir_root}/hybrid_repl_{tag}")
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
