"""Batch runner: drive N tasks through the Chinese dashboard end-to-end and
record each one at 1080p, 5 at a time.

Each task: launch the agent (own dashboard port, deepseek-v4-pro via
claude_code), wait for its dashboard URL, run scripts/record_dashboard.py
(which clicks 开始运行 + 发送 and records the live run, early-stopping on
``done``), convert the resulting webm to mp4 with ffmpeg, drop the webm, and
kill the agent tree. Concurrency is capped at --workers (default 5).

Usage:
  run_batch.py --batch-root <dir> --duration 300 --workers 5 \
               --tasks 0 1 2 3 4 5 6 7 8 9 --seeds 1 2 3
"""
from __future__ import annotations

import argparse
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path("/mnt/public/zhuchunyang_rl/PhysicalAgent")
PYBIN = "/mnt/public/zhuchunyang_rl/.venv_agenticvla/bin/python"
VLA = "http://127.0.0.1:50008"
# Anthropic credentials are read from the parent environment (ANTHROPIC_BASE_URL,
# ANTHROPIC_AUTH_TOKEN) — never hardcoded here — and passed through to each
# spawned agent. The parent process is launched with them exported.
PYTHONPATH = ":".join([
    "/mnt/public/zhuchunyang_rl/rlinf_libero_camera_meta",
    str(REPO),
    "/mnt/public/zhuchunyang_rl/.venv_agenticvla/libero",
    "/mnt/public/zhuchunyang_rl/.venv_agenticvla/libero_pro",
    "/mnt/public/zhuchunyang_rl/.venv_agenticvla/libero_plus",
])

URL_RE = re.compile(r"Dashboard: (http://127\.0\.0\.1:\d+)")


def _base_env() -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = PYTHONPATH
    if "ANTHROPIC_BASE_URL" not in env or "ANTHROPIC_AUTH_TOKEN" not in env:
        raise RuntimeError(
            "ANTHROPIC_BASE_URL / ANTHROPIC_AUTH_TOKEN must be set in the "
            "environment before launching the batch."
        )
    return env


def _wait_for_url(run_log: Path, timeout: float = 90.0) -> str | None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            text = run_log.read_text(errors="replace")
        except Exception:
            text = ""
        m = URL_RE.search(text)
        if m:
            return m.group(1)
        time.sleep(0.5)
    return None


def run_one(task: int, seed: int, out: Path, duration: int, height: int) -> dict:
    """Run a single task: agent -> record -> mp4 -> cleanup. Returns a report."""
    out.mkdir(parents=True, exist_ok=True)
    run_log = out / "run.log"
    rec_log = out / "record.log"
    env = _base_env()
    report = {"task": task, "seed": seed, "out": str(out)}

    cmd = [
        PYBIN, str(REPO / "cli" / "main.py"),
        "--dashboard", "--dashboard_language", "zh-cn", "--dashboard-host", "127.0.0.1",
        "--vla-endpoint", VLA,
        "--suite", "libero_object_task", "--task", str(task), "--seed", str(seed),
        "--cerebrum", "claude_code", "--model", "deepseek-v4-pro",
        "--max-turns", "100", "--max-tokens", "4096", "--output-dir", str(out),
    ]
    with open(run_log, "w") as lf:
        agent = subprocess.Popen(
            cmd, env=env, stdout=lf, stderr=subprocess.STDOUT,
            start_new_session=True,  # own process group -> killable whole
        )
    report["agent_pid"] = agent.pid

    url = _wait_for_url(run_log)
    if not url:
        report["status"] = "no_dashboard"
        _kill_tree(agent.pid, out)
        return report
    report["url"] = url

    rec_cmd = [PYBIN, "-u", str(REPO / "scripts" / "record_dashboard.py"),
               url, str(out), str(duration), str(height)]
    try:
        with open(rec_log, "w") as rf:
            subprocess.run(rec_cmd, stdout=rf, stderr=subprocess.STDOUT, check=False)
    except Exception as e:
        report["status"] = f"record_error: {e}"
        _kill_tree(agent.pid, out)
        return report

    # webm -> mp4 (every frame kept), then drop the webm so only one stays.
    webm = out / "dashboard.webm"
    mp4 = out / "dashboard.mp4"
    if webm.exists():
        ff = subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "warning", "-i", str(webm),
             "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
             str(mp4)],
            stdout=open(rec_log, "a"), stderr=subprocess.STDOUT,
        )
        if mp4.exists():
            webm.unlink(missing_ok=True)
        report["ffmpeg_rc"] = ff.returncode

    _kill_tree(agent.pid, out)
    report["status"] = "ok" if mp4.exists() else "no_mp4"
    report["mp4_bytes"] = mp4.stat().st_size if mp4.exists() else 0
    return report


def _kill_tree(pgid: int, out: Path) -> None:
    try:
        os.killpg(os.getpgid(pgid), signal.SIGKILL)
    except Exception:
        pass
    # env_server is a child but be explicit too (match by output dir path).
    pat = str(out).replace("/", r"\/")
    subprocess.run(["pkill", "-9", "-f", f"env_server.py.*{pat}"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-root", required=True)
    ap.add_argument("--duration", type=int, default=2400,
                    help="per-task recording cap (s); early-stops on run done")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--height", type=int, default=720,
                    help="video height (720 -> 1280x720); width=height*16/9")
    ap.add_argument("--tasks", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    args = ap.parse_args()

    batch_root = Path(args.batch_root)
    batch_root.mkdir(parents=True, exist_ok=True)
    jobs = [(t, s) for t in args.tasks for s in args.seeds]
    print(f"[batch] {len(jobs)} jobs, {args.workers} parallel, "
          f"duration={args.duration}s, root={batch_root}", flush=True)

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, t, s, batch_root / f"t{t}_s{s}", args.duration, args.height): (t, s)
                for (t, s) in jobs}
        for fut in as_completed(futs):
            t, s = futs[fut]
            try:
                r = fut.result()
            except Exception as e:
                r = {"task": t, "seed": s, "status": f"exception: {e}"}
            results.append(r)
            print(f"[batch] done t{t}_s{s}: {r.get('status')} "
                  f"mp4={r.get('mp4_bytes', 0)}B", flush=True)

    ok = sum(1 for r in results if r.get("status") == "ok")
    print(f"[batch] {ok}/{len(results)} produced dashboard.mp4", flush=True)
    (batch_root / "batch_summary.json").write_text(
        __import__("json").dumps(results, indent=2, ensure_ascii=False))
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
