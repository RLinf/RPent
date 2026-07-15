"""Drive the Chinese dashboard through a full run and record it.

Single Playwright context records from the first frame (before the Run click)
through the Send click and the whole live run, so nothing in the session is
missed. Output: one ``dashboard.webm`` in <video_dir> (renamed from the
auto-named Playwright webm); convert to mp4 separately with ffmpeg.

Usage: record_dashboard.py <dashboard_url> <video_dir> [duration_s] [height]
  duration_s  default 300  (max seconds recorded after Send; stops early when
                            the run reaches state=="done")
  height      default 1080 (sets width=height*16/9; 1080 -> 1920x1080)
"""
from __future__ import annotations

import sys
import time
import urllib.request
import json
from pathlib import Path

from playwright.sync_api import sync_playwright


def _launch_state(url: str) -> dict:
    try:
        with urllib.request.urlopen(f"{url}/api/launch/state", timeout=3) as r:
            return json.loads(r.read())
    except Exception:
        return {}


def main() -> int:
    url = sys.argv[1].rstrip("/")
    video_dir = Path(sys.argv[2])
    duration_s = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    height = int(sys.argv[4]) if len(sys.argv) > 4 else 1080
    width = (height * 16) // 9
    video_dir.mkdir(parents=True, exist_ok=True)

    # Wait until the launcher is armed so the page actually shows the start
    # screen (otherwise boot() skips straight to the monitor with no run).
    deadline = time.time() + 60
    while time.time() < deadline:
        st = _launch_state(url)
        if st.get("enabled") and st.get("pending"):
            break
        time.sleep(0.4)

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
        )
        context = browser.new_context(
            viewport={"width": width, "height": height},
            record_video_dir=str(video_dir),
            record_video_size={"width": width, "height": height},
        )
        page = context.new_page()
        page.on("console", lambda m: print(f"[browser:{m.type}] {m.text}", flush=True))
        page.on("pageerror", lambda e: print(f"[browser:error] {e}", flush=True))

        page.goto(url, wait_until="domcontentloaded")
        print(f"[record] page loaded at {width}x{height}; waiting for launcher", flush=True)

        page.wait_for_selector("#runBtn", state="visible", timeout=60000)
        print("[record] clicking 开始运行 (#runBtn)", flush=True)
        page.click("#runBtn")

        ready = """() => {
          const inp = document.querySelector('#chatInput');
          const snd = document.querySelector('#chatSend');
          const composer = document.querySelector('#composer');
          if (!inp || !snd || !composer) return false;
          if (composer.style.display === 'none') return false;
          if (!inp.value || !inp.value.trim()) return false;
          if (snd.disabled) return false;
          return true;
        }"""
        try:
            page.wait_for_function(ready, timeout=60000)
        except Exception:
            # pollForRun didn't flip to the monitor — reload once; with the run
            # now registered boot() goes straight to the monitor. Recording
            # continues across the reload, so no frame is lost.
            print("[record] monitor not ready; reloading to force it", flush=True)
            page.reload(wait_until="domcontentloaded")
            page.wait_for_function(ready, timeout=120000)

        print("[record] prompt prefilled & 发送 enabled; clicking 发送", flush=True)
        page.click("#chatSend")

        # Record the live run until it finishes (state=="done") or the duration
        # cap — whichever comes first — so a fast episode isn't padded with
        # idle frames and a slow one is capped.
        print(f"[record] recording up to {duration_s}s (early-stop on done) …", flush=True)
        deadline = time.time() + duration_s
        while time.time() < deadline:
            page.wait_for_timeout(5000)
            try:
                with urllib.request.urlopen(f"{url}/api/runs", timeout=4) as r:
                    runs = json.loads(r.read()).get("runs") or []
                if runs and runs[0].get("state") == "done":
                    print("[record] run done; stopping early", flush=True)
                    break
            except Exception:
                pass

        context.close()
        browser.close()

        webm = next(video_dir.glob("*.webm"), None)
        if webm is None:
            print("[record] no .webm produced!", flush=True)
            return 1
        final = video_dir / "dashboard.webm"
        webm.replace(final)
        print(f"[record] saved {final} ({final.stat().st_size} bytes)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
