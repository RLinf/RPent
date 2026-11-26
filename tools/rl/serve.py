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

"""Launch and supervise SAM3 + Sam3Pool + warm simulator pools & routers from YAML."""

from __future__ import annotations

import argparse
import os
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import yaml

from rpent.utils.logging import get_logger
from rpent.utils.rpc import make_rpc_client, wait_for_ready

logger = get_logger("rpent_serve")


def _spawn(
    name: str,
    cmd: list[str],
    log_dir: Path,
    env_overrides: dict[str, str] | None = None,
) -> subprocess.Popen:
    log_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    fh = (log_dir / f"{name}.log").open("wb")
    proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
    fh.close()
    logger.info("Started %s (pid=%d): %s", name, proc.pid, " ".join(cmd))
    return proc


def _wait_http_health(
    url: str, proc: subprocess.Popen, timeout_s: float = 600.0
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"Service exited early with code {proc.returncode} ({url})"
            )
        try:
            resp = httpx.get(f"{url}/health", timeout=10.0, trust_env=False)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
        time.sleep(1.0)
    raise TimeoutError(f"Timed out waiting for {url}/health")


def _raise_nofile_limit(target: int = 1048576) -> None:
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        desired = target if hard == resource.RLIM_INFINITY else min(target, hard)
        if soft < desired:
            resource.setrlimit(resource.RLIMIT_NOFILE, (desired, hard))
    except (OSError, ValueError):
        pass


def main() -> None:
    _raise_nofile_limit()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    log_dir = Path(cfg["log_dir"])
    host = cfg.get("host", "127.0.0.1")
    common_env = {str(k): str(v) for k, v in (cfg.get("env") or {}).items()}
    procs: list[tuple[str, subprocess.Popen]] = []

    def _shutdown(*_args) -> None:
        logger.info("Stopping all %d RPent services...", len(procs))
        for _name, p in reversed(procs):
            if p.poll() is None:
                p.terminate()
        deadline = time.monotonic() + 10.0
        for _name, p in reversed(procs):
            if p.poll() is None:
                try:
                    p.wait(timeout=max(0.1, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    p.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        # 1. Start local SAM3 backends + Sam3Pool (optional for sim-only nodes)
        sam3_pool_url: str | None = cfg.get("sam3_endpoint")
        if cfg.get("sam3"):
            sam3_cfg = cfg["sam3"]
            sam3_py = sam3_cfg.get("python", sys.executable)
            sam3_host = sam3_cfg.get("host", host)
            backend_host = sam3_cfg.get("backend_host", "127.0.0.1")
            sam3_env = {
                **common_env,
                **{str(k): str(v) for k, v in (sam3_cfg.get("env") or {}).items()},
                "SAM3_CHECKPOINT_PATH": str(sam3_cfg["checkpoint"]),
            }
            backend_urls: list[str] = []
            base_port = int(sam3_cfg["backend_base_port"])
            for idx, gpu in enumerate(sam3_cfg["gpus"]):
                port = base_port + idx
                url = f"http://{backend_host}:{port}"
                backend_urls.append(url)
                svc_name = f"sam3_{idx}_gpu{gpu}"
                procs.append(
                    (
                        svc_name,
                        _spawn(
                            svc_name,
                            [
                                sam3_py,
                                "-m",
                                "rpent.robots.components.sam3_server",
                                "--transport",
                                "http",
                                "--host",
                                backend_host,
                                "--port",
                                str(port),
                                "--cuda-device",
                                str(gpu),
                            ],
                            log_dir,
                            sam3_env,
                        ),
                    )
                )
            for url in backend_urls:
                client = make_rpc_client(url)
                wait_for_ready(client, timeout_s=240.0)
                client.close()
            logger.info("All %d SAM3 backends ready", len(backend_urls))

            extra_backends = [str(u) for u in (sam3_cfg.get("extra_backends") or [])]
            pool_port = int(sam3_cfg["pool_port"])
            sam3_pool_url = f"http://{sam3_host}:{pool_port}"
            pool_cmd = [
                sam3_py,
                "-m",
                "rpent.robots.components.sam3_pool",
                "--host",
                sam3_host,
                "--port",
                str(pool_port),
                "--max-pending",
                str(int(sam3_cfg.get("max_pending", 256))),
                "--timeout-s",
                str(float(sam3_cfg.get("timeout_s", 100.0))),
                "--backends",
                *backend_urls,
                *extra_backends,
            ]
            procs.append(
                ("sam3_pool", _spawn("sam3_pool", pool_cmd, log_dir, common_env))
            )
            client = make_rpc_client(sam3_pool_url)
            wait_for_ready(client, timeout_s=60.0)
            client.close()
            logger.info(
                "Sam3Pool ready at %s (max_pending=%s)",
                sam3_pool_url,
                sam3_cfg.get("max_pending", 256),
            )

        # 2. Start warm simulator pools (optional on router-only nodes)
        sim_procs: list[tuple[str, str, subprocess.Popen]] = []
        for sim_name, sim_cfg in (cfg.get("simulators") or {}).items():
            sim_host = sim_cfg.get("host", host)
            port = int(sim_cfg["port"])
            sim_py = str(sim_cfg.get("python", sys.executable))
            sim_env = {
                **common_env,
                **{str(k): str(v) for k, v in (sim_cfg.get("env") or {}).items()},
            }
            endpoint = sim_cfg.get("sam3_endpoint", sam3_pool_url)
            cmd = [
                sim_py,
                "-m",
                "robots.rl.server",
                "serve",
                "--env",
                sim_name,
                "--host",
                sim_host,
                "--port",
                str(port),
                "--workers",
                str(int(sim_cfg["workers"])),
                "--sim-python",
                sim_py,
                "--sam3-timeout-s",
                str(float(sim_cfg.get("sam3_timeout_s", 120.0))),
                "--max-episodes-per-worker",
                str(int(sim_cfg.get("max_episodes_per_worker", 30))),
                "--max-concurrent-spawns",
                str(int(sim_cfg.get("max_concurrent_spawns", 16))),
                "--log-dir",
                str(log_dir / sim_name),
            ]
            if endpoint:
                cmd.extend(["--sam3-endpoint", str(endpoint)])
            if sim_cfg.get("sleepable"):
                cmd.append("--sleepable")
            if sim_cfg.get("gpus"):
                cmd.extend(["--gpus", *(str(g) for g in sim_cfg["gpus"])])
            if sim_cfg.get("assets_path"):
                cmd.extend(["--assets-path", str(sim_cfg["assets_path"])])
            if sim_cfg.get("libero_type"):
                cmd.extend(["--libero-type", str(sim_cfg["libero_type"])])
            proc = _spawn(f"pool_{sim_name}", cmd, log_dir, sim_env)
            procs.append((f"pool_{sim_name}", proc))
            sim_procs.append((sim_name, f"http://{sim_host}:{port}", proc))

        for sim_name, url, proc in sim_procs:
            status = _wait_http_health(url, proc, timeout_s=600.0)
            logger.info("Simulator pool %s ready at %s: %s", sim_name, url, status)

        # 3. Start multi-host SimPoolRouter frontends (optional)
        router_procs: list[tuple[str, str, subprocess.Popen]] = []
        for router_name, router_cfg in (cfg.get("routers") or {}).items():
            router_host = router_cfg.get("host", host)
            port = int(router_cfg["port"])
            router_py = str(router_cfg.get("python", sys.executable))
            router_env = {
                **common_env,
                **{str(k): str(v) for k, v in (router_cfg.get("env") or {}).items()},
            }
            cmd = [
                router_py,
                "-m",
                "robots.rl.server",
                "router",
                "--env",
                str(router_cfg.get("env_name", router_name)),
                "--host",
                router_host,
                "--port",
                str(port),
                "--request-timeout-s",
                str(float(router_cfg.get("request_timeout_s", 1800.0))),
                "--backends",
                *(str(b) for b in router_cfg["backends"]),
            ]
            proc = _spawn(f"router_{router_name}", cmd, log_dir, router_env)
            procs.append((f"router_{router_name}", proc))
            router_procs.append((router_name, f"http://{router_host}:{port}", proc))

        for router_name, url, proc in router_procs:
            status = _wait_http_health(url, proc, timeout_s=120.0)
            logger.info("Simulator router %s ready at %s: %s", router_name, url, status)

        logger.info("ALL SERVICES READY. Monitoring child processes...")
        while True:
            for name, proc in procs:
                code = proc.poll()
                if code is not None:
                    raise RuntimeError(
                        f"Service {name} exited unexpectedly with code {code}"
                    )
            time.sleep(2.0)
    except Exception:
        logger.exception("Service supervisor failed; shutting down")
        _shutdown()


if __name__ == "__main__":
    main()
