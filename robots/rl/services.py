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

"""Start the local services declared in a shared simulator deployment."""

import argparse
import signal
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import yaml

from robots.rl.contracts import CAMERAS
from rpent.utils.daemon import ProcessDaemon
from rpent.utils.logging import get_logger
from rpent.utils.rpc import wait_for_endpoint, wait_for_ready
from rpent.utils.rpc.http_rpc import HttpRpcClient

logger = get_logger("rl_services")


def _listener(service: dict) -> list[str]:
    return ["--host", service.get("host", "127.0.0.1"), "--port", str(service["port"])]


def _gpu_counts(counts: dict[int, int]) -> dict[int, int]:
    if not counts or any(
        type(gpu) is not int or gpu < 0 or type(count) is not int or count < 1
        for gpu, count in counts.items()
    ):
        raise ValueError("expected nonnegative GPU ordinals with positive capacities")
    return counts


def local_services(
    config: dict, node: str
) -> list[tuple[str, str, str, list[str], dict[str, str]]]:
    """Compile this node's backend and router commands before opening any process."""
    nodes = config["nodes"]
    row = nodes[node]
    output = Path(row["output_dir"])
    commands = []
    for simulator, backend in row.get("backends", {}).items():
        if simulator not in CAMERAS:
            raise ValueError(f"unknown simulator: {simulator}")
        slots = _gpu_counts(backend["gpu_slots"])
        commands.append(
            (
                f"{simulator}-toolkit",
                backend["python"],
                "robots.rl.remote_manager",
                [
                    "--simulator",
                    simulator,
                    "--gpu-slots",
                    ",".join(f"{gpu}:{count}" for gpu, count in slots.items()),
                    "--max-initializing-per-gpu",
                    str(backend.get("max_initializing_per_gpu", 2)),
                    "--sam3-endpoint",
                    backend["sam3_endpoint"],
                    "--output-dir",
                    str(output / simulator),
                    *_listener(backend),
                ],
                backend.get("env", {}),
            )
        )
    for simulator, router in row.get("routers", {}).items():
        group_size = router["group_size"]
        peers = []
        for peer in router["nodes"]:
            backend = nodes[peer]["backends"][simulator]
            slots = sum(_gpu_counts(backend["gpu_slots"]).values())
            peers.append((backend["endpoint"], slots))
        if not peers or group_size < 1 or sum(slots for _, slots in peers) < group_size:
            raise ValueError(f"{simulator} router needs capacity for a complete group")
        commands.append(
            (
                f"{simulator}-router",
                router.get("python", sys.executable),
                "robots.rl.router",
                [
                    *_listener(router),
                    "--group-size",
                    str(group_size),
                    *[
                        part
                        for url, slots in peers
                        for part in ("--backend", f"{url}={slots}")
                    ],
                ],
                {},
            )
        )
    return commands


def run_node(config: dict, node: str) -> None:
    """Start this node's SAM3 pool, simulator services and optional routers."""
    commands = local_services(config, node)
    row = config["nodes"][node]
    sam3 = row.get("sam3")
    output = Path(row["output_dir"])
    if sam3:
        replicas = _gpu_counts(sam3["replicas"])
        checkpoint = Path(sam3["checkpoint"])
        if not checkpoint.is_file():
            raise ValueError(f"SAM3 checkpoint not found: {checkpoint}")
    output.mkdir(parents=True, exist_ok=False)

    def stop(_signum, _frame):
        raise SystemExit(0)

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, stop)
    with ExitStack() as cleanup:
        daemons = []

        def start(name, python, module, args, env):
            log = output / f"{name}.log"
            log.touch()
            daemon = ProcessDaemon(
                name=name,
                cmd=[python, "-m", module, *args, "--parent-watch"],
                env_overrides={key: str(value) for key, value in env.items()},
                log_path=str(log),
                start_new_session=True,
            )
            cleanup.callback(daemon.stop)
            daemon.start()
            daemons.append(daemon)
            return daemon

        def ready(daemon):
            endpoint = wait_for_endpoint(daemon, timeout_s=600)
            # The process announces its bound endpoint before serving requests.
            client = HttpRpcClient(endpoint)
            try:
                wait_for_ready(client, timeout_s=30, daemon=daemon)
            finally:
                client.close()
            return endpoint

        if sam3:
            workers = []
            for gpu, count in replicas.items():
                for replica in range(count):
                    name = f"sam3-gpu{gpu}-{replica}"
                    workers.append(
                        start(
                            name,
                            sam3["python"],
                            "rpent.robots.components.sam3_server",
                            [
                                "--cuda-device",
                                str(gpu),
                                "--port",
                                "0",
                                "--metrics",
                                str(output / f"{name}.jsonl"),
                            ],
                            {
                                **sam3.get("env", {}),
                                "SAM3_CHECKPOINT_PATH": str(checkpoint.resolve()),
                            },
                        )
                    )
            pool = start(
                "sam3-pool",
                sam3["python"],
                "rpent.robots.components.sam3_pool",
                [
                    *_listener(sam3),
                    "--backends",
                    *map(ready, workers),
                    "--timeout-s",
                    str(sam3.get("timeout_s", 600)),
                    "--max-pending",
                    str(sam3.get("max_pending", 256)),
                    "--metrics",
                    str(output / "sam3-pool.jsonl"),
                ],
                sam3.get("env", {}),
            )
            logger.info("SAM3 pool ready on %s", ready(pool))

        for command in commands:
            daemon = start(*command)
            logger.info("%s ready on %s", daemon.name, ready(daemon))
        while True:
            for daemon in daemons:
                if (code := daemon.poll()) is not None:
                    raise RuntimeError(
                        f"{daemon.name} exited with code {code}; see {daemon.log_path}"
                    )
            time.sleep(1)


def main() -> None:
    """Run a single node from the shared deployment YAML."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--node", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    run_node(config, args.node)


if __name__ == "__main__":
    main()
