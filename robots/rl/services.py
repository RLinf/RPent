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

"""Own a node's SAM3 workers, pooled endpoint and LIBERO episode backend."""

import argparse
import json
import os
import signal
import socket
import sys
import time
from contextlib import ExitStack
from pathlib import Path

from rpent.utils.daemon import ProcessDaemon
from rpent.utils.logging import get_logger
from rpent.utils.rpc import wait_for_endpoint

logger = get_logger("rl_services")


def main() -> None:
    """Start local services with the caller's interpreter and clean them up together."""
    parser = argparse.ArgumentParser(description=__doc__)
    from robots.rl.contracts import CAMERAS

    parser.add_argument("--simulator", choices=tuple(CAMERAS), default="libero")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--envs-per-gpu", type=int, default=3)
    parser.add_argument("--max-initializing-per-gpu", type=int, default=2)
    parser.add_argument("--sam3-per-gpu", type=int, default=1)
    parser.add_argument("--sam3-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sam-port", type=int, default=28424)
    parser.add_argument("--backend-port", type=int, default=28490)
    args = parser.parse_args()
    try:
        gpus = [int(value) for value in args.gpus.split(",")]
    except ValueError:
        parser.error("--gpus must contain comma-separated nonnegative GPU indices")
    if (
        not gpus
        or len(set(gpus)) != len(gpus)
        or min(gpus) < 0
        or args.envs_per_gpu < 1
        or args.max_initializing_per_gpu < 1
        or args.sam3_per_gpu < 1
    ):
        parser.error(
            "GPUs must be unique nonnegative indices; per-GPU counts must be positive"
        )
    if not args.sam3_checkpoint.is_file():
        parser.error(f"SAM3 checkpoint not found: {args.sam3_checkpoint}")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    def stop(_signum, _frame):
        raise SystemExit(0)

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, stop)
    with ExitStack() as cleanup:
        daemons = []

        def start(name, module, arguments, environment=None):
            log = args.output_dir / f"{name}.log"
            log.touch()
            daemon = ProcessDaemon(
                name=name,
                cmd=[sys.executable, "-m", module, *arguments, "--parent-watch"],
                env_overrides=environment,
                log_path=str(log),
                start_new_session=True,
            )
            daemon.start()
            cleanup.callback(daemon.stop)
            daemons.append(daemon)
            return daemon

        sam_workers = [
            start(
                f"sam3-gpu{gpu}-{replica}",
                "rpent.robots.components.sam3_server",
                [
                    "--cuda-device",
                    str(gpu),
                    "--port",
                    "0",
                    "--metrics",
                    str(args.output_dir / f"sam3-gpu{gpu}-{replica}.jsonl"),
                ],
                {"SAM3_CHECKPOINT_PATH": str(args.sam3_checkpoint.resolve())},
            )
            for gpu in gpus
            for replica in range(args.sam3_per_gpu)
        ]
        endpoints = [wait_for_endpoint(worker, timeout_s=600) for worker in sam_workers]
        pool = start(
            "sam3-pool",
            "rpent.robots.components.sam3_pool",
            [
                "--backends",
                *endpoints,
                "--port",
                str(args.sam_port),
                "--timeout-s",
                "600",
                "--metrics",
                str(args.output_dir / "sam3-pool.jsonl"),
            ],
        )
        pool_endpoint = wait_for_endpoint(pool)
        backend = start(
            "backend",
            "robots.rl.remote_manager",
            [
                "--simulator",
                args.simulator,
                "--output-dir",
                str(args.output_dir / "episodes"),
                "--gpu-slots",
                ",".join(f"{gpu}:{args.envs_per_gpu}" for gpu in gpus),
                "--max-initializing-per-gpu",
                str(args.max_initializing_per_gpu),
                "--sam3-endpoint",
                pool_endpoint,
                "--port",
                str(args.backend_port),
            ],
        )
        manifest = {
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "python": sys.executable,
            "prefix": sys.prefix,
            "gpus": gpus,
            "envs_per_gpu": args.envs_per_gpu,
            "max_initializing_per_gpu": args.max_initializing_per_gpu,
            "sam3_per_gpu": args.sam3_per_gpu,
            "sam3_workers": endpoints,
            "sam3_endpoint": pool_endpoint,
            "backend_endpoint": wait_for_endpoint(backend),
        }
        (args.output_dir / "services.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        logger.info("Local LIBERO services ready: %s", manifest)
        while True:
            time.sleep(1)
            for daemon in daemons:
                if daemon.poll() is not None:
                    raise RuntimeError(
                        f"{daemon.name} exited with code {daemon.poll()}; see {daemon.log_path}"
                    )


if __name__ == "__main__":
    main()
