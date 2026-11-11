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

"""Configuration for the Cosmos3 RPC worker and its borrowed SGLang service."""

import argparse
from pathlib import Path

from rpent.robots.components.cosmos3 import USES_SESSIONS as USES_SESSIONS
from rpent.robots.components.cosmos3.adapter import ADAPTERS as ADAPTERS

OPTION_FIELDS = ("cosmos3_config", "cosmos3_sglang_endpoint")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group(
        "Cosmos3 local RPC worker",
        "Connect the RPC worker to a separately deployed Cosmos3 SGLang service. "
        "The checkpoint and RLinf configuration must match that service.",
    )
    group.add_argument(
        "--cosmos3-config",
        metavar="YAML",
        help="Resolved RLinf rollout/model and env configuration (required)",
    )
    group.add_argument(
        "--cosmos3-sglang-endpoint",
        metavar="URL",
        help="Native SGLang action service, e.g. http://127.0.0.1:30000 (required)",
    )


def validate_options(args: argparse.Namespace) -> None:
    """Require the RLinf policy configuration and native service address."""
    if not args.cosmos3_config or not Path(args.cosmos3_config).expanduser().is_file():
        raise ValueError("--cosmos3-config must reference a local file")
    if not args.cosmos3_sglang_endpoint:
        raise ValueError(
            "--cosmos3-sglang-endpoint must reference a deployed Cosmos3 service"
        )


def worker_arguments(args: argparse.Namespace) -> list[str]:
    """Pass model-specific settings to the RPC entry point."""
    return [
        "--config",
        str(Path(args.cosmos3_config).expanduser().resolve()),
        "--sglang-endpoint",
        args.cosmos3_sglang_endpoint,
    ]
