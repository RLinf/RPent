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

"""Resolve a UniWAM resource bundle for the shared AM launcher."""

import argparse
from pathlib import Path

from rpent.robots.components.uniwam import USES_SESSIONS as USES_SESSIONS
from rpent.robots.components.uniwam.adapter import ADAPTERS as ADAPTERS

OPTION_FIELDS = ()


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """UniWAM uses the common checkpoint and Python options."""


def validate_options(args: argparse.Namespace) -> None:
    """Require the complete policy and checkpoint bundle before worker startup."""
    root = Path(args.am_checkpoint).expanduser().resolve()
    for name in ("policy", "checkpoint/pytorch_model", "wan", "vlm"):
        if not (root / name).is_dir():
            raise ValueError(f"UniWAM resource directory not found: {root / name}")
    for name in ("policy/deploy_policy.py", "checkpoint/config.json"):
        if not (root / name).is_file():
            raise ValueError(f"UniWAM resource file not found: {root / name}")
    args.am_checkpoint = str(root)


def worker_arguments(args: argparse.Namespace) -> list[str]:
    """The worker resolves resources relative to its checkpoint bundle."""
    return []
