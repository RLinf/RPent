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

"""Console entry point for the reviewed BEHAVIOR dual-venv installer."""

from __future__ import annotations

import argparse
import subprocess
from collections.abc import Sequence
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        prog="behavior-install-runtime",
        description=(
            "Install the reviewed BEHAVIOR dual-venv runtime. Paths and version "
            "pins are controlled by the environment variables documented in "
            "the BEHAVIOR usage guide."
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    _build_parser().parse_args(argv)
    script = Path(__file__).with_name("install_behavior_runtime.sh")
    if not script.is_file():
        raise RuntimeError(f"packaged BEHAVIOR installer is missing: {script}")
    return subprocess.run(["bash", str(script)], check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
