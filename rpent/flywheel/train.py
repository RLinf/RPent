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

"""Launch Pi0.5 SFT through an official RLinf checkout."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any


def train_rlinf(
    *,
    dataset: Path | str,
    checkpoint: Path | str,
    rlinf_root: Path | str,
    output_dir: Path | str,
    max_steps: int,
    save_interval: int = 100,
    cuda_device: int = 0,
) -> dict[str, Any]:
    """Run RLinf's native VLA SFT entry point with the Flywheel config."""
    dataset = Path(dataset).expanduser().resolve()
    checkpoint = Path(checkpoint).expanduser().resolve()
    rlinf_root = Path(rlinf_root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    python = rlinf_root / ".venv" / "bin" / "python"
    trainer = rlinf_root / "examples" / "sft" / "train_vla_sft.py"
    if not (dataset / "meta" / "info.json").is_file():
        raise ValueError(f"not a LeRobot dataset: {dataset}")
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if not trainer.is_file():
        raise ValueError(f"not an RLinf source checkout: {rlinf_root}")
    if not python.is_file():
        raise FileNotFoundError(f"RLinf virtual environment not found: {python}")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if save_interval == 0 or save_interval < -1:
        raise ValueError("save_interval must be positive or -1")
    if output_dir.exists():
        raise FileExistsError(f"training output already exists: {output_dir}")
    rlinf_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=rlinf_root, text=True
    ).strip()
    output_dir.mkdir(parents=True)

    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(cuda_device),
            "PYTHONPATH": str(rlinf_root),
            "RPENT_FLYWHEEL_DATASET": str(dataset),
            "RPENT_FLYWHEEL_CHECKPOINT": str(checkpoint),
            "RPENT_FLYWHEEL_OUTPUT": str(output_dir),
            "RLINF_SFT_CONFIG": str(rlinf_root / "examples" / "sft" / "config"),
        }
    )
    config_dir = Path(__file__).with_name("config")
    command = [
        str(python),
        str(trainer),
        "--config-path",
        str(config_dir),
        "--config-name",
        "pi05_sft",
        f"runner.max_steps={max_steps}",
        f"runner.save_interval={save_interval}",
        f"actor.optim.total_training_steps={max_steps}",
    ]
    log_path = output_dir / "rlinf.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            cwd=rlinf_root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )

    report = {
        "command": command,
        "cuda_device": cuda_device,
        "dataset": str(dataset),
        "checkpoint": str(checkpoint),
        "rlinf_commit": rlinf_commit,
        "rlinf_root": str(rlinf_root),
        "rlinf_log": str(log_path),
        "max_steps": max_steps,
        "save_interval": save_interval,
        "return_code": process.returncode,
        "passed": process.returncode == 0,
    }
    report_path = output_dir / "training_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not report["passed"]:
        raise RuntimeError(f"RLinf training failed; inspect {report_path}")
    return report
