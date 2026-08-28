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

"""Harden the external preliminary driver without modifying the source snapshot."""

from __future__ import annotations

import argparse
import fcntl
import os
import stat
import sys
import tempfile
import time
from pathlib import Path
from types import ModuleType

import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _read_frozen_source(path: Path, expected_sha256: str) -> str:
    from rpent.reproduce.robocasa.secure_script import _read_source

    return _read_source(path, expected_sha256)


def _configure_runtime_sources(driver_source: Path) -> None:
    runtime_root = driver_source.resolve(strict=True).parent.parent
    sources = (
        runtime_root / "external_dependencies/robocasa365",
        runtime_root / "external_dependencies/robocasa365/robosuite",
    )
    for source in reversed(sources):
        metadata = source.lstat()
        if (
            source.is_symlink()
            or not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_mode & 0o022
        ):
            raise RuntimeError(f"runtime dependency root is unsafe: {source}")
        sys.path.insert(0, str(source))


def _write_staged_source(path: Path, source: str) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o400)
    try:
        raw = source.encode("utf-8")
        written = 0
        while written < len(raw):
            written += os.write(descriptor, raw[written:])
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _load_driver(path: Path, source: str) -> ModuleType:
    module = ModuleType("_rpent_external_driver")
    module.__file__ = str(path)
    module.__package__ = ""
    exec(compile(source, str(path), "exec", dont_inherit=True), module.__dict__)
    return module


def _regular_nonempty(path: Path) -> None:
    metadata = path.lstat()
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_size <= 0
    ):
        raise RuntimeError(f"missing or unsafe navview artifact: {path.name}")


def _validate_navview(workdir: Path, step: int) -> None:
    suffix = f"{step:02d}"
    image_path = workdir / f"image_nav_{suffix}.png"
    floor_path = workdir / f"image_nav_floor_{suffix}.png"
    world_path = workdir / f"world_nav_{suffix}.npy"
    for path in (image_path, floor_path, world_path):
        _regular_nonempty(path)
    with Image.open(image_path) as image:
        image.load()
        if image.mode != "RGB" or image.size != (256, 256):
            raise RuntimeError("navview RGB must be a 256x256 image")
    world = np.load(world_path, allow_pickle=False)
    if (
        world.shape != (256, 256, 3)
        or world.dtype != np.float32
        or not np.isfinite(world).all()
    ):
        raise RuntimeError("navview world map must be finite float32[256,256,3]")


def _publish_done_before_deadline(
    original_publish_done, path: str | Path, *, run_id: str
) -> None:
    from rpent.reproduce.robocasa.deadline_supervisor import (
        CONTRACT_NAME,
        DEADLINE_PROTOCOL,
        GATE_NAME,
        RECEIPT_PREFIX,
        SEAL_NAME,
        DeadlineError,
        _atomic_json,
        _open_gate,
        _read_trusted_json,
        _remove_done_marker,
        process_identity,
    )

    marker = Path(path)
    raw_step = marker.name[5:-5] if marker.name.startswith("done_") else ""
    try:
        step = int(raw_step)
    except ValueError as exc:
        raise DeadlineError(f"invalid commit marker name: {marker.name}") from exc
    if marker.name != f"done_{step:02d}.flag":
        raise DeadlineError(f"non-canonical commit marker name: {marker.name}")
    if step == 0:
        original_publish_done(path)
        return

    workdir = marker.parent
    gate_descriptor = _open_gate(workdir / GATE_NAME, create=False)
    try:
        fcntl.flock(gate_descriptor, fcntl.LOCK_EX)
        contract, contract_sha256 = _read_trusted_json(workdir / CONTRACT_NAME)
        driver = contract.get("driver")
        started_ns = contract.get("started_monotonic_ns")
        deadline_ns = contract.get("deadline_monotonic_ns")
        timeout_ns = contract.get("timeout_ns")
        nonce = contract.get("nonce")
        current_identity = process_identity(os.getpid())
        expected_contract_keys = {
            "schema_version",
            "protocol",
            "run_id",
            "nonce",
            "started_monotonic_ns",
            "deadline_monotonic_ns",
            "timeout_ns",
            "driver",
            "external_deadline_sha256",
        }
        expected_driver = (
            {key: current_identity[key] for key in ("pid", "pgid", "start_time_ticks")}
            if current_identity is not None
            else None
        )
        external_sha256 = contract.get("external_deadline_sha256")
        if (
            set(contract) != expected_contract_keys
            or contract.get("schema_version") != 1
            or contract.get("protocol") != DEADLINE_PROTOCOL
            or contract.get("run_id") != run_id
            or driver != expected_driver
            or not isinstance(nonce, str)
            or len(nonce) != 32
            or any(character not in "0123456789abcdef" for character in nonce)
            or type(started_ns) is not int
            or type(deadline_ns) is not int
            or type(timeout_ns) is not int
            or started_ns <= 0
            or timeout_ns <= 0
            or deadline_ns != started_ns + timeout_ns
            or not isinstance(external_sha256, str)
            or len(external_sha256) != 64
            or any(character not in "0123456789abcdef" for character in external_sha256)
        ):
            raise DeadlineError("deadline contract does not match this rollout driver")
        try:
            seal_metadata = (workdir / SEAL_NAME).lstat()
        except FileNotFoundError:
            seal_metadata = None
        if seal_metadata is not None:
            raise SystemExit(75)
        commit_started_ns = time.monotonic_ns()
        if commit_started_ns < started_ns:
            raise DeadlineError("deadline contract starts in the future")
        if commit_started_ns >= deadline_ns:
            raise SystemExit(75)
        receipt_path = workdir / f"{RECEIPT_PREFIX}{step:02d}.json"
        if (
            marker.exists()
            or marker.is_symlink()
            or receipt_path.exists()
            or receipt_path.is_symlink()
        ):
            raise DeadlineError(f"deadline commit step {step} was already published")
        # The marker's completed write, not the pre-write check, is the deadline
        # receipt. The gate keeps the freezer out until a late marker is removed.
        original_publish_done(path)
        published_ns = time.monotonic_ns()
        _atomic_json(
            receipt_path,
            {
                "schema_version": 1,
                "protocol": DEADLINE_PROTOCOL,
                "run_id": run_id,
                "nonce": nonce,
                "step": step,
                "deadline_monotonic_ns": deadline_ns,
                "done_published_monotonic_ns": published_ns,
                "contract_sha256": contract_sha256,
            },
        )
        if published_ns >= deadline_ns:
            _remove_done_marker(marker)
            raise SystemExit(75)
    finally:
        try:
            fcntl.flock(gate_descriptor, fcntl.LOCK_UN)
        finally:
            os.close(gate_descriptor)


def _patch_driver(module: ModuleType, *, run_id: str | None = None) -> None:
    from rpent.reproduce.robocasa.protocol import command_problem

    class EpisodeSuccess(RuntimeError):
        """Control-flow signal raised before any post-success simulator step."""

    driver_class = module.RoboCasaDriver
    original_dump = driver_class.dump_state
    original_execute = driver_class.execute
    original_publish_done = getattr(module, "_publish_done", None)

    if original_publish_done is not None:

        def publish_done(path):
            marker = Path(path)
            is_initial = marker.name == "done_00.flag"
            if run_id is None and not is_initial:
                raise RuntimeError("deadline run identity is unavailable")
            if is_initial:
                original_publish_done(path)
            else:
                assert run_id is not None
                _publish_done_before_deadline(
                    original_publish_done, path, run_id=run_id
                )
            if getattr(module, "_rpent_exit_after_done", False):
                raise SystemExit(0)

        module._publish_done = publish_done

    environment_class = getattr(module, "RoboCasaInteractiveEnv", None)
    if environment_class is not None:
        original_reset = environment_class.reset
        original_step = environment_class.step

        def reset(self):
            observation = original_reset(self)
            self._terminated = bool(self.env._check_success())
            return observation

        def step(self, action):
            if self._terminated:
                raise EpisodeSuccess
            return original_step(self, action)

        environment_class.reset = reset
        environment_class.step = step

    def dump_state(self, step_idx, publish_done=True):
        state = original_dump(self, step_idx, publish_done=False)
        _validate_navview(Path(self.workdir), step_idx)
        state["success"] = bool(self.env.terminated)
        module._rpent_exit_after_done = state["success"]
        module._atomic_write_json(
            f"{self.workdir}/state_{step_idx:02d}.json", state, indent=2
        )
        if publish_done:
            module._publish_done(f"{self.workdir}/done_{step_idx:02d}.flag")
        return state

    def execute(self, command):
        task_language = self.env.current_raw_obs.get(
            "language"
        ) or self.env.env.get_ep_meta().get("lang", "")
        if problem := command_problem(command, task_language=task_language):
            return {"error": f"invalid command schema: {problem}"}
        if self.env.terminated:
            raise SystemExit(0)
        action = command["action"]
        try:
            if action == "move_delta":
                self._vla_desync = True
                return self.move_delta(
                    command["dxyz"],
                    command.get("gripper", "hold"),
                    command.get("step_clip", 0.02),
                    command.get("max_steps", 80),
                )
            if action == "rotate_pitch":
                self._vla_desync = True
                return self.rotate_pitch(
                    command.get("target_pitch", 0.6),
                    command.get("gripper", 1),
                    command.get("n", 12),
                )
            if action == "navigate_to":
                self._vla_desync = True
                return self.navigate_to(
                    command["xy"],
                    command.get("tol", 0.20),
                    command.get("max_steps", 300),
                    command.get("gripper", "hold"),
                )
            if action == "move_base":
                self._vla_desync = True
                return self.move_base(
                    command.get("forward", 0),
                    command.get("lateral", 0),
                    command.get("turn", 0),
                    command.get("steps", 10),
                    command.get("gripper", "hold"),
                )
            return original_execute(self, command)
        except EpisodeSuccess:
            return {"status": "task_success", "success": True}

    driver_class.dump_state = dump_state
    driver_class.execute = execute


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver-source", type=Path, required=True)
    parser.add_argument("--driver-sha256", required=True)
    parser.add_argument("--interactive-env-sha256", required=True)
    parser.add_argument("--rldx-skill-sha256", required=True)
    parser.add_argument("--env", required=True)
    parser.add_argument("--split", choices=("target", "pretrain", "all"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    _configure_runtime_sources(args.driver_source)
    driver_source = _read_frozen_source(args.driver_source, args.driver_sha256)
    interactive_source = _read_frozen_source(
        args.driver_source.parent / "robocasa_interactive_env.py",
        args.interactive_env_sha256,
    )
    skill_source = _read_frozen_source(
        args.driver_source.parent / "rldx_skill.py", args.rldx_skill_sha256
    )
    with tempfile.TemporaryDirectory(prefix="rpent-robocasa-runtime-") as temporary:
        stage = Path(temporary)
        driver_path = stage / "robocasa_interactive_driver.py"
        _write_staged_source(stage / "robocasa_interactive_env.py", interactive_source)
        _write_staged_source(stage / "rldx_skill.py", skill_source)
        module = _load_driver(driver_path, driver_source)
        _patch_driver(module, run_id=args.run_id)
        module._die_with_parent()
        module.RoboCasaDriver(
            args.env,
            args.split,
            args.seed,
            str(args.workdir),
        ).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
