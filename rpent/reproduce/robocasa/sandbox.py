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

"""Adapt the audited RLDX Codex launcher to a configurable rollout root.

The migration launcher is deliberately reused for the preliminary local parity
run.  Its only machine-specific assumption is that workdirs live under /tmp;
this adapter replaces that containment check with the explicit RPent run root.
It also keeps the root-only deadline authority outside the launcher's planner
read-sharing pass.  Fixed-mailbox, setuid, and Landlock logic remains in the
audited launcher.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import stat
import sys
from pathlib import Path
from types import ModuleType

DEADLINE_CONTROL_NAMES = frozenset({"_deadline_commit.gate", "_deadline_contract.json"})


def _load_launcher(path: Path, expected_sha256: str) -> ModuleType:
    """Execute the pinned launcher from one securely opened file description."""
    if len(expected_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha256
    ):
        raise SystemExit("isolation launcher SHA-256 must be lowercase hexadecimal")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SystemExit(f"cannot securely open isolation launcher: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.geteuid()
            or before.st_mode & 0o022
            or before.st_size <= 0
            or before.st_size > 1024 * 1024
        ):
            raise SystemExit(
                "isolation launcher must be owned, regular, immutable, and bounded"
            )
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        source = b"".join(chunks)
        after = os.fstat(descriptor)
        identity = lambda value: (  # noqa: E731 - compact immutable identity tuple
            value.st_dev,
            value.st_ino,
            value.st_uid,
            value.st_gid,
            value.st_mode,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )
        if identity(before) != identity(after) or len(source) != before.st_size:
            raise SystemExit("isolation launcher changed while it was being read")
    finally:
        os.close(descriptor)
    if hashlib.sha256(source).hexdigest() != expected_sha256:
        raise SystemExit("isolation launcher does not match its frozen SHA-256")
    try:
        text = source.decode("utf-8")
    except UnicodeError as exc:
        raise SystemExit("isolation launcher must be UTF-8 source") from exc
    module = ModuleType("_rpent_rldx_isolation")
    module.__file__ = str(path)
    module.__package__ = ""
    exec(compile(text, str(path), "exec", dont_inherit=True), module.__dict__)
    return module


def _inside(path: Path, parent: Path, label: str) -> Path:
    resolved = path.resolve()
    root = parent.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise SystemExit(f"{label} must be inside {root}: {resolved}") from exc
    if not relative.parts:
        raise SystemExit(f"{label} cannot be the rollout root itself: {resolved}")
    return resolved


def _rollout_root_problem(path: Path) -> str | None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        return f"cannot inspect rollout root: {exc}"
    if path.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
        return f"rollout root must be a real directory: {path}"
    if metadata.st_uid != os.geteuid():
        return "rollout root must be owned by the caller"
    if stat.S_IMODE(metadata.st_mode) != 0o711:
        return "rollout root must have exact mode 0711"
    resolved = path.resolve()
    for parent in reversed(resolved.parents):
        try:
            parent_metadata = parent.stat()
        except OSError as exc:
            return f"cannot inspect rollout ancestor {parent}: {exc}"
        if parent_metadata.st_mode & stat.S_IXOTH == 0:
            return f"rollout ancestor is not planner-traversable: {parent}"
    return None


def _deadline_control_metadata(
    path: Path,
    descriptor: int,
    *,
    owner_uid: int,
    owner_gid: int,
) -> os.stat_result:
    try:
        opened = os.fstat(descriptor)
        linked = path.lstat()
    except OSError as exc:
        raise SystemExit(f"cannot inspect deadline control {path.name}: {exc}") from exc
    if (
        not stat.S_ISREG(opened.st_mode)
        or not stat.S_ISREG(linked.st_mode)
        or opened.st_nlink != 1
        or (opened.st_dev, opened.st_ino) != (linked.st_dev, linked.st_ino)
        or opened.st_uid != owner_uid
        or opened.st_gid != owner_gid
        or stat.S_IMODE(opened.st_mode) != 0o600
    ):
        raise SystemExit(f"deadline control is unsafe: {path.name}")
    return opened


def _control_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _pin_deadline_controls(
    workdir: Path, *, owner_uid: int, owner_gid: int
) -> tuple[dict[str, int], dict[str, tuple[int, ...]]]:
    present = {
        entry.name for entry in workdir.iterdir() if entry.name.startswith("_deadline_")
    }
    if present != DEADLINE_CONTROL_NAMES:
        raise SystemExit(
            "deadline control set differs before isolation: "
            f"expected {sorted(DEADLINE_CONTROL_NAMES)}, found {sorted(present)}"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptors: dict[str, int] = {}
    identities: dict[str, tuple[int, ...]] = {}
    try:
        for name in sorted(DEADLINE_CONTROL_NAMES):
            path = workdir / name
            descriptor = os.open(path, flags)
            descriptors[name] = descriptor
            identities[name] = _control_identity(
                _deadline_control_metadata(
                    path,
                    descriptor,
                    owner_uid=owner_uid,
                    owner_gid=owner_gid,
                )
            )
    except BaseException:
        for descriptor in descriptors.values():
            os.close(descriptor)
        raise
    return descriptors, identities


def _prepare_isolated_workdir(
    module: ModuleType,
    workdir: Path,
    uid: int,
    gid: int,
    *,
    owner_uid: int = 0,
    owner_gid: int = 0,
) -> tuple[Path, Path]:
    """Apply the pinned launcher's policy without exposing deadline controls."""
    metadata = workdir.lstat()
    if (
        workdir.is_symlink()
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != owner_uid
        or metadata.st_gid != owner_gid
    ):
        raise SystemExit(
            f"workdir must be an authority-owned non-symlink directory: {workdir}"
        )
    descriptors, identities = _pin_deadline_controls(
        workdir, owner_uid=owner_uid, owner_gid=owner_gid
    )
    try:
        task_memory = module.seal_task_memory(workdir)
        for entry in list(workdir.iterdir()):
            if entry.name in DEADLINE_CONTROL_NAMES:
                continue
            if task_memory is not None and entry == task_memory:
                continue
            entry_metadata = entry.lstat()
            if not stat.S_ISREG(entry_metadata.st_mode):
                raise SystemExit(
                    f"unexpected non-regular workdir entry before isolation: {entry}"
                )
            if entry_metadata.st_uid != owner_uid:
                raise SystemExit(
                    f"driver artifact must remain authority-owned: {entry}"
                )
            os.chown(entry, owner_uid, gid)
            mode = 0o600 if entry.name in ("driver.log", "agent.log") else 0o640
            os.chmod(entry, mode)

        private_paths = [workdir / name for name in module.PLANNER_WRITABLE_DIRECTORIES]
        for path in private_paths:
            module._create_private_directory(path, uid, gid)
        for name in module.PLANNER_WRITABLE_MAILBOXES:
            module._create_fixed_mailbox(workdir / name, gid)

        os.chown(workdir, owner_uid, gid)
        os.chmod(workdir, 0o2750)
        module._publish_mailbox_marker(workdir, uid, gid)
        return workdir / ".codex", workdir / "tmp"
    finally:
        try:
            for name, descriptor in descriptors.items():
                after = _deadline_control_metadata(
                    workdir / name,
                    descriptor,
                    owner_uid=owner_uid,
                    owner_gid=owner_gid,
                )
                if _control_identity(after) != identities[name]:
                    raise SystemExit(
                        f"deadline control changed during isolation: {name}"
                    )
        finally:
            for descriptor in descriptors.values():
                os.close(descriptor)


def _anonymous_secret_fd() -> int:
    """Create an anonymous, close-on-exec regular file for one credential."""
    creator = getattr(os, "memfd_create", None)
    if creator is not None:
        return creator("rpent-planner-key", getattr(os, "MFD_CLOEXEC", 0x0001))

    flags = getattr(os, "O_TMPFILE", 0)
    if not flags:
        raise SystemExit("anonymous planner credential storage is unavailable")
    last_error: OSError | None = None
    for directory in ("/dev/shm", "/tmp"):
        descriptor: int | None = None
        try:
            descriptor = os.open(
                directory, flags | os.O_RDWR | getattr(os, "O_CLOEXEC", 0), 0o600
            )
            os.fchmod(descriptor, 0o600)
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 0
                or stat.S_IMODE(metadata.st_mode) != 0o600
            ):
                raise OSError("O_TMPFILE did not create the required anonymous inode")
            os.set_inheritable(descriptor, False)
            return descriptor
        except OSError as exc:
            last_error = exc
            if descriptor is not None:
                os.close(descriptor)
    raise SystemExit(
        "anonymous planner credential storage is unavailable"
    ) from last_error


def _secret_memfd(path: Path) -> tuple[int, str]:
    """Copy an owned mode-0600 secret into an anonymous, exec-closing fd."""
    metadata = path.lstat()
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or metadata.st_size <= 1
        or metadata.st_size > 16 * 1024
    ):
        raise SystemExit("API key file must be owned, regular, mode 0600, and bounded")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    source = os.open(path, flags)
    try:
        opened = os.fstat(source)
        identity = (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
        if identity != (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
        ):
            raise SystemExit("API key file changed while opening")
        raw = os.read(source, metadata.st_size + 1)
        after = os.fstat(source)
        if (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ) != identity or len(raw) != metadata.st_size:
            raise SystemExit("API key file changed while reading")
    finally:
        os.close(source)
    try:
        text = raw.decode("utf-8")
    except UnicodeError as exc:
        raise SystemExit("API key file must contain UTF-8 text") from exc
    if len(text.splitlines()) != 1 or not text.strip():
        raise SystemExit("API key file must contain exactly one non-empty line")
    descriptor = _anonymous_secret_fd()
    try:
        remaining = memoryview(raw)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("anonymous credential write made no progress")
            remaining = remaining[written:]
        os.lseek(descriptor, 0, os.SEEK_SET)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor, f"/proc/self/fd/{descriptor}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--launcher-sha256", required=True)
    parser.add_argument("--rollout-root", type=Path, required=True)
    parser.add_argument("launcher_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if not args.launcher_args or args.launcher_args[0] != "--":
        parser.error("expected audited launcher arguments after --")

    raw_rollout_root = args.rollout_root.expanduser()
    if problem := _rollout_root_problem(raw_rollout_root):
        raise SystemExit(problem)
    rollout_root = raw_rollout_root.resolve()
    if not rollout_root.is_dir():
        raise SystemExit(f"rollout root must be a real directory: {rollout_root}")
    launcher_args = list(args.launcher_args[1:])
    if launcher_args.count("--key-file") != 1:
        raise SystemExit("audited launcher requires exactly one --key-file")
    key_index = launcher_args.index("--key-file")
    if key_index + 1 >= len(launcher_args):
        raise SystemExit("audited launcher --key-file has no value")
    module = _load_launcher(args.launcher.expanduser(), args.launcher_sha256)
    secret_fd, secret_path = _secret_memfd(Path(launcher_args[key_index + 1]))
    launcher_args[key_index + 1] = secret_path

    original_install_landlock = module.install_landlock_boundary

    def prepare_isolated_workdir(workdir: Path, uid: int, gid: int):
        _prepare_isolated_workdir(module, workdir, uid, gid)
        return Path(".codex"), Path("tmp")

    def install_landlock_boundary(_workdir: Path) -> None:
        original_install_landlock(Path("."))

    module.prepare_isolated_workdir = prepare_isolated_workdir
    module.install_landlock_boundary = install_landlock_boundary

    def require_inside(path: Path, _ignored: Path, label: str) -> Path:
        return _inside(path, rollout_root, label)

    module.require_inside = require_inside
    previous = sys.argv
    try:
        sys.argv = [str(args.launcher), *launcher_args]
        result = module.main()
    finally:
        sys.argv = previous
        os.close(secret_fd)
    return int(result or 0)


if __name__ == "__main__":
    raise SystemExit(main())
