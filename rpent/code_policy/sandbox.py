# Copyright 2026 The RPent Authors.
# SPDX-License-Identifier: Apache-2.0

"""Construct the filesystem and process boundary for the Python worker."""

import importlib.util
import shutil
import sys
import sysconfig
from pathlib import Path


def bubblewrap() -> str:
    """Return the executable required for Code-as-Policy on Linux."""
    executable = shutil.which("bwrap") if sys.platform == "linux" else None
    if not executable:
        raise ValueError("--policy-mode code requires Bubblewrap (bwrap) on Linux")
    return executable


def worker_command(scratch: Path) -> list[str]:
    """Mount the selected interpreter and its dependencies read-only."""
    python = Path(sys.executable).resolve()
    # A virtualenv can report its own lib directory as platstdlib.
    library_roots = dict.fromkeys(
        (
            Path(sysconfig.get_path("stdlib")),
            Path(
                sysconfig.get_path(
                    "platstdlib", vars={"platbase": sys.base_exec_prefix}
                )
            ),
        )
    )
    command = [
        bubblewrap(),
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--clearenv",
        "--setenv",
        "HOME",
        "/scratch",
        "--setenv",
        "TMPDIR",
        "/scratch",
        "--dev",
        "/dev",
        "--bind",
        str(scratch),
        "/scratch",
        "--chdir",
        "/scratch",
        "--ro-bind",
        str(python),
        str(python),
        "--ro-bind",
        str(Path(__file__).with_name("worker.py")),
        "/worker.py",
    ]
    # Standard Linux loader and libraries. Do not mount system Python packages.
    multiarch = sysconfig.get_config_var("MULTIARCH")
    for lib in (Path("/lib"), Path("/lib64"), Path("/usr/lib"), Path("/usr/lib64")):
        for library in sorted(lib.glob("*.so*")):
            if library.is_file():
                command.extend(("--ro-bind", str(library), str(library)))
        if multiarch and (lib / multiarch).is_dir():
            path = str(lib / multiarch)
            command.extend(("--ro-bind", path, path))

    # Shared CPython installations outside the system library directories.
    soname = sysconfig.get_config_var("INSTSONAME")
    if soname:
        library = Path(sys.base_prefix) / "lib" / soname
        if library.is_file():
            command.extend(("--ro-bind", str(library), str(library)))
            command.extend(("--setenv", "LD_LIBRARY_PATH", str(library.parent)))
    for root in library_roots:
        command.extend(("--ro-bind", str(root), str(root)))
        for name in ("site-packages", "dist-packages"):
            if (root / name).is_dir():
                command.extend(("--tmpfs", str(root / name)))

    # Expose NumPy without inheriting editable source paths or other packages.
    numpy = importlib.util.find_spec("numpy")
    if numpy is not None and numpy.origin is not None:
        package = Path(numpy.origin).parent
        command.extend(("--ro-bind", str(package), "/packages/numpy"))
        if (package.parent / "numpy.libs").is_dir():
            command.extend(
                (
                    "--ro-bind",
                    str(package.parent / "numpy.libs"),
                    "/packages/numpy.libs",
                )
            )
    return [*command, "--", str(python), "-I", "-S", "-u", "/worker.py"]
