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

"""Execute one pinned external Python script from securely opened source bytes."""

from __future__ import annotations

import argparse
import hashlib
import os
import stat
import sys
from pathlib import Path


def _read_source(path: Path, expected_sha256: str) -> str:
    if len(expected_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha256
    ):
        raise SystemExit("expected SHA-256 must be lowercase hexadecimal")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SystemExit(f"cannot securely open external script: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.geteuid()
            or before.st_mode & 0o022
            or before.st_size <= 0
            or before.st_size > 4 * 1024 * 1024
        ):
            raise SystemExit(
                "external script must be owned, regular, immutable, and bounded"
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
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_uid,
            before.st_gid,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_uid,
            after.st_gid,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if before_identity != after_identity or len(source) != before.st_size:
            raise SystemExit("external script changed while it was being read")
    finally:
        os.close(descriptor)
    if hashlib.sha256(source).hexdigest() != expected_sha256:
        raise SystemExit("external script does not match its frozen SHA-256")
    try:
        return source.decode("utf-8")
    except UnicodeError as exc:
        raise SystemExit("external script must be UTF-8 source") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if not args.script_args or args.script_args[0] != "--":
        parser.error("expected external script arguments after --")
    source = _read_source(args.source, args.sha256)
    previous = sys.argv
    namespace = {
        "__name__": "__main__",
        "__file__": str(args.source),
        "__package__": None,
        "__builtins__": __builtins__,
    }
    try:
        sys.argv = [str(args.source), *args.script_args[1:]]
        exec(
            compile(source, str(args.source), "exec", dont_inherit=True),
            namespace,
        )
    finally:
        sys.argv = previous
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
