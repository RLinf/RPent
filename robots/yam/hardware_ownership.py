# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
"""Exclusive RPent ownership and a preflight check for existing CAN sockets."""

from __future__ import annotations

import fcntl
import hashlib
import os
from pathlib import Path


class HardwareLease:
    """Lock RPent writers and reject pre-existing SocketCAN subscriptions.

    Other programs do not share these locks. The operator must still keep
    collectors and other controllers stopped throughout the session.
    """

    def __init__(self, channels: list[str], *, lock_dir: Path = Path("/tmp")):
        if len(channels) != len(set(channels)) or not all(channels):
            raise ValueError("YAM follower CAN channels must be nonempty and distinct")
        self.channels = channels
        self.lock_dir = lock_dir
        self._files = []

    def acquire(self) -> None:
        try:
            for channel in sorted(self.channels):
                digest = hashlib.sha256(channel.encode()).hexdigest()[:16]
                stream = (self.lock_dir / f"rpent-yam-can-{digest}.lock").open("a+")
                self._files.append(stream)
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise RuntimeError(
                        f"CAN {channel} is owned by another RPent process"
                    ) from error
                stream.seek(0)
                stream.truncate()
                stream.write(f"pid={os.getpid()} channel={channel}\n")
                stream.flush()
        except BaseException:
            self.close()
            raise

    def check_subscriptions(self, *, proc_root: Path = Path("/proc/net/can")) -> None:
        files = list(proc_root.glob("rcvlist_*"))
        if not files:
            raise RuntimeError(
                "Cannot verify SocketCAN ownership: receive lists unavailable"
            )
        for path in files:
            for line in path.read_text().splitlines():
                fields = line.split()
                if fields and fields[0] in {*self.channels, "any"}:
                    raise RuntimeError(
                        f"Existing CAN subscription on {fields[0]}; stop its owner normally "
                        "with all four arms supported before starting RPent"
                    )

    def close(self) -> None:
        for stream in reversed(self._files):
            stream.close()
        self._files.clear()
