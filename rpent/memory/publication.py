# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

"""Durable outcomes for the existing memory corpus merge operation."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from rpent.memory import MemoryManager
from rpent.utils.logging import get_logger

logger = get_logger("memory")


def _write_record(path: Path, record: dict) -> None:
    """Replace the publication record without exposing partial JSON."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(record, stream, indent=2, default=str)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def merge_run_memory(
    memory: MemoryManager, *, cell_tag: str, run_state_dir: str | Path, solved: bool
) -> dict:
    """Record merge progress and results while preserving the manager's behavior.

    A failed merge may have made partial changes. Recording an outcome does not
    roll back the corpus or alter the existing inbox/conflict workflow.
    """
    root = Path(run_state_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "memory-merge.json"
    record = {
        "status": "started",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "cell_tag": cell_tag,
        "run_state_dir": str(root),
        "solved": solved,
    }
    _write_record(path, record)
    try:
        result = memory.merge_memory(
            cell_tag=cell_tag, run_state_dir=run_state_dir, solved=solved
        )
    except Exception as exc:
        record.update(
            status="failed",
            finished_at=datetime.now(timezone.utc).isoformat(),
            error_type=type(exc).__name__,
            error=str(exc),
            partial_changes_possible=True,
        )
        try:
            _write_record(path, record)
        except Exception:
            logger.exception("Could not persist the memory merge failure")
        raise
    record.update(
        status="completed",
        finished_at=datetime.now(timezone.utc).isoformat(),
        result=result,
    )
    _write_record(path, record)
    return result
