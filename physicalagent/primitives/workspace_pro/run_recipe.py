#!/usr/bin/env python3
"""Compatibility wrapper for the root-level trace replay tool."""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    runpy.run_path(str(root / "replay_trace.py"), run_name="__main__")
