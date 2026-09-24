#!/usr/bin/env python3
# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS.
"""Summarize provider-reported RoboDojo prompt cache usage from decision logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize(log_root: Path, run_prefix: str) -> dict[str, object]:
    totals = {
        "decisions": 0,
        "decisions_with_usage": 0,
        "model_requests": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_input_tokens": 0,
        "cache_write_tokens": 0,
    }
    writes_reported = False
    runs: set[str] = set()
    for path in log_root.glob(f"{run_prefix}*/episode_*/decision_*.json"):
        runs.add(path.parents[1].name)
        totals["decisions"] += 1
        record = json.loads(path.read_text(encoding="utf-8"))
        usage = record.get("usage") or {}
        if not usage:
            continue
        totals["decisions_with_usage"] += 1
        totals["model_requests"] += int(record.get("attempts") or 0)
        totals["input_tokens"] += int(usage.get("prompt_tokens") or 0)
        totals["output_tokens"] += int(usage.get("completion_tokens") or 0)
        details = usage.get("prompt_tokens_details") or {}
        totals["cached_input_tokens"] += int(details.get("cached_tokens") or 0)
        if "cache_write_tokens" in details:
            writes_reported = True
            totals["cache_write_tokens"] += int(details["cache_write_tokens"] or 0)
    if totals["cached_input_tokens"] > totals["input_tokens"]:
        raise ValueError("cached tokens exceed total input tokens")
    input_tokens = totals["input_tokens"]
    return {
        "run_prefix": run_prefix,
        "runs": len(runs),
        **totals,
        "cache_hit_rate": (
            totals["cached_input_tokens"] / input_tokens if input_tokens else None
        ),
        "cache_write_tokens_reported": writes_reported,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = summarize(args.log_root, args.run_prefix)
    payload = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
