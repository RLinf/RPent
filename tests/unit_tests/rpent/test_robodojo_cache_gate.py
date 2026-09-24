# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS.

"""Check the minimum representative sample and strict cache threshold."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _gate_module():
    path = Path(__file__).parents[3] / "examples/robodojo/cache_gate.py"
    spec = importlib.util.spec_from_file_location("cache_gate", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cache_gate_requires_three_tasks_and_more_than_sixty_percent() -> None:
    gate = _gate_module()
    samples = [
        {
            "task": task,
            "input_tokens": 100,
            "cached_input_tokens": 60,
            "cache_write_tokens": 0,
        }
        for task in ("align_blocks", "classify_objects", "general_pickup")
        for _ in range(4)
    ]
    at_threshold = gate.decide_gate(samples, expected=12, threshold=0.60)
    assert at_threshold["representative"]
    assert not at_threshold["qualified"]

    samples[0]["cached_input_tokens"] = 61
    assert gate.decide_gate(samples, expected=12, threshold=0.60)["qualified"]
    assert not gate.decide_gate(samples[:11], expected=12, threshold=0.60)["qualified"]
