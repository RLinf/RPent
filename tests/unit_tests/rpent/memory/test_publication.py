# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

"""Memory publication leaves durable conflict and partial-failure evidence."""

import json
from types import SimpleNamespace

import pytest

from rpent.memory.publication import merge_run_memory


@pytest.mark.parametrize("fails", [False, True])
def test_merge_receipt_preserves_outcome_and_original_exception(tmp_path, fails):
    expected = {"conflicts": 2, "skipped": ["draft lacking evidence"]}
    error = RuntimeError("failed after a partial publication")

    def merge_memory(**kwargs):
        assert (
            json.loads((tmp_path / "memory-merge.json").read_text())["status"]
            == "started"
        )
        (tmp_path / "published.md").write_text("already published")
        if fails:
            raise error
        return expected

    memory = SimpleNamespace(merge_memory=merge_memory)
    if fails:
        with pytest.raises(RuntimeError) as raised:
            merge_run_memory(
                memory, cell_tag="task", run_state_dir=tmp_path, solved=False
            )
        assert raised.value is error
    else:
        assert (
            merge_run_memory(
                memory, cell_tag="task", run_state_dir=tmp_path, solved=True
            )
            == expected
        )
    receipt = json.loads((tmp_path / "memory-merge.json").read_text())
    assert receipt["status"] == ("failed" if fails else "completed")
    if fails:
        assert receipt["error_type"] == "RuntimeError"
        assert receipt["partial_changes_possible"] is True
    else:
        assert receipt["result"] == expected
    assert (tmp_path / "published.md").read_text() == "already published"
    assert not list(tmp_path.glob(".memory-merge.json.*.tmp"))
