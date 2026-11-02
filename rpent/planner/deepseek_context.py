# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

"""Consumed operator context for fresh dsh exploration sessions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def consumed_operator_inputs(
    candidates: dict[str, dict],
    observed_receipts: dict[str, int | None],
    control_receipts: dict[str, dict],
) -> list[dict]:
    """Join original host input with positive runtime consumption evidence."""
    confirmed = {}
    for message_id, candidate in candidates.items():
        receipt = candidate.get("receipt")
        if receipt in observed_receipts:
            confirmed[message_id] = (receipt, observed_receipts[receipt])
    for receipt, record in control_receipts.items():
        message_id = record.get("messageId")
        if record.get("status") == "consumed" and message_id in candidates:
            confirmed[message_id] = (receipt, record.get("eventSeq"))
    ordered = []
    for index, (message_id, candidate) in enumerate(candidates.items()):
        if message_id not in confirmed:
            continue
        receipt, sequence = confirmed[message_id]
        ordered.append(
            (
                sequence if type(sequence) is int else float("inf"),
                index,
                {
                    "message_id": message_id,
                    "receipt_id": receipt,
                    "text": candidate["text"],
                },
            )
        )
    return [record for _, _, record in sorted(ordered, key=lambda row: row[:2])]


class DeepSeekHandoffContext:
    """Keep operator text once per message ID within one exploration TaskRun."""

    def __init__(self, output_dir: Path):
        self.path = Path(output_dir) / "dsh-operator-context.json"
        self._records: list[dict] = []
        self._seen: set[str] = set()

    def merge(self, stats: dict[str, Any], session_number: int) -> None:
        """Persist confirmed inputs before another independent session starts."""
        changed = False
        for record in stats.get("consumed_operator_inputs", []):
            message_id, text = record.get("message_id"), record.get("text")
            if (
                not isinstance(message_id, str)
                or not isinstance(text, str)
                or not text.strip()
                or message_id in self._seen
            ):
                continue
            self._seen.add(message_id)
            self._records.append(
                {
                    "message_id": message_id,
                    "receipt_id": record.get("receipt_id"),
                    "session": session_number,
                    "text": text,
                }
            )
            changed = True
        if changed:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(
                    {"schema_version": 1, "consumed_operator_inputs": self._records},
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n"
            )
            temporary.replace(self.path)

    def append_to(self, message: str) -> str:
        """Quote prior steering without resubmitting it as a new action request."""
        if not self._records:
            return message
        history = [
            {"session": item["session"], "text": item["text"]} for item in self._records
        ]
        return (
            message
            + "\n\nPreviously consumed operator instructions (historical context):\n"
            "Continue to honor still-applicable goals and constraints; later explicit "
            "operator corrections take precedence over earlier ones. These are not "
            "new commands to replay. Do not repeat completed one-off actions merely "
            "because they appear here. Determine action state from fresh observations "
            "and execution records; ask the operator if applicability is unclear.\n"
            + json.dumps(history, ensure_ascii=False, indent=2)
        )
