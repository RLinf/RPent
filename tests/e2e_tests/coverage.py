# Copyright 2026 The RPent Authors.
"""Compare planner tool calls with the schemas RPent advertised."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _images(value: Any) -> set[str]:
    if isinstance(value, dict):
        if str(value.get("media_type", "")).startswith("image/") and "data" in value:
            return {json.dumps(value["data"], sort_keys=True)}
        value = value.values()
    elif not isinstance(value, list):
        return set()
    return {image for item in value for image in _images(item)}


def inspect_trajectory(
    output_dir: Path,
    action: str | None = None,
    skip_tools: set[str] | list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Write a per-tool report even when a CLI run stopped early."""
    skipped = set(skip_tools or ())
    inventory = output_dir / "tool-inventory.json"
    expected = (
        set(json.loads(inventory.read_text())) - skipped
        if inventory.is_file()
        else set()
    )
    executed = []
    failures = []
    calls = []
    previous_images = set()
    new_images = 0
    after_action = False
    files = list(output_dir.glob("trajectories/*/planners.jsonl"))
    if len(files) != 1:
        failures.append(f"expected one trajectory, found {len(files)}")
    else:
        with files[0].open() as records:
            for turn, line in enumerate(records, 1):
                record = json.loads(line)
                request = (record["request"] or {}).get("data", {})
                params = request.get("model_request_parameters", {})
                for group in ("function_tools", "output_tools"):
                    expected.update(t["name"] for t in params.get(group, []))
                images = _images(request.get("messages", []))
                if after_action:
                    new_images += len(images - previous_images)
                previous_images = images
                tools = record["tool_calls"]
                if record["status"] != "completed" or len(tools) != 1:
                    failures.append(
                        f"turn {turn}: {record['status']}, {len(tools)} tools"
                    )
                after_action = (
                    any(t["name"] == action for t in tools) if action else False
                )
                for tool in tools:
                    name = tool["name"]
                    calls.append(
                        {
                            "turn": turn,
                            "name": name,
                            "status": tool["status"],
                            "arguments": tool["arguments"]["data"],
                        }
                    )
                    if tool["origin"] == "model" and tool["status"] == "completed":
                        executed.append(name)
                    else:
                        reason = (tool.get("raw_result") or {}).get("data", {})
                        if isinstance(reason, dict):
                            reason = reason.get("data", reason)
                        error = (
                            reason.get("error") if isinstance(reason, dict) else reason
                        )
                        if not error:
                            error = (tool.get("missing") or {}).get("raw_result")
                        if not error:
                            feedback = (tool.get("feedback") or {}).get("data", {})
                            error = (
                                feedback.get("return_value")
                                if isinstance(feedback, dict)
                                else feedback
                            )
                        calls[-1]["error"] = error
                        failures.append(
                            f"turn {turn}: {name} {tool['status']}: {error}"
                        )
    if not executed or executed[-1] != "finish" or (action and action not in executed):
        expected_desc = f"{action} and a final finish" if action else "a final finish"
        failures.append(f"expected {expected_desc}")
    missing = sorted((expected - skipped) - set(executed))
    report = {
        "expected": sorted(expected),
        "attempted": sorted({call["name"] for call in calls}),
        "executed": executed,
        "missing": missing,
        "failures": failures,
        "calls": calls,
        "post_action_new_image_count": new_images,
        "passed": bool(expected) and not (failures or missing),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tool-coverage.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
