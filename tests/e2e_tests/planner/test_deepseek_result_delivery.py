# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

"""Real rc2 output materialization, using MCP and an offline model only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rpent.tools import ToolResult, tool

from .test_deepseek_runtime import RecordingToolkit, call
from .test_deepseek_runtime import run_runtime as run_runtime


def events(result):
    return [
        row["payload"]["event"]
        for line in Path(result.stats["raw_stream_path"]).read_text().splitlines()
        if (row := json.loads(line)).get("method") == "session.event"
    ]


def test_negative_zero_coordinates_are_delivered_normally(run_runtime):
    class ZeroToolkit(RecordingToolkit):
        @tool
        def act(self) -> ToolResult:
            """Return rounded native coordinates, including signed zero."""
            return ToolResult(
                data={"final_eef_pos": [0.0705, -0.0, 1.0194], "pitch": -0.0}
            )

    result, toolkit, requests, control = run_runtime(
        [{"calls": [call("act")]}, {"calls": [call("finish")]}],
        toolkit=ZeroToolkit(),
    )
    assert result.error is None
    assert toolkit.calls == ["act", "finish"]
    assert len(requests) == 2
    assert not control.get("toolDeliveryErrors")
    outcomes = [event for event in events(result) if event["type"] == "tool/result"]
    assert all(not event["data"]["message"].get("isError") for event in outcomes)
    assert result.stop_info["result_delivery_complete"] is True


@pytest.mark.parametrize("number", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_result_cannot_admit_another_action(run_runtime, number):
    class NonfiniteToolkit(RecordingToolkit):
        @tool
        def act(self) -> ToolResult:
            """Complete an action whose observation cannot be serialized."""
            return ToolResult(data={"position": [number]})

    result, toolkit, requests, control = run_runtime(
        [{"calls": [call("act"), call("act")]}, {"calls": [call("act")]}],
        toolkit=NonfiniteToolkit(),
    )
    assert toolkit.calls == ["act"]
    assert len(requests) == 1
    assert result.error and result.stats["stop_reason"] == "tool_transport_error"
    assert control["stopReason"] == "tool_transport_error"
    assert result.stop_info["allow_new_session"] is False


@pytest.mark.parametrize("accepted_finish", [False, True])
def test_final_runtime_output_failure_stops_same_batch(
    run_runtime, monkeypatch, accepted_finish
):
    import rpent.planner.utils.http_mcp_server as mcp_server

    canonicalize = mcp_server.mcp_result
    first = "finish" if accepted_finish else "act"

    def inject_after_canonicalization(result, **kwargs):
        wire = canonicalize(result, **kwargs)
        if result.data.get("_finish" if accepted_finish else "acted"):
            # Valid JSON -0 survives HTTP/MCP, but rc2 rejects it only after
            # the plugin's execute() returns, in snapshotToolValue().
            wire["structuredContent"]["fault_injection_negative_zero"] = -0.0
        return wire

    monkeypatch.setattr(mcp_server, "mcp_result", inject_after_canonicalization)
    result, toolkit, requests, control = run_runtime(
        [{"calls": [call(first), call("act")]}, {"calls": [call("act")]}]
    )
    assert toolkit.calls == [first]
    assert len(requests) == 1
    assert result.error and result.stats["stop_reason"] == "tool_transport_error"
    assert control["toolDeliveryErrors"][0]["code"] == "INVALID_TOOL_OUTPUT"
    assert control["deniedCalls"][0]["name"] == "mcp__rpent__act"
    assert result.stop_info["result_delivery_complete"] is False
    assert result.stop_info["allow_new_session"] is False
    if accepted_finish:
        assert result.finish_result["_finish"] is True
        assert control["finishResult"]["_finish"] is True


def test_explicit_business_refusal_remains_recoverable(run_runtime):
    result, toolkit, requests, control = run_runtime(
        [
            {"calls": [call("finish")]},
            {"calls": [call("act")]},
            {"calls": [call("finish")]},
        ],
        toolkit=RecordingToolkit(refuse_finish=True),
    )
    assert result.error is None
    assert toolkit.calls == ["finish", "act", "finish"]
    assert len(requests) == 3
    assert not control.get("toolDeliveryErrors")


@pytest.mark.parametrize("number", [float("nan"), float("inf"), -float("inf")])
def test_business_error_with_nonfinite_payload_is_not_a_delivery_ack(
    run_runtime, number
):
    class BrokenRefusalToolkit(RecordingToolkit):
        @tool
        def act(self) -> ToolResult:
            """Refuse an action with an observation that cannot cross MCP."""
            return ToolResult(data={"error": "business refusal", "position": [number]})

    result, toolkit, requests, control = run_runtime(
        [{"calls": [call("act"), call("act")]}, {"calls": [call("finish")]}],
        toolkit=BrokenRefusalToolkit(),
    )
    assert toolkit.calls == ["act"]
    assert len(requests) == 1
    assert result.error and control["stopReason"] == "tool_transport_error"
    assert result.stop_info["result_delivery_complete"] is False
    assert result.stop_info["allow_new_session"] is False


@pytest.mark.parametrize("fault", ["changed", "lost"])
def test_business_error_requires_exact_canonical_wire_text(
    run_runtime, monkeypatch, fault
):
    import rpent.planner.deepseek_harness as planner_module
    import rpent.planner.utils.http_mcp_server as mcp_server

    canonicalize = mcp_server.mcp_result
    seen = 0

    def break_wire_after_provenance(result, **kwargs):
        nonlocal seen
        wire = canonicalize(result, **kwargs)
        if result.is_error:
            seen += 1
            # First conversion establishes the ledger's canonical digest;
            # second conversion is the actual HTTP/MCP response delivery.
            if seen == 2:
                if fault == "lost":
                    raise OSError("Lost the canonical business-error response")
                wire["content"][0]["text"] = "Unrelated transport error"
        return wire

    monkeypatch.setattr(mcp_server, "mcp_result", break_wire_after_provenance)
    monkeypatch.setattr(planner_module, "mcp_result", break_wire_after_provenance)
    result, toolkit, requests, control = run_runtime(
        [{"calls": [call("finish"), call("act")]}, {"calls": [call("act")]}],
        toolkit=RecordingToolkit(refuse_finish=True),
    )
    assert seen == 2
    assert toolkit.calls == ["finish"]
    assert len(requests) == 1
    assert result.error and control["stopReason"] == "tool_transport_error"
    assert result.stop_info["result_delivery_complete"] is False
    assert result.stop_info["allow_new_session"] is False


def test_business_refusal_cannot_silently_drop_observation_images(run_runtime):
    import io

    from PIL import Image

    png = io.BytesIO()
    Image.new("RGB", (8, 6), "blue").save(png, format="PNG")

    class ImageRefusalToolkit(RecordingToolkit):
        @tool
        def act(self) -> ToolResult:
            """Refuse and attach a fresh observation that must be delivered."""
            return ToolResult(data={"error": "business refusal"}, images=self.images)

    result, toolkit, requests, control = run_runtime(
        [{"calls": [call("act"), call("act")]}, {"calls": [call("finish")]}],
        toolkit=ImageRefusalToolkit(images=[png.getvalue()]),
    )
    assert toolkit.calls == ["act"]
    assert len(requests) == 1
    assert result.error and control["stopReason"] == "tool_transport_error"
    assert result.stop_info["result_delivery_complete"] is False
    assert result.stop_info["allow_new_session"] is False


def test_negative_zero_business_refusal_is_recoverable(run_runtime):
    class ZeroRefusalToolkit(RecordingToolkit):
        @tool(readonly=True)
        def finish(self) -> ToolResult:
            """Return a canonical business refusal, then accept completion."""
            if self.refuse_finish:
                self.refuse_finish = False
                return ToolResult(data={"error": "not ready", "position": [-0.0]})
            return ToolResult(data={"_finish": True, "status": "success"})

    result, toolkit, requests, control = run_runtime(
        [
            {"calls": [call("finish")]},
            {"calls": [call("act")]},
            {"calls": [call("finish")]},
        ],
        toolkit=ZeroRefusalToolkit(refuse_finish=True),
    )
    assert result.error is None
    assert toolkit.calls == ["finish", "act", "finish"]
    assert len(requests) == 3
    assert not control.get("toolDeliveryErrors")


def test_truncated_business_error_is_not_complete_delivery(run_runtime):
    class LargeRefusalToolkit(RecordingToolkit):
        @tool
        def act(self) -> ToolResult:
            """Refuse with a payload larger than the model text limit."""
            return ToolResult(
                data={"error": "business refusal", "observation": "x" * 70000}
            )

    result, toolkit, requests, control = run_runtime(
        [{"calls": [call("act"), call("act")]}, {"calls": [call("finish")]}],
        toolkit=LargeRefusalToolkit(),
    )
    assert toolkit.calls == ["act"]
    assert len(requests) == 1
    assert result.error and control["stopReason"] == "tool_transport_error"
    assert result.stop_info["result_delivery_complete"] is False
    assert result.stop_info["allow_new_session"] is False
