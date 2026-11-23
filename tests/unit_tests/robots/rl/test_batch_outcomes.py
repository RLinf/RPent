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

"""Task failures count as evaluations without becoming valid training samples."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "rl_batch_outcomes", Path(__file__).parents[4] / "tools/rl/run_batch.py"
)
batch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(batch)


@pytest.mark.parametrize(
    "kind", ["response_truncated", "context_limit", "tool_arguments", "episode_timeout"]
)
def test_normal_terminal_errors_are_not_retried_as_infrastructure(tmp_path, kind):
    (tmp_path / "transcript_api.json").write_text(json.dumps({"error_kind": kind}))
    (tmp_path / "states.json").write_text(
        json.dumps({"steps": [{"terminated": False}]})
    )
    result = batch.terminal_failure(tmp_path)
    assert result["success"] is False and result["training_usable"] is False
    assert result["termination_reason"] == kind


def test_infrastructure_failure_is_not_a_task_outcome(tmp_path):
    (tmp_path / "transcript_api.json").write_text(
        json.dumps({"error_kind": "model_error"})
    )
    assert batch.terminal_failure(tmp_path) is None


@pytest.mark.parametrize("success", [False, True])
def test_recovered_invalid_reply_counts_once_with_native_success(tmp_path, success):
    (tmp_path / "result.json").write_text(
        json.dumps({"success": success, "invalid_tool_responses": 1})
    )
    (tmp_path / "transcript_api.json").write_text(json.dumps({"error": None}))
    (tmp_path / "states.json").write_text(
        json.dumps({"steps": [{"terminated": success}]})
    )
    result = batch.terminal_failure(tmp_path)
    assert result["termination_reason"] == "invalid_tool_reply"
    assert result["success"] is success and not result["training_usable"]


def test_rejected_reply_does_not_hide_later_infrastructure_error(tmp_path):
    (tmp_path / "result.json").write_text(
        json.dumps({"invalid_tool_responses": 1, "agent_error": "connection lost"})
    )
    (tmp_path / "transcript_api.json").write_text(
        json.dumps({"error_kind": "model_error", "error": "connection lost"})
    )
    assert batch.terminal_failure(tmp_path) is None


def test_native_failure_artifact_takes_precedence_over_planner_completion(tmp_path):
    (tmp_path / "infrastructure_error.json").write_text('{"error":"CUDA failure"}')
    (tmp_path / "result.json").write_text(
        json.dumps({"success": False, "invalid_tool_responses": 1})
    )
    (tmp_path / "transcript_api.json").write_text(json.dumps({"error": None}))
    (tmp_path / "states.json").write_text(
        json.dumps({"steps": [{"terminated": False}]})
    )
    assert batch.terminal_failure(tmp_path) is None
    assert batch.completed_without_training_trace(tmp_path, "missing trace") is None


def test_deadline_without_final_result_is_retained_but_not_training_data(tmp_path):
    (tmp_path / "states.json").write_text(
        json.dumps({"steps": [{"terminated": False}]})
    )
    native = tmp_path / "api_trace/native"
    native.mkdir(parents=True)
    (native / "000001.json").write_text("{}")
    result = batch.terminal_failure(tmp_path, deadline=True, job={"seed": 1})
    assert result["task"] == {"seed": 1} and not result["training_usable"]


def test_simulator_startup_timeout_is_infrastructure_failure(tmp_path):
    assert batch.terminal_failure(tmp_path, deadline=True, job={"seed": 1}) is None


def test_resume_rejects_a_modified_evaluation(tmp_path):
    attempt = tmp_path / "attempt-001"
    attempt.mkdir()
    transcript = attempt / "transcript_api.json"
    transcript.write_text("original response")
    (attempt / "EVALUATED.json").write_text(
        json.dumps(
            {
                "outcome": {"success": False, "training_usable": False},
                "hashes": {
                    transcript.name: hashlib.sha256(transcript.read_bytes()).hexdigest()
                },
            }
        )
    )
    assert batch.evaluated(tmp_path)["success"] is False
    transcript.write_text("damaged response")
    assert batch.evaluated(tmp_path) is None


@pytest.mark.parametrize("no_thinking", [False, True])
@pytest.mark.parametrize("missing_trace", [False, True])
def test_batch_thinking_reaches_process_and_cannot_change_on_resume(
    tmp_path, monkeypatch, no_thinking, missing_trace
):
    """Check the actual child argv and manifest boundary with an offline worker."""
    import sys

    scripts = tmp_path / "tools" / "rl"
    scripts.mkdir(parents=True)
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "p=Path(os.environ['RPENT_OUTPUT_DIR'])\n"
        "assert os.environ['RPENT_RECORD_MODEL_REQUESTS']=='1'\n"
        "(p/'argv.json').write_text(json.dumps(sys.argv[1:]))\n"
        "(p/'states.json').write_text(json.dumps({'steps':[{'terminated':False}]}))\n"
        + (
            "(p/'result.json').write_text(json.dumps({'task':{'seed':0},'success':False,'stats':{'turns_used':1},'max_turns':50}))\n"
            "(p/'transcript_api.json').write_text(json.dumps({'error':None}))\n"
            "(p/'rollout.json').write_text('{}')\n"
            if missing_trace
            else "(p/'transcript_api.json').write_text(json.dumps({'error_kind':'context_limit'}))\n"
        )
    )
    launcher = batch.EVAL_DIR / "run.sh"
    (scripts / "run.sh").write_text(
        "#!/usr/bin/env bash\n"
        f'if [[ "$1" == --print-defaults ]]; then exec bash "{launcher}" --print-defaults; fi\n'
        'exec "$RPENT_TEST_PYTHON" "$RPENT_TEST_WORKER" "$@"\n'
    )
    (scripts / "planner.sh").write_text("# offline fixture\n")
    monkeypatch.setattr(batch, "EVAL_DIR", scripts)
    monkeypatch.setenv("RPENT_TEST_PYTHON", sys.executable)
    monkeypatch.setenv("RPENT_TEST_WORKER", str(worker))
    monkeypatch.setenv("RPENT_MODEL", "openai-chat:qwen3.8-27b")
    jobs = tmp_path / "jobs.json"
    jobs.write_text(json.dumps([{"suite": "libero_object", "task": 0, "seed": 0}]))
    output = tmp_path / "batch"
    output.mkdir()
    args = argparse.Namespace(
        jobs=jobs,
        seeds=None,
        workflow="episode",
        prompt="simple",
        image_size=1024,
        no_thinking=no_thinking,
        gpus="0",
        per_gpu=1,
        max_initializing_per_gpu=1,
        endpoints="http://127.0.0.1:1/v1",
        infrastructure_retries=1,
        episode_timeout=30,
        start_interval=0,
        start_barrier=False,
    )
    handlers = {
        sig: batch.signal.getsignal(sig)
        for sig in (batch.signal.SIGTERM, batch.signal.SIGINT)
    }
    try:
        batch.collect(args, output)
        attempt = next(output.glob("*/attempt-001"))
        assert not list(output.glob("*/attempt-002"))
        if missing_trace:
            outcome = batch.evaluated(attempt.parent)
            assert outcome["termination_reason"] == "planner_completed"
            assert outcome["training_usable"] is False
            assert "completed turn coverage" in outcome["audit_error"]
        argv = json.loads((attempt / "argv.json").read_text())
        assert ("--no-thinking" in argv) is no_thinking
        assert (
            json.loads((output / "manifest.json").read_text())["thinking"]
            is not no_thinking
        )
        args.no_thinking = not no_thinking
        with pytest.raises(ValueError, match="Resume configuration differs"):
            batch.collect(args, output)
    finally:
        for sig, handler in handlers.items():
            batch.signal.signal(sig, handler)
