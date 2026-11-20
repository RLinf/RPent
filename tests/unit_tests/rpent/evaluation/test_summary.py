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
"""The batch denominator comes from the manifest, including absent episodes."""

import json

import pytest

from rpent.evaluation import TrialRecorder
from rpent.evaluation.summary import summarize


def trial(seed, task="cup", scene="living_room"):
    return {
        "trial_id": f"{task}/{scene}/{seed}",
        "method": "policy",
        "task": task,
        "seed": seed,
        "scene": scene,
    }


def episode(root, spec, success, *, infra=False):
    now = [1.0]
    rec = TrialRecorder(root, **spec, clock=lambda: now[0])
    rec.start()
    now[0] = 3.0
    rec.record_native_verdict(success, source="env")
    rec.stop("complete")
    now[0] = 4.0
    rec.quiesce()
    rec.record_usage(
        request_id="1",
        model="m1",
        module="planner",
        stage="plan",
        input_tokens=100,
        output_tokens=20,
    )
    if not success:
        rec.record_usage(request_id="2", model="m2", module="camera", stage="check")
    return json.loads(rec.finish(infra_invalid=infra).read_text())


def test_manifest_and_infrastructure_retry(tmp_path):
    a, b, c = (trial(i) for i in range(3))
    records = [
        episode(tmp_path, a, False, infra=True),
        episode(tmp_path, a, True),
        episode(tmp_path, b, False),
    ]
    output = summarize([a, b, c], records)
    row = output["cells"][0]
    assert (
        row["planned"],
        row["valid"],
        row["unresolved"],
        row["infra_invalid_attempts"],
    ) == (3, 2, 1, 1)
    assert row["native_sr"] == 0.5
    assert row["run_s"] == {"measured": 2, "missing": 0, "median": 3.0, "p90": 3.0}
    assert row["success_s"]["median"] == 2.0
    assert row["token_usage"]["episode_counts"] == {
        "complete": 1,
        "partial": 1,
        "unknown": 0,
        "not_applicable": 0,
    }
    assert output["methods"][0]["native_sr_macro"] is None
    with pytest.raises(ValueError, match="Multiple non-infrastructure"):
        summarize([a, b, c], [*records, episode(tmp_path, a, False)])
    with pytest.raises(ValueError, match="Duplicate run_id"):
        summarize([a, b, c], [*records, records[0]])
    with pytest.raises(ValueError, match="Mismatched seed"):
        summarize([{**a, "seed": 999}], records[:1])


def test_nested_macro_and_unknown_time(tmp_path):
    trials = [trial(0), trial(1), trial(2, scene="lab"), trial(3, task="ball")]
    records = [episode(tmp_path, spec, i in (0, 1)) for i, spec in enumerate(trials)]
    records[1]["timing"]["run_s"] = None
    result = summarize(trials, records)
    assert result["methods"] == [
        {"method": "policy", "complete": True, "native_sr_macro": 0.25}
    ]
    assert result["cells"][0]["run_s"]["missing"] == 1


def test_human_intervention_and_autonomous_run_have_separate_counts(tmp_path):
    spec = trial(9)
    human = episode(tmp_path, spec, True)
    human.update(status="unresolved", manual_intervention=True)
    official = episode(tmp_path, spec, False)
    row = summarize([spec], [human, official])["cells"][0]
    assert (row["assisted_attempts"], row["valid"], row["successes"]) == (1, 1, 0)


def test_reviewed_environment_start_allows_only_same_condition_retry(tmp_path):
    slot = trial(7)
    original = TrialRecorder(tmp_path, **slot)
    original.stop("environment_initialization_failed")
    path = original.finish(error_type="ValueError")
    immutable_bytes = path.read_bytes()
    unresolved = json.loads(immutable_bytes)
    # Only an episode that reached reset can bind the instruction.
    spec = {**slot, "instruction_version": "v2"}
    official = episode(tmp_path, spec, True)
    assert summarize([spec], [unresolved])["cells"][0]["unresolved"] == 1
    with pytest.raises(ValueError, match="Multiple non-infrastructure"):
        summarize([spec], [unresolved, official])

    reviewed = {
        **spec,
        "infrastructure_reviews": {
            original.run_id: "Cluster incident #17 verified by the operator."
        },
    }
    out = summarize([reviewed], [unresolved, official])
    row = out["cells"][0]
    assert (
        row["planned"],
        row["valid"],
        row["successes"],
        row["infra_invalid_attempts"],
    ) == (1, 1, 1, 1)
    assert out["infrastructure_reviews"] == [
        {
            "trial_id": slot["trial_id"],
            "method": slot["method"],
            "run_id": original.run_id,
            "original_status": "unresolved",
            "effective_status": "infra_invalid",
            "reason": reviewed["infrastructure_reviews"][original.run_id],
        }
    ]
    assert path.read_bytes() == immutable_bytes and unresolved["status"] == "unresolved"
    with pytest.raises(ValueError, match="Mismatched seed"):
        summarize([reviewed], [unresolved, {**official, "seed": 8}])
    with pytest.raises(ValueError, match="Mismatched budget for replacement"):
        summarize([reviewed], [unresolved, {**official, "budget": {"wall_s": 42}}])
    for wrong in (
        official,
        {**unresolved, "status": "failure"},
        {**unresolved, "manual_intervention": True},
        {**unresolved, "stop_reason": "runtime_initialization_failed"},
    ):
        with pytest.raises(ValueError, match="requires an unresolved environment"):
            summarize(
                [{**slot, "infrastructure_reviews": {wrong["run_id"]: "reason"}}],
                [wrong],
            )
    for invalid in ({"other-run": "reason"}, {original.run_id: "  "}, []):
        with pytest.raises(ValueError):
            summarize([{**spec, "infrastructure_reviews": invalid}], [unresolved])
    # A real run ID from a different slot still cannot be reviewed here.
    other = trial(8)
    foreign = TrialRecorder(tmp_path, **other)
    foreign.stop("environment_initialization_failed")
    foreign_record = json.loads(foreign.finish().read_text())
    with pytest.raises(ValueError, match="Unknown infrastructure review run_id"):
        summarize(
            [
                {**spec, "infrastructure_reviews": {foreign.run_id: "different case"}},
                other,
            ],
            [unresolved, foreign_record],
        )
