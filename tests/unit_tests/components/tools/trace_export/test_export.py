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

import json

import pytest

from components.tools.trace_export.export import export
from components.tools.trace_export.trace import TraceError, sha256_hex

METHODS = {"move_pose", "move_delta", "set_gripper", "parallel", "finish"}


def _rows(output):
    lines = (output / "rows.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


def _assert_nothing_exported(output, audit):
    assert audit["rows"] == 0
    assert (output / "rows.jsonl").read_text() == ""
    assert not (output / "images").exists()


def test_exports_projected_rows_with_audit(make_episode, tmp_path):
    source = make_episode()
    output = tmp_path / "out"
    audit = export([source], output)

    rows = _rows(output)
    assert [r["id"] for r in rows] == ["ep0:turn_000001", "ep0:turn_000002"]
    assert {r["environment_success"] for r in rows} == {True}
    first = rows[0]
    assert first["schema"] == "rpent.trace_export.v1"
    assert first["source"]["model"] == "model-x"
    code = first["target"]["tool_calls"][0]["function"]["arguments"]["code"]
    assert "explanation" not in code
    assert "explanation" in first["source"]["original_code"]
    image = first["images"][0]
    assert image["path"] == f"images/{image['png_sha256']}.png"
    assert (output / image["path"]).is_file()
    assert first["messages"][1]["content"][1] == {
        "type": "image",
        "path": image["path"],
    }
    assert audit == json.loads((output / "export-audit.json").read_text())
    assert audit["schema"] == "rpent.trace_export.v1"
    assert audit["accepted_episodes"] == 1
    assert audit["successful_rows"] == 2
    assert audit["unsuccessful_rows"] == 0
    assert set(audit) == {
        "schema",
        "trace_schema",
        "projection",
        "accepted_episodes",
        "rows",
        "successful_rows",
        "unsuccessful_rows",
        "quarantined_episodes",
        "rows_sha256",
        "split_policy",
    }
    assert audit["quarantined_episodes"] == []
    assert audit["rows_sha256"] == sha256_hex((output / "rows.jsonl").read_bytes())


def test_identical_images_are_stored_once(make_episode, tmp_path):
    source = make_episode(colors=[(5, 5, 5), (5, 5, 5)])
    output = tmp_path / "out"
    export([source], output)

    assert len(list((output / "images").glob("*.png"))) == 1
    assert len(_rows(output)) == 2


def test_discovers_episodes_under_parent_directory(make_episode, tmp_path):
    make_episode("batch_a/ep0", episode_id="ep0")
    make_episode("batch_b/nested/ep1", episode_id="ep1")
    output = tmp_path / "out"
    audit = export(tmp_path / "traces", output)

    assert audit["accepted_episodes"] == 2
    assert audit["rows"] == 4
    assert {r["episode_id"] for r in _rows(output)} == {"ep0", "ep1"}


def test_discovers_mixed_robots_using_each_episodes_robo_methods(
    make_episode, tmp_path
):
    make_episode(
        "libero_ep",
        robot="libero",
        robo_methods=["move_pose", "finish"],
        codes=('robo.move_pose(xyz=[0, 0, 0], explanation="Go.")\n', "robo.finish()\n"),
    )
    make_episode(
        "robotwin_ep",
        robot="robotwin",
        robo_methods=["move_arm", "finish"],
        codes=('robo.move_arm(arm="left", explanation="Lift.")\n', "robo.finish()\n"),
    )
    output = tmp_path / "out"
    audit = export(tmp_path / "traces", output)

    assert audit["accepted_episodes"] == 2
    assert audit["quarantined_episodes"] == []
    assert {r["source"]["robot"] for r in _rows(output)} == {"libero", "robotwin"}


def test_episode_without_robo_methods_requires_fallback_and_accepts_it(
    make_episode, tmp_path
):
    source = make_episode(robo_methods=None)
    missing_output = tmp_path / "missing"
    missing_audit = export([source], missing_output)

    _assert_nothing_exported(missing_output, missing_audit)
    reason = missing_audit["quarantined_episodes"][0]["reason"]
    assert "episode has no robo_methods" in reason

    fallback_output = tmp_path / "fallback"
    fallback_audit = export([source], fallback_output, methods=METHODS)
    assert fallback_audit["accepted_episodes"] == 1
    assert fallback_audit["rows"] == 2


def test_episode_robo_methods_takes_precedence_over_fallback(make_episode, tmp_path):
    source = make_episode(robo_methods=["finish"])
    output = tmp_path / "out"
    audit = export([source], output, methods=METHODS)

    _assert_nothing_exported(output, audit)
    assert "aliased/dynamic robo" in audit["quarantined_episodes"][0]["reason"]


@pytest.mark.parametrize(
    "bad_methods",
    [
        [],
        ["_private"],
        ["move_pose", "move_pose"],
        ["123bad"],
        ["class"],
        "move_pose",
    ],
)
def test_invalid_episode_robo_methods_quarantines_episode(
    make_episode, tmp_path, bad_methods
):
    source = make_episode(robo_methods=bad_methods)
    output = tmp_path / "out"
    audit = export([source], output)

    _assert_nothing_exported(output, audit)
    assert "episode robo_methods" in audit["quarantined_episodes"][0]["reason"]


def test_invalid_fallback_methods_fail_before_creating_output(make_episode, tmp_path):
    source = make_episode()
    output = tmp_path / "out"
    with pytest.raises(TraceError, match="fallback robo_methods"):
        export([source], output, methods=["_private"])
    assert not output.exists()


@pytest.mark.parametrize(
    ("accepted", "truth_source"),
    [(False, "environment"), (True, "operator"), (None, None)],
)
def test_verdict_without_environment_success_is_rejected(
    make_episode, tmp_path, accepted, truth_source
):
    source = make_episode(accepted=accepted, truth_source=truth_source)
    output = tmp_path / "out"
    audit = export([source], output)

    _assert_nothing_exported(output, audit)
    assert "opt-in required" in audit["quarantined_episodes"][0]["reason"]


def test_opt_in_exports_unsuccessful_episode(make_episode, tmp_path):
    source = make_episode(accepted=False)
    output = tmp_path / "out"
    audit = export([source], output, allow_unsuccessful=True)

    assert {r["environment_success"] for r in _rows(output)} == {False}
    assert audit["unsuccessful_rows"] == 2
    assert audit["successful_rows"] == 0


def _reset(turns):
    turns[1]["namespace_reset"] = True


def _extra_key(turns):
    turns[0]["hint"] = "x"


def _different_source(turns):
    turns[0]["executed_code"] = 'robo.finish(status="success")\n'


def _not_accepted(turns):
    turns[1]["accepted"] = False


def _gap(turns):
    turns[1]["turn"] = 5


def _escape(turns):
    turns[0]["images"][0]["path"] = "../escape.png"


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        (_reset, "isolate the episode"),
        (_extra_key, "keys must be exactly"),
        (_different_source, "executed source differs"),
        (_not_accepted, "was not accepted"),
        (_gap, "turn gap"),
        (_escape, "escaped"),
    ],
)
def test_bad_turn_quarantines_the_whole_episode(
    make_episode, edit_turns, tmp_path, change, reason
):
    source = make_episode()
    edit_turns(source, change)
    output = tmp_path / "out"
    audit = export([source], output)

    _assert_nothing_exported(output, audit)
    assert reason in audit["quarantined_episodes"][0]["reason"]


def test_corrupt_image_writes_no_images_for_the_episode(make_episode, tmp_path):
    source = make_episode()
    sorted((source / "images").glob("*.png"))[-1].write_bytes(b"not a png")
    output = tmp_path / "out"
    audit = export([source], output)

    _assert_nothing_exported(output, audit)
    assert "checksum mismatch" in audit["quarantined_episodes"][0]["reason"]


def test_missing_image_is_rejected(make_episode, tmp_path):
    source = make_episode()
    sorted((source / "images").glob("*.png"))[0].unlink()
    output = tmp_path / "out"
    audit = export([source], output)

    _assert_nothing_exported(output, audit)
    assert "missing image" in audit["quarantined_episodes"][0]["reason"]


def test_cell_that_needs_a_missing_earlier_cell_is_rejected(make_episode, tmp_path):
    source = make_episode(codes=("robo.move_pose(xyz=target)\n",))
    output = tmp_path / "out"
    audit = export([source], output)

    _assert_nothing_exported(output, audit)
    reason = audit["quarantined_episodes"][0]["reason"]
    assert "missing namespace definitions: target" in reason


def test_good_episode_survives_next_to_a_bad_one(make_episode, tmp_path):
    bad = make_episode("bad", accepted=False)
    good = make_episode("good")
    output = tmp_path / "out"
    audit = export([bad, good], output)

    assert audit["accepted_episodes"] == 1
    assert len(audit["quarantined_episodes"]) == 1
    assert {r["episode_id"] for r in _rows(output)} == {"good"}


def test_broken_nested_episode_missing_episode_json_is_quarantined(
    make_episode, tmp_path
):
    make_episode("good")
    broken = tmp_path / "traces" / "broken"
    broken.mkdir()
    (broken / "turns.jsonl").write_text("{}\n")
    output = tmp_path / "out"
    audit = export(tmp_path / "traces", output)

    assert audit["accepted_episodes"] == 1
    assert len(audit["quarantined_episodes"]) == 1
    assert audit["quarantined_episodes"][0]["episode"] == str(broken.resolve())


def test_duplicate_episode_is_not_counted_twice(make_episode, tmp_path):
    first = make_episode("a", episode_id="same")
    second = make_episode("b", episode_id="same")
    output = tmp_path / "out"
    audit = export([first, second], output)

    assert audit["accepted_episodes"] == 1
    assert len(_rows(output)) == 2
    assert "duplicate episode" in audit["quarantined_episodes"][0]["reason"]


def test_output_inside_a_source_is_rejected(make_episode):
    source = make_episode()
    with pytest.raises(TraceError, match="outside source traces"):
        export([source], source / "out")


def test_output_inside_parent_source_is_rejected(make_episode, tmp_path):
    make_episode()
    with pytest.raises(TraceError, match="outside source traces"):
        export(tmp_path / "traces", tmp_path / "traces" / "out")


def test_empty_parent_directory_is_rejected(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    output = tmp_path / "out"
    with pytest.raises(TraceError, match="no source episodes found"):
        export(empty, output)
    assert not output.exists()


def test_missing_source_directory_is_rejected(tmp_path):
    output = tmp_path / "out"
    with pytest.raises(TraceError, match="source directory does not exist"):
        export(tmp_path / "missing", output)
    assert not output.exists()


def test_existing_output_directory_is_rejected(make_episode, tmp_path):
    source = make_episode()
    output = tmp_path / "out"
    output.mkdir()
    with pytest.raises(FileExistsError):
        export([source], output)


def test_no_sources_is_rejected(tmp_path):
    with pytest.raises(TraceError, match="no source episodes"):
        export([], tmp_path / "out")
