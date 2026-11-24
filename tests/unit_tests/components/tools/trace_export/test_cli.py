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

from components.tools.trace_export.cli import main


def _argv(sources, output, *extra):
    return [*(str(source) for source in sources), "-o", str(output), *extra]


def test_clean_export_returns_zero(make_episode, tmp_path):
    output = tmp_path / "out"

    assert main(_argv([make_episode()], output)) == 0
    audit = json.loads((output / "export-audit.json").read_text())
    assert audit["rows"] == 2
    assert audit["quarantined_episodes"] == []


def test_directory_discovery_exports_all_episodes(make_episode, tmp_path):
    make_episode("ep0")
    make_episode("ep1")
    output = tmp_path / "out"

    assert main([str(tmp_path / "traces"), "--output", str(output)]) == 0
    audit = json.loads((output / "export-audit.json").read_text())
    assert audit["accepted_episodes"] == 2
    assert audit["rows"] == 4


def test_fallback_robo_method_comma_separated_and_repeated(make_episode, tmp_path):
    source = make_episode("legacy", robo_methods=None)

    assert main(_argv([source], tmp_path / "missing")) == 1
    assert (
        main(
            _argv(
                [source],
                tmp_path / "csv",
                "--robo-method",
                "move_pose,set_gripper,finish",
            )
        )
        == 0
    )
    assert (
        main(
            _argv(
                [source],
                tmp_path / "repeated",
                "--robo-method",
                "move_pose",
                "--robo-method",
                "set_gripper",
                "--robo-method",
                "finish",
            )
        )
        == 0
    )


def test_invalid_fallback_robo_method_returns_one(make_episode, tmp_path):
    output = tmp_path / "out"

    assert main(_argv([make_episode()], output, "--robo-method", "_bad")) == 1
    assert not output.exists()


def test_quarantined_episode_returns_one_and_still_writes_the_audit(
    make_episode, tmp_path
):
    output = tmp_path / "out"
    sources = [make_episode("good"), make_episode("bad", accepted=False)]

    assert main(_argv(sources, output)) == 1
    audit = json.loads((output / "export-audit.json").read_text())
    assert audit["accepted_episodes"] == 1
    assert len(audit["quarantined_episodes"]) == 1


def test_include_unsuccessful_flag_exports_unsuccessful_episode(make_episode, tmp_path):
    output = tmp_path / "out"
    source = make_episode(accepted=False)

    assert main(_argv([source], output, "--include-unsuccessful")) == 0
    audit = json.loads((output / "export-audit.json").read_text())
    assert audit["unsuccessful_rows"] == 2


def test_existing_output_directory_returns_one(make_episode, tmp_path):
    output = tmp_path / "out"
    output.mkdir()

    assert main(_argv([make_episode()], output)) == 1


def test_output_inside_a_source_returns_one(make_episode):
    source = make_episode()

    assert main(_argv([source], source / "out")) == 1


def test_sources_are_required(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(["-o", str(tmp_path / "out")])
    assert exc.value.code == 2
