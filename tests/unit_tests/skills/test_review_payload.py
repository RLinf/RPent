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

from __future__ import annotations

import json
import runpy
from pathlib import Path
from unittest.mock import Mock

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / ".agents/skills/review-pr/lint_review.py"
review = runpy.run_path(str(SCRIPT))

BODY_WITH_HERE = """\
**Changes requested** · 0 critical, 1 major, 0 minor · head `abcdef1` · base `main@1234567`

Failed startup leaks an owned env daemon.

### Reproduction

- Result: **Not run** — offline check covers startup rollback directly.
- Setup: requires GPU and LIBERO assets for full simulator startup.

### Findings

| ID | Severity | Category | Location | Summary | Status | Posted |
|---|---|---|---|---|---|---|
| F1 | major | correctness | `rpent/robots/runtime.py:42` | Startup failure leaks env daemon | confirmed | here |

#### F1 · major · correctness: Failed startup leaks an owned daemon

`init_runtime` spawns the env server without registering it in `daemons`, so a later failure leaves the process running.

Suggested fix: pass `daemons` to `try_spawn_server` before waiting for readiness.

Evidence: `rpent/robots/runtime.py:42` returns without calling `stop_owned_daemons`.
"""

BODY_WITH_INLINE = """\
**Changes requested** · 0 critical, 1 major, 0 minor · head `abcdef1` · base `main@1234567`

Failed startup leaks an owned env daemon.

### Reproduction

- Result: **Not run** — offline check covers startup rollback directly.
- Setup: requires GPU and LIBERO assets for full simulator startup.

### Findings

| ID | Severity | Category | Location | Summary | Status | Posted |
|---|---|---|---|---|---|---|
| F1 | major | correctness | `rpent/x.py:2` | Startup failure leaks env daemon | confirmed | inline |
"""

INLINE_COMMENT = (
    "**F1 · major · correctness: Failed startup leaks an owned daemon**\n\n"
    "The error path returns before closing the spawned worker.\n\n"
    "Suggested fix: close the worker in `finally`.\n\n"
    "Evidence: `rpent/x.py:2` exits early without stopping the daemon."
)


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        ([], "payload: must be a JSON object"),
        ({"commit_id": None}, "commit_id"),
        ({"commit_id": "not-a-sha"}, "40-character hexadecimal commit SHA"),
        ({"comments": None}, "comments: must be a list"),
        ({"comments": ["text"]}, "comment 1: must be an object"),
        ({"body": ["paragraph"]}, "body: must be a string"),
        (
            {"comments": [{"path": "rpent/x.py", "line": 2, "body": [1]}]},
            "comment 1 (rpent/x.py:2): body must be a non-empty string",
        ),
        (
            {
                "body": BODY_WITH_HERE.replace(
                    "**Changes requested**", "Blocked workers need a timeout."
                )
            },
            "first line must start with one bold recommendation",
        ),
        (
            {"body": BODY_WITH_HERE.replace("### Reproduction", "### Setup")},
            "sections must be ['### Reproduction', '### Findings']",
        ),
        (
            {"body": BODY_WITH_HERE + "\n#### F1 · minor · docs: Duplicate\n\nText."},
            "F1 has a second section",
        ),
        (
            {"body": BODY_WITH_HERE.replace("- Setup:", "- Host:")},
            "Reproduction needs a '- Setup:' bullet",
        ),
        (
            {"body": BODY_WITH_HERE + "\n```python\nx = 1\n"},
            "opens a code fence that is never closed",
        ),
        (
            {"body": BODY_WITH_HERE + "\n<!-- leftover hint -->\n"},
            "HTML or an unfilled template placeholder",
        ),
    ],
)
def test_lint_rejects_invalid_payload(tmp_path, capsys, payload, error):
    path = tmp_path / "review.json"
    if isinstance(payload, dict):
        payload = {"body": BODY_WITH_HERE, **payload}
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert review["main"](["lint", str(path)]) == 1
    assert error in capsys.readouterr().out


def test_inline_table_row_pairs_with_inline_comment_and_rejects_mismatch():
    comment = {"path": "rpent/x.py", "line": 2, "side": "RIGHT", "body": INLINE_COMMENT}
    findings = review["ReviewLinter"](
        {"body": BODY_WITH_INLINE, "comments": [comment]}, None
    ).run()
    assert [f for f in findings if f.level == "ERROR"] == []

    # Inline comment without a matching table row fails.
    stray = {
        "path": "rpent/other.py",
        "line": 9,
        "side": "RIGHT",
        "body": INLINE_COMMENT,
    }
    findings = review["ReviewLinter"](
        {"body": BODY_WITH_INLINE, "comments": [comment, stray]}, None
    ).run()
    assert any("has no inline row" in f.message for f in findings if f.level == "ERROR")

    # Inline finding repeated as a #### section in the body fails.
    duplicated = (
        BODY_WITH_INLINE
        + "\n#### F1 · major · correctness: Duplicate\n\nBody.\n\nSuggested fix: x.\n\nEvidence: y.\n"
    )
    findings = review["ReviewLinter"](
        {"body": duplicated, "comments": [comment]}, None
    ).run()
    assert any("posted inline; do not repeat" in f.message for f in findings)


@pytest.mark.parametrize(
    ("fields", "errors"),
    [
        ({"start_line": 1, "side": "LEFT"}, ["multi-line comment missing start_side"]),
        ({"start_line": 1, "start_side": "RIGHT"}, ["multi-line comment missing side"]),
        ({"start_line": 1, "side": "LEFT", "start_side": "LEFT"}, []),
    ],
)
def test_multiline_comment_requires_both_sides(fields, errors):
    body = BODY_WITH_INLINE.replace("`rpent/x.py:2`", "`rpent/x.py:1-2`")
    comment = {"path": "rpent/x.py", "line": 2, "body": INLINE_COMMENT, **fields}
    findings = review["ReviewLinter"]({"body": body, "comments": [comment]}, None).run()
    assert [f.message for f in findings if f.level == "ERROR"] == errors


def test_template_can_be_filled_and_linted(tmp_path, capsys):
    assert review["main"](["template"]) == 0
    template = capsys.readouterr().out
    assert "### Reproduction" in template
    assert "### Findings" in template

    clean_no_findings = """\
**Ready for senior review** · no findings · head `abcdef1` · base `main@1234567`

Fixes a typo in the CLI help text.

### Reproduction

- Result: **Not run** — help-text typo fix requires no simulator run.
- Setup: offline CLI check only.

### Findings

No findings. Checked `rpent/cli/main.py` and `pytest tests/unit_tests/rpent/cli/ -v`.
"""
    path = tmp_path / "review.json"
    path.write_text(json.dumps({"body": clean_no_findings}), encoding="utf-8")
    assert review["main"](["lint", str(path)]) == 0


@pytest.mark.parametrize(
    ("fields", "status"),
    [
        ({}, 0),
        ({"commit_id": "A" * 40, "comments": []}, 0),
        ({"commit_id": "b" * 40}, 1),
    ],
)
def test_post_uses_reviewed_commit(tmp_path, monkeypatch, fields, status):
    path = tmp_path / "review.json"
    path.write_text(json.dumps({"body": BODY_WITH_HERE, **fields}), encoding="utf-8")
    head = "a" * 40
    post = Mock(
        return_value="https://github.com/RLinf/RPent/pull/364#pullrequestreview-1"
    )
    monkeypatch.setattr(review["PullRequest"], "head_sha", head)
    monkeypatch.setattr(review["PullRequest"], "post", post)

    assert review["main"](["post", "--pr", "364", str(path)]) == status
    if status:
        post.assert_not_called()
    else:
        post.assert_called_once_with(
            {"body": BODY_WITH_HERE, "event": "COMMENT", **fields, "commit_id": head}
        )


def test_update_validates_body_against_posted_inline_comments(tmp_path, monkeypatch):
    path = tmp_path / "review.json"
    path.write_text(json.dumps({"body": BODY_WITH_INLINE}), encoding="utf-8")
    posted_inline = [
        {
            "path": "rpent/x.py",
            "line": 2,
            "start_line": None,
            "side": "RIGHT",
            "start_side": "RIGHT",
            "body": INLINE_COMMENT,
        }
    ]
    update = Mock(
        return_value="https://github.com/RLinf/RPent/pull/364#pullrequestreview-9"
    )
    monkeypatch.setattr(
        review["PullRequest"], "review_comments", lambda self, rid: posted_inline
    )
    monkeypatch.setattr(review["PullRequest"], "update", update)

    assert review["main"](["update", "--pr", "364", "--review", "9", str(path)]) == 0
    update.assert_called_once_with(9, BODY_WITH_INLINE)


def test_thread_posted_row_supports_carried_forward_replies(monkeypatch):
    body_with_thread = BODY_WITH_INLINE.replace("| inline |", "| thread |")

    # Offline lint accepts a carried-forward thread row with no new comments[] entry.
    findings = review["ReviewLinter"](
        {"body": body_with_thread, "comments": []}, None
    ).run()
    assert [f for f in findings if f.level == "ERROR"] == []

    # Repeating a thread finding as a #### section in the main body fails.
    duplicated = (
        body_with_thread
        + "\n#### F1 · major · correctness: Duplicate\n\nBody.\n\nSuggested fix: x.\n\nEvidence: y.\n"
    )
    findings = review["ReviewLinter"]({"body": duplicated, "comments": []}, None).run()
    assert any("replied on an existing thread" in f.message for f in findings)

    # With --pr, active and outdated threads both pass, shifted lines warn, missing files error.
    pr = review["PullRequest"]("RLinf/RPent", 372)
    monkeypatch.setattr(
        pr,
        "api_items",
        lambda endpoint, jq: [
            {
                "path": "rpent/x.py",
                "side": "RIGHT",
                "line": 2,
                "original_line": 2,
                "user": "reviewer",
                "in_reply_to_id": None,
            },
            {
                "path": "rpent/outdated.py",
                "side": "RIGHT",
                "line": None,
                "original_line": 14,
                "user": "reviewer",
                "in_reply_to_id": None,
            },
            {
                "path": "rpent/outdated.py",
                "side": "RIGHT",
                "line": None,
                "original_line": 14,
                "user": "author",
                "in_reply_to_id": 100,
            },
        ],
    )
    assert pr.commented_lines == {("rpent/x.py", "RIGHT", 2): "reviewer"}
    assert pr.thread_lines == {
        ("rpent/x.py", "RIGHT", 2): "reviewer",
        ("rpent/outdated.py", "RIGHT", 14): "reviewer",
    }

    findings = review["ReviewLinter"](
        {"body": body_with_thread, "comments": []}, pr
    ).run()
    assert findings == []

    outdated_body = body_with_thread.replace("`rpent/x.py:2`", "`rpent/outdated.py:14`")
    findings = review["ReviewLinter"]({"body": outdated_body, "comments": []}, pr).run()
    assert findings == []

    outdated_moved_body = body_with_thread.replace(
        "`rpent/x.py:2`", "`rpent/outdated.py:20`"
    )
    findings = review["ReviewLinter"](
        {"body": outdated_moved_body, "comments": []}, pr
    ).run()
    assert [f for f in findings if f.level == "ERROR"] == []
    assert any(
        f.level == "WARN" and "no existing thread sits on line 20" in f.message
        for f in findings
    )

    shifted_body = body_with_thread.replace("`rpent/x.py:2`", "`rpent/x.py:10`")
    findings = review["ReviewLinter"]({"body": shifted_body, "comments": []}, pr).run()
    assert [f for f in findings if f.level == "ERROR"] == []
    assert any(
        f.level == "WARN" and "no existing thread sits on line 10" in f.message
        for f in findings
    )

    missing_body = body_with_thread.replace("`rpent/x.py:2`", "`rpent/other.py:2`")
    findings = review["ReviewLinter"]({"body": missing_body, "comments": []}, pr).run()
    assert any(
        f.level == "ERROR"
        and "no existing inline thread sits on rpent/other.py" in f.message
        for f in findings
    )
