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

"""Lint and post an RPent PR review written as a GitHub review payload.

The payload is the JSON body of GitHub's create-review API
(``POST /repos/{owner}/{repo}/pulls/{number}/reviews``).

.. code-block:: json

    {
        "commit_id": "<PR head SHA>",
        "event": "COMMENT",
        "body": "<the main comment>",
        "comments": [
            {"path": "rpent/x.py", "line": 42, "side": "RIGHT", "body": "..."},
            {"path": "robots/libero/y.py", "start_line": 10, "start_side": "RIGHT",
             "line": 12, "side": "RIGHT", "body": "..."}
        ]
    }

Subcommands::

    template                         print the review body template
    lint [--pr N] review.json        check the payload (and its lines, with --pr)
    post --pr N review.json          lint, then post it as one review

Exit code is 1 when any ERROR is reported, 0 otherwise (warnings pass).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from functools import cached_property

RECOMMENDATIONS = ("Ready for senior review", "Changes requested", "Blocked")
TEMPLATE = f"""\
<!-- On the first line write one recommendation in bold: {", ".join(RECOMMENDATIONS)}. -->

<!-- Summarize the change and explain the recommendation. -->

### Reproduction

<!-- Identify the runner, revision, settings, commands, and results, or explain why the workflow was not run. -->

### Findings

<!-- Write "No findings." or give each issue a heading such as "#### F1 · major · correctness · Title" followed by its location, trigger, impact, and evidence. -->
"""
FINDING_HEADING = re.compile(
    r"#### (F[1-9][0-9]*) · (?:critical|major|minor) · "
    r"(?:correctness|design|reuse|tests|docs|docs-style|style) · \S.*"
)
EVENTS = ("COMMENT", "REQUEST_CHANGES", "APPROVE")
FIX_LEAD = "Suggested fix:"
PAYLOAD_KEYS = {"commit_id", "event", "body", "comments"}
COMMENT_KEYS = {"path", "line", "side", "start_line", "start_side", "body"}
MAX_COMMENT_PROSE = 1000


@dataclass(frozen=True)
class Finding:
    level: str  # "ERROR" | "WARN"
    where: str
    message: str


class Markdown:
    """Prose checks for text GitHub renders as a comment.

    GitHub turns every newline in a comment into a line break, so a paragraph
    must stay on one line.
    """

    FENCE = re.compile(r"^\s*(`{3,}|~{3,})(.*)$")
    BLOCK = re.compile(r"^\s*([#|>]|[-*+] |\d+[.)] |<)")
    HEADING = re.compile(r"^(#{1,6})\s")
    HINT = "<!--"

    @classmethod
    def unfenced(cls, text: str) -> list[tuple[int, str]]:
        """Returns (line number, line) for every line outside code fences.

        As in CommonMark, only a bare fence of the opening character, at least
        as long as the opening one, closes a block, so fences can nest.
        """
        out, fence = [], ""
        for n, line in enumerate(text.splitlines(), 1):
            m = cls.FENCE.match(line)
            if not fence:
                if m:
                    fence = m[1]
                else:
                    out.append((n, line))
            elif (
                m
                and m[1][0] == fence[0]
                and len(m[1]) >= len(fence)
                and not m[2].strip()
            ):
                fence = ""
        return out

    @classmethod
    def wrapped(cls, text: str) -> list[int]:
        """Returns the lines that continue the paragraph above them."""
        hits, prev_n, prev_prose = [], -1, False
        for n, line in cls.unfenced(text):
            stripped = line.strip()
            if (
                prev_prose
                and n == prev_n + 1
                and stripped
                and not cls.BLOCK.match(line)
            ):
                hits.append(n)
            prev_n = n
            prev_prose = bool(stripped) and not stripped.startswith(("#", "|", "<"))
        return hits

    @classmethod
    def prose_length(cls, text: str) -> int:
        return sum(len(line.strip()) for _, line in cls.unfenced(text))


class PullRequest:
    """The PR state a review is checked against, read through the `gh` CLI."""

    HUNK = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")

    def __init__(self, repo: str, number: int) -> None:
        self.repo = repo
        self.number = number

    @staticmethod
    def gh(*args: str, stdin: str | None = None) -> str:
        proc = subprocess.run(
            ["gh", *args], input=stdin, capture_output=True, text=True, check=False
        )
        if proc.returncode != 0:
            sys.exit(f"error: gh {' '.join(args[:2])} failed: {proc.stderr.strip()}")
        return proc.stdout

    def api_items(self, endpoint: str, jq: str) -> list[dict]:
        out = self.gh("api", endpoint, "--paginate", "--jq", jq)
        return [json.loads(line) for line in out.splitlines() if line.strip()]

    @cached_property
    def head_sha(self) -> str:
        out = self.gh(
            "pr", "view", str(self.number), "--repo", self.repo, "--json", "headRefOid"
        )
        return json.loads(out)["headRefOid"]

    @cached_property
    def diff_lines(self) -> dict[tuple[str, str], dict[int, int]]:
        """Maps (path, side) to {line number: hunk index} for commentable lines."""
        files = self.api_items(
            f"repos/{self.repo}/pulls/{self.number}/files",
            ".[] | {filename, patch}",
        )
        lines: dict[tuple[str, str], dict[int, int]] = {}
        for f in files:
            right = lines.setdefault((f["filename"], "RIGHT"), {})
            left = lines.setdefault((f["filename"], "LEFT"), {})
            self.parse_patch(f.get("patch") or "", left, right)
        return lines

    @classmethod
    def parse_patch(cls, patch: str, left: dict, right: dict) -> None:
        hunk, old, new = -1, 0, 0
        for line in patch.splitlines():
            m = cls.HUNK.match(line)
            if m:
                hunk, old, new = hunk + 1, int(m[1]), int(m[2])
            elif line.startswith("+"):
                right[new], new = hunk, new + 1
            elif line.startswith("-"):
                left[old], old = hunk, old + 1
            elif not line.startswith("\\"):
                left[old], right[new] = hunk, hunk
                old, new = old + 1, new + 1

    @cached_property
    def commented_lines(self) -> dict[tuple[str, str, int], str]:
        """Maps (path, side, line) of existing review threads to their author."""
        items = self.api_items(
            f"repos/{self.repo}/pulls/{self.number}/comments",
            ".[] | {path, side, line, user: .user.login, in_reply_to_id}",
        )
        return {
            (c["path"], c["side"], c["line"]): c["user"]
            for c in items
            if c["line"] is not None and c["in_reply_to_id"] is None
        }

    def post(self, payload: dict) -> str:
        endpoint = f"repos/{self.repo}/pulls/{self.number}/reviews"
        out = self.gh(
            "api", "-X", "POST", endpoint, "--input", "-", stdin=json.dumps(payload)
        )
        return json.loads(out)["html_url"]


class ReviewLinter:
    """Checks a review payload against the review-pr skill's output format."""

    def __init__(self, payload: dict, pr: PullRequest | None) -> None:
        self.payload = payload
        self.pr = pr
        self.findings: list[Finding] = []

    def error(self, where: str, message: str) -> None:
        self.findings.append(Finding("ERROR", where, message))

    def warn(self, where: str, message: str) -> None:
        self.findings.append(Finding("WARN", where, message))

    def run(self) -> list[Finding]:
        if not isinstance(self.payload, dict):
            self.error("payload", "must be a JSON object")
            return self.findings
        comments = self.payload.get("comments", [])
        if not isinstance(comments, list):
            self.error("comments", "must be a list (omit it or use [] for no comments)")
            return self.findings

        extra = set(self.payload) - PAYLOAD_KEYS
        if extra:
            self.error(
                "payload",
                f"unknown keys {sorted(extra)}; expected {sorted(PAYLOAD_KEYS)}",
            )
        if self.payload.get("event", "COMMENT") not in EVENTS:
            self.error(
                "payload",
                f"event must be one of {EVENTS}, found {self.payload['event']!r}",
            )
        if "commit_id" in self.payload:
            commit_id = self.payload["commit_id"]
            if not isinstance(commit_id, str) or not re.fullmatch(
                r"[0-9a-fA-F]{40}", commit_id
            ):
                self.error("commit_id", "must be a 40-character hexadecimal commit SHA")
            elif self.pr and commit_id.lower() != self.pr.head_sha:
                self.error(
                    "commit_id",
                    f"{commit_id[:7]} is not the PR head {self.pr.head_sha[:7]}; "
                    "re-check the findings on the new head",
                )
        self.check_body(self.payload.get("body"))
        seen: set[tuple[str, str, int]] = set()
        for i, comment in enumerate(comments, 1):
            self.check_comment(i, comment, seen)
        return self.findings

    def check_prose(self, where: str, text: str) -> None:
        for n in Markdown.wrapped(text):
            self.error(
                where,
                f"line {n} is hard-wrapped; GitHub renders it as a line break, so keep each paragraph on one line",
            )
        for n, line in Markdown.unfenced(text):
            if Markdown.HINT in line:
                self.error(
                    where,
                    f"line {n} still holds a template hint: {line.strip()[:60]!r}",
                )

    def check_body(self, body: object) -> None:
        if not isinstance(body, str):
            self.error("body", "must be a string")
            return
        self.check_prose("body", body)
        body = body.strip()
        first_line = body.partition("\n")[0].rstrip()
        if not any(first_line in (label, f"**{label}**") for label in RECOMMENDATIONS):
            self.error(
                "body",
                f"start with one recommendation on its own line: {RECOMMENDATIONS}",
            )

        lines = body.splitlines()
        sections = []
        for n, line in Markdown.unfenced(body):
            heading = Markdown.HEADING.match(line)
            if heading and len(heading[1]) <= 3:
                sections.append((n, line.rstrip()))
        if [title for _, title in sections] != ["### Reproduction", "### Findings"]:
            self.error("body", "expected ### Reproduction followed by ### Findings")
            return

        reproduction, findings = (n for n, _ in sections)
        if not any(line.strip() for line in lines[1 : reproduction - 1]):
            self.error("body", "explain the recommendation before ### Reproduction")
        if not any(line.strip() for line in lines[reproduction : findings - 1]):
            self.error(
                "body", "describe the reproduction or explain why it was not run"
            )
        self.check_findings("\n".join(lines[findings:]).strip())

    def check_findings(self, text: str) -> None:
        if text == "No findings.":
            if self.payload.get("comments"):
                self.error(
                    "body", 'a review with inline comments cannot say "No findings."'
                )
            return

        headings = [
            (n - 1, line.rstrip())
            for n, line in Markdown.unfenced(text)
            if line.startswith("#### ")
        ]
        if not headings or headings[0][0] != 0:
            self.error(
                "body",
                "start each finding with a numbered heading or write No findings.",
            )
            return

        lines = text.splitlines()
        seen = set()
        for i, (start, title) in enumerate(headings):
            match = FINDING_HEADING.fullmatch(title)
            if not match:
                self.error(
                    "body",
                    f"expected #### F<n> · severity · category · title, found {title!r}",
                )
                continue
            finding_id = match[1]
            if finding_id in seen:
                self.error("body", f"duplicate finding ID {finding_id}")
            seen.add(finding_id)
            end = headings[i + 1][0] if i + 1 < len(headings) else len(lines)
            if not any(line.strip() for line in lines[start + 1 : end]):
                self.error("body", f"{finding_id} needs a description")

    def check_comment(self, i: int, c: object, seen: set) -> None:
        if not isinstance(c, dict):
            self.error(f"comment {i}", "must be an object")
            return
        path, line, body = c.get("path"), c.get("line"), c.get("body")
        side = c.get("side", "RIGHT")
        where = f"comment {i} ({path}:{line})"
        extra = set(c) - COMMENT_KEYS
        if extra:
            self.error(
                where, f"unknown keys {sorted(extra)}; expected {sorted(COMMENT_KEYS)}"
            )
        if not isinstance(body, str) or not body.strip():
            self.error(where, "body must be a non-empty string")
            return
        if not isinstance(path, str) or not isinstance(line, int):
            self.error(where, "needs a string path and an integer line")
            return
        start = c.get("start_line")
        if start is not None and (missing := {"side", "start_side"} - c.keys()):
            self.error(
                where, f"multi-line comment missing {', '.join(sorted(missing))}"
            )
            return
        if side not in ("RIGHT", "LEFT") or c.get("start_side", side) != side:
            self.error(where, "side and start_side must match and be RIGHT or LEFT")
        if start is not None and (not isinstance(start, int) or start >= line):
            self.error(
                where,
                f"start_line must be an integer below line {line}, found {start!r}",
            )
        if FIX_LEAD not in body:
            self.error(where, f"every inline comment needs a '{FIX_LEAD}' paragraph")
        if "```suggestion" in body and side == "LEFT":
            self.error(
                where, "a suggestion block only applies to RIGHT-side (new) lines"
            )
        self.check_prose(where, body)
        if Markdown.prose_length(body) > MAX_COMMENT_PROSE:
            self.warn(
                where,
                f"prose is over {MAX_COMMENT_PROSE} characters; put the walkthrough in the review body",
            )
        key = (path, side, line)
        if key in seen:
            self.warn(
                where,
                "another comment in this review sits on the same line; merge them",
            )
        seen.add(key)
        if self.pr:
            self.check_location(where, path, side, start, line)

    def check_location(
        self, where: str, path: str, side: str, start: int | None, line: int
    ) -> None:
        lines = self.pr.diff_lines.get((path, side))
        if lines is None:
            self.error(
                where, "the PR does not change this file; move the finding to the body"
            )
            return
        for n in (start, line):
            if n is not None and n not in lines:
                self.error(
                    where,
                    f"{side} line {n} is outside the diff hunks; GitHub rejects the whole review",
                )
        if start in lines and line in lines and lines[start] != lines[line]:
            self.error(where, f"lines {start}-{line} span two diff hunks")
        author = self.pr.commented_lines.get((path, side, line))
        if author:
            self.warn(
                where,
                f"@{author} already has a thread on this line; reply there instead",
            )


def report(findings: list[Finding], payload: dict, checked_lines: bool) -> int:
    errors = [f for f in findings if f.level == "ERROR"]
    warns = [f for f in findings if f.level == "WARN"]
    if not errors:
        n = len(payload.get("comments", []))
        print(
            f"review: {payload.get('event', 'COMMENT')}, body + {n} inline comment(s)"
        )
    print(f"{len(errors)} error(s), {len(warns)} warning(s)")
    for f in errors + warns:
        print(f"[{f.level[0]}] {f.where}: {f.message}")
    if not checked_lines:
        print("note: comment lines and existing threads were not checked; pass --pr")
    return 1 if errors else 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--repo", default="RLinf/RPent", help="owner/name of the PR's repo")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("template", help="print the review body template")
    for name, help_text in (
        ("lint", "check a review payload"),
        ("post", "lint, then post"),
    ):
        cmd = sub.add_parser(name, help=help_text)
        cmd.add_argument("--pr", type=int, required=name == "post", help="PR number")
        cmd.add_argument("payload", help="review payload JSON file")
    args = p.parse_args(argv)

    if args.cmd == "template":
        print(TEMPLATE, end="")
        return 0

    with open(args.payload, encoding="utf-8") as fh:
        payload = json.load(fh)
    pr = PullRequest(args.repo, args.pr) if args.pr else None
    status = report(ReviewLinter(payload, pr).run(), payload, pr is not None)
    if args.cmd == "lint" or status:
        return status
    payload["commit_id"] = pr.head_sha
    payload.setdefault("event", "COMMENT")
    print(f"posted: {pr.post(payload)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
