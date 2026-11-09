#!/usr/bin/env python3
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

"""Publish a review-bot result to the pull request.

Review and design modes post a single comment from ``review.md``. Suggest mode
posts a review whose inline comments carry ``suggestion`` blocks; suggestions
whose location is not commentable in the diff are folded into the body.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

MAX_INLINE_COMMENTS = 20
HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def parse_diff(path: Path) -> dict[str, set[int]]:
    """Return the new-side line numbers present in each file's hunks."""
    allowed: dict[str, set[int]] = {}
    current: str | None = None
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("+++ b/"):
            current = line[6:]
            allowed.setdefault(current, set())
        elif line.startswith("+++ /dev/null"):
            current = None
        elif line.startswith("@@") and current is not None:
            match = HUNK.match(line)
            if match:
                start = int(match.group(1))
                count = int(match.group(2) or 1)
                allowed[current].update(range(start, start + count))
    return allowed


def load_suggestions(path: Path) -> dict:
    data = json.loads(path.read_text())
    for key in ("structured_output", "result"):
        value = data.get(key) if isinstance(data, dict) else None
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
    if isinstance(data, dict) and "suggestions" in data:
        return data
    raise SystemExit(f"{path}: no structured suggestions in the model output")


def post_comment(repo: str, pr: str, body: str) -> None:
    path = Path(os.environ.get("RUNNER_TEMP", "/tmp")) / "review-comment.md"
    path.write_text(body)
    subprocess.run(
        ["gh", "pr", "comment", pr, "--repo", repo, "--body-file", str(path)],
        check=True,
    )


def post_review(repo: str, pr: str, sha: str, payload: dict) -> None:
    payload_path = Path(os.environ.get("RUNNER_TEMP", "/tmp")) / "review-payload.json"
    payload_path.write_text(json.dumps(payload))
    subprocess.run(
        ["gh", "api", f"repos/{repo}/pulls/{pr}/reviews", "--input", str(payload_path)],
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", required=True, choices=["review", "design", "suggest"]
    )
    parser.add_argument("--repo", required=True)
    parser.add_argument("--pr", required=True)
    parser.add_argument("--sha", required=True)
    parser.add_argument("--dir", required=True, type=Path)
    args = parser.parse_args()

    if args.mode in ("review", "design"):
        body = (args.dir / "review.md").read_text().strip()
        if not body:
            raise SystemExit("review.md is empty")
        post_comment(args.repo, args.pr, body)
        return 0

    result = load_suggestions(args.dir / "suggest.json")
    summary = str(result.get("summary", "")).strip()
    allowed = parse_diff(args.dir / "diff.patch")

    inline: list[dict] = []
    folded: list[dict] = []
    for item in result.get("suggestions", []):
        path = str(item.get("path", "")).strip()
        line = item.get("line")
        start = item.get("start_line")
        reason = str(item.get("reason", "")).strip()
        replacement = str(item.get("replacement", ""))
        valid = (
            reason
            and replacement.strip()
            and "```" not in replacement
            and path in allowed
            and isinstance(line, int)
            and line in allowed[path]
            and (
                start is None
                or (isinstance(start, int) and start <= line and start in allowed[path])
            )
        )
        if valid and len(inline) < MAX_INLINE_COMMENTS:
            comment_body = f"{reason}\n\n```suggestion\n{replacement.rstrip()}\n```"
            comment = {
                "path": path,
                "line": line,
                "side": "RIGHT",
                "body": comment_body,
            }
            if start is not None and start < line:
                comment["start_line"] = start
                comment["start_side"] = "RIGHT"
            inline.append(comment)
        else:
            folded.append(item)

    parts = [summary] if summary else []
    if folded:
        parts += ["---", "### Suggestions outside commentable diff ranges"]
        for item in folded:
            parts.append(f"`{item.get('path', '?')}:{item.get('line', '?')}`")
            if explanation := str(item.get("reason", "")).strip():
                parts.append(explanation)
            replacement = str(item.get("replacement", "")).rstrip()
            if replacement:
                parts.append(
                    "\n".join(
                        f"    {code_line}" for code_line in replacement.splitlines()
                    )
                )
    body = "\n\n".join(parts)

    if not inline:
        if not body:
            raise SystemExit("no suggestions and no summary")
        post_comment(args.repo, args.pr, body)
        return 0

    post_review(
        args.repo,
        args.pr,
        args.sha,
        {"body": body, "event": "COMMENT", "commit_id": args.sha, "comments": inline},
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
