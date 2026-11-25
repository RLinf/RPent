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

"""Lint an RPent pull-request title and body against CONTRIBUTING.md.

Checks the Conventional Commits title format and the
`.github/PULL_REQUEST_TEMPLATE.md` section structure (`Description`, `Testing`,
`Manual verification`, `Checklist`), and catches common Markdown rendering
defects: bodies pasted with a uniform leading indent, tables whose rows and
header disagree on indentation, missing or empty template sections, and
checklists with nothing ticked.

Subcommands::

    lint      --pr N | --title T [--body-file F]   check a PR or a draft
    fix       --pr N | --body-file F               emit a de-indented body
    template                                       print a clean skeleton

Exit code is 1 when any ERROR is reported, 0 otherwise (warnings pass).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import textwrap
from collections.abc import Iterable
from dataclasses import dataclass, field

ALLOWED_TYPES = (
    "feat",
    "fix",
    "docs",
    "style",
    "refactor",
    "test",
    "chore",
    "perf",
    "build",
    "ci",
    "revert",
)

TITLE_RE = re.compile(
    r"^(?P<type>[a-zA-Z]+)(?:\((?P<scope>[^)]*)\))?(?P<bang>!?): (?P<desc>.+)$"
)

VAGUE_SCOPES = {
    "new",
    "update",
    "updates",
    "change",
    "changes",
    "misc",
    "various",
    "code",
    "main",
    "all",
    "general",
    "tmp",
    "temp",
    "wip",
    "pr",
}

# Section headings of RPent's .github/PULL_REQUEST_TEMPLATE.md, in order.
TEMPLATE_SECTIONS = (
    ("Description", True),
    ("Testing", False),
    ("Manual verification", False),
    ("Checklist", True),
)

TITLE_DESC_MAX = 72
SCOPE_RE = re.compile(r"^[a-z0-9_-]+$")
NON_ASCII_RE = re.compile(r"[^\x20-\x7e]")
DESCRIPTION_MAX_WORDS = 250

COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
HEADING_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?P<hashes>#{1,6})\s+(?P<text>.*?)\s*:?\s*$"
)
FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})(.*)$")
CHECKED_RE = re.compile(r"^\s*[-*]\s*\[[xX]\]")
UNCHECKED_RE = re.compile(r"^\s*[-*]\s*\[\s*\]")
LIST_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
ISSUE_SUFFIX_RE = re.compile(r"\(#\d+\)\s*$")

CI_NON_IMPERATIVE = {
    "added",
    "adds",
    "adding",
    "updated",
    "updates",
    "updating",
    "fixed",
    "fixes",
    "fixing",
    "removed",
    "removes",
    "removing",
    "deleted",
    "deletes",
    "deleting",
}

PAST_TENSE_STARTS = (
    "added",
    "fixed",
    "updated",
    "removed",
    "changed",
    "refactored",
    "implemented",
    "supported",
    "improved",
)

PAST_TENSE_STARTS_IMPERATIVE = {
    "added": "add",
    "fixed": "fix",
    "updated": "update",
    "removed": "remove",
    "changed": "change",
    "refactored": "refactor",
    "implemented": "implement",
    "supported": "support",
    "improved": "improve",
}

DROPS_E = {"updat", "remov", "delet", "creat", "mov", "renam", "merg"}


@dataclass
class Finding:
    level: str  # "ERROR" | "WARN"
    where: str  # "title" | "body"
    code: str
    message: str
    fix: str = ""


@dataclass
class Section:
    name: str
    start: int
    lines: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        stripped = COMMENT_RE.sub("", "\n".join(self.lines)).strip()
        return not stripped

    @property
    def text(self) -> str:
        return COMMENT_RE.sub("", "\n".join(self.lines)).strip()


def fence_mask_with_state(lines: Iterable[str]) -> tuple[list[bool], bool]:
    """Return (per-line in-code mask, whether a fence remained unclosed)."""
    mask: list[bool] = []
    fence = ""
    for line in lines:
        m = FENCE_RE.match(line)
        if not fence:
            if m:
                fence = m.group(1)
                mask.append(True)
            else:
                mask.append(False)
        else:
            mask.append(True)
            if (
                m
                and m.group(1)[0] == fence[0]
                and len(m.group(1)) >= len(fence)
                and not m.group(2).strip()
            ):
                fence = ""
    return mask, bool(fence)


def fence_mask(lines: Iterable[str]) -> list[bool]:
    """Return, per line, whether it sits inside a fenced code block."""
    return fence_mask_with_state(lines)[0]


def strip_comments_outside_fences(text: str) -> str:
    """Strip HTML comments outside fenced code blocks while preserving code lines."""
    lines = text.splitlines(keepends=True)
    in_code = fence_mask(ln.rstrip("\r\n") for ln in lines)
    chunks: list[str] = []
    prose_buf: list[str] = []
    for line, code in zip(lines, in_code):
        if code:
            if prose_buf:
                chunks.append(COMMENT_RE.sub("", "".join(prose_buf)))
                prose_buf = []
            chunks.append(line)
        else:
            prose_buf.append(line)
    if prose_buf:
        chunks.append(COMMENT_RE.sub("", "".join(prose_buf)))
    return "".join(chunks)


def split_sections(body: str) -> list[Section]:
    """Split a PR body into top-level heading-delimited sections (code fences respected)."""
    lines = body.splitlines()
    in_code = fence_mask(lines)
    sections: list[Section] = []
    for i, line in enumerate(lines):
        if in_code[i]:
            if sections:
                sections[-1].lines.append(line)
            continue
        m = HEADING_RE.match(line)
        if m and len(m.group("hashes")) <= 2:
            sections.append(Section(name=m.group("text").strip(), start=i))
        elif sections:
            sections[-1].lines.append(line)
    return sections


def find_section(sections: list[Section], name: str) -> Section | None:
    key = name.lower().rstrip(":?").strip()
    for sec in sections:
        if sec.name.lower().rstrip(":?").strip().startswith(key):
            return sec
    return None


def fetch_pr(number: str, repo: str = "RLinf/RPent") -> tuple[str, str]:
    """Return (title, body) for a PR via the `gh` CLI."""
    out = subprocess.run(
        ["gh", "pr", "view", str(number), "--repo", repo, "--json", "title,body"],
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode != 0:
        sys.exit(f"error: gh pr view {number} failed:\n{out.stderr.strip()}")
    data = json.loads(out.stdout)
    return data["title"], (data.get("body") or "").replace("\r\n", "\n")


def imperative_of(word: str) -> str:
    """Best-effort imperative form of a past-tense/third-person verb."""
    stem = word.lower()
    if stem in PAST_TENSE_STARTS_IMPERATIVE:
        return PAST_TENSE_STARTS_IMPERATIVE[stem]
    for suffix in ("ing", "ed", "es", "s"):
        if stem.endswith(suffix):
            base = stem[: -len(suffix)]
            return base + "e" if suffix in ("ing", "ed") and base in DROPS_E else base
    return stem


def check_title(title: str) -> list[Finding]:
    out: list[Finding] = []
    if not title.strip():
        return [Finding("ERROR", "title", "title-empty", "PR title is empty.")]

    bad = NON_ASCII_RE.findall(title)
    if bad:
        out.append(
            Finding(
                "ERROR",
                "title",
                "title-non-ascii",
                f"Title contains non-ASCII characters: {''.join(sorted(set(bad)))!r}",
                "Use plain ASCII (32-126) — no smart quotes, em dashes, "
                "subscripts, or CJK.",
            )
        )

    m = TITLE_RE.match(title)
    if not m:
        out.append(
            Finding(
                "ERROR",
                "title",
                "title-format",
                f"Title is not Conventional Commits: {title!r}",
                "Rewrite as `<type>(<scope>): <description>`, e.g. "
                "`fix(toolkit): retain the original handler error`.",
            )
        )
        return out

    typ, scope, desc = m.group("type"), m.group("scope"), m.group("desc")

    if typ.lower() not in ALLOWED_TYPES:
        out.append(
            Finding(
                "ERROR",
                "title",
                "title-type",
                f"Unknown type {typ!r}.",
                "Use one of: " + ", ".join(ALLOWED_TYPES) + ".",
            )
        )
    elif typ != typ.lower():
        out.append(
            Finding(
                "ERROR", "title", "title-type-case", f"Type {typ!r} must be lowercase."
            )
        )

    if scope is not None:
        if not scope.strip():
            out.append(
                Finding(
                    "ERROR",
                    "title",
                    "title-scope-empty",
                    "Empty scope `()`.",
                    "Drop the parentheses or name the touched component.",
                )
            )
        elif not SCOPE_RE.match(scope):
            out.append(
                Finding(
                    "ERROR",
                    "title",
                    "title-scope-format",
                    f"Scope {scope!r} is not lowercase [a-z0-9_-].",
                    "Scopes take lowercase letters, digits, hyphens and "
                    "underscores only.",
                )
            )
        elif scope in VAGUE_SCOPES:
            out.append(
                Finding(
                    "WARN",
                    "title",
                    "title-scope-vague",
                    f"Scope {scope!r} says nothing about what changed.",
                    "Name the component the diff touches (e.g. `planner`, "
                    "`toolkit`, `libero`, `efe`, `docs`) or drop the scope.",
                )
            )

    if len(desc) > TITLE_DESC_MAX:
        out.append(
            Finding(
                "ERROR",
                "title",
                "title-desc-long",
                f"Description is {len(desc)} chars; max is {TITLE_DESC_MAX} "
                "(the part after `: `, not the whole title).",
                "Say the single most important thing and move the rest into "
                "the Description section.",
            )
        )

    if desc != desc.rstrip():
        out.append(
            Finding(
                "ERROR", "title", "title-trailing-space", "Drop the trailing space."
            )
        )
    if re.search(r":\s{2,}", title):
        out.append(
            Finding(
                "ERROR",
                "title",
                "title-colon-spacing",
                "Use exactly one space after the colon.",
            )
        )
    if "!" in desc:
        out.append(
            Finding(
                "ERROR",
                "title",
                "title-bang-position",
                "`!` may only appear right before the colon as the breaking-"
                "change marker — never inside the description.",
            )
        )

    desc = desc.strip()
    if desc.endswith("."):
        out.append(
            Finding("ERROR", "title", "title-period", "Drop the trailing period.")
        )

    first = desc.split()[0] if desc.split() else ""
    if not re.match(r"^[a-z]", desc):
        out.append(
            Finding(
                "ERROR",
                "title",
                "title-capital",
                f"Description must start with a lowercase a-z, not {first!r}.",
            )
        )
    if first.lower() in CI_NON_IMPERATIVE:
        out.append(
            Finding(
                "ERROR",
                "title",
                "title-mood",
                f"Use the imperative mood, not {first!r} "
                f"(e.g. {imperative_of(first)}).",
            )
        )
    elif first.lower() in PAST_TENSE_STARTS:
        out.append(
            Finding(
                "WARN",
                "title",
                "title-mood",
                f"Use the imperative mood, not {first!r} "
                f"(e.g. {imperative_of(first)}).",
            )
        )
    if desc.lower().startswith(("this pr", "pr ", "a pr")):
        out.append(
            Finding(
                "WARN", "title", "title-throat-clearing", "Drop the 'this PR' preamble."
            )
        )
    if ISSUE_SUFFIX_RE.search(desc):
        out.append(
            Finding(
                "WARN",
                "title",
                "title-pr-number",
                "Drop the trailing `(#N)`; GitHub appends it on squash-merge.",
            )
        )
    return out


def line_indent(line: str) -> int:
    return len(line) - len(line.lstrip(" \t"))


def continues_list(lines: list[str], i: int, in_code: list[bool]) -> bool:
    """True when line `i` is an indented continuation of a markdown list."""
    for j in range(i - 1, -1, -1):
        if in_code[j]:
            return False
        if not lines[j].strip():
            continue
        if LIST_RE.match(lines[j]):
            return line_indent(lines[i]) > line_indent(lines[j])
        return False
    return False


def check_tables(lines: list[str], in_code: list[bool]) -> list[Finding]:
    """Flag markdown tables whose rows disagree on leading indentation."""
    out: list[Finding] = []
    block: list[int] = []

    def flush(rows: list[int]) -> None:
        if len(rows) < 2:
            return
        indents = {line_indent(lines[i]) for i in rows}
        if len(indents) > 1:
            out.append(
                Finding(
                    "ERROR",
                    "body",
                    "body-table-indent",
                    f"Table at line {rows[0] + 1} mixes row indentation "
                    f"{sorted(indents)} — GitHub will not render it as a table.",
                    "Give every row of the table the same (preferably zero) "
                    "indentation.",
                )
            )
        elif indents and min(indents) >= 4:
            out.append(
                Finding(
                    "ERROR",
                    "body",
                    "body-table-indent",
                    f"Table at line {rows[0] + 1} is indented "
                    f"{min(indents)} spaces — it renders as a code block.",
                    "Un-indent the table.",
                )
            )

    for i, line in enumerate(lines):
        if not in_code[i] and line.strip().count("|") >= 2:
            block.append(i)
        else:
            flush(block)
            block = []
    flush(block)
    return out


def check_indentation(body: str) -> list[Finding]:
    """Catch bodies pasted with a uniform leading indent."""
    out: list[Finding] = []
    lines = body.splitlines()
    in_code = fence_mask(lines)

    indented_headings = [
        i
        for i, line in enumerate(lines)
        if not in_code[i] and (m := HEADING_RE.match(line)) and m.group("indent")
    ]
    if indented_headings:
        out.append(
            Finding(
                "ERROR",
                "body",
                "body-indented-heading",
                f"{len(indented_headings)} template heading(s) have leading "
                f"whitespace (first at line {indented_headings[0] + 1}).",
                "Strip the indent — run `lint_pr.py fix` and repost the body.",
            )
        )

    prose = [
        i
        for i, line in enumerate(lines)
        if not in_code[i] and line.strip() and not HEADING_RE.match(line)
    ]
    if prose:
        indented = [
            i
            for i in prose
            if line_indent(lines[i]) and not continues_list(lines, i, in_code)
        ]
        if len(indented) >= max(4, 0.3 * len(prose)):
            out.append(
                Finding(
                    "ERROR",
                    "body",
                    "body-indented",
                    f"{len(indented)}/{len(prose)} prose lines start with "
                    "whitespace — the body was pasted with a uniform indent.",
                    "Run `lint_pr.py fix` to de-indent, then repost.",
                )
            )

    out.extend(check_tables(lines, in_code))
    return out


def check_body(body: str, *, docs_only: bool = False) -> list[Finding]:
    out: list[Finding] = []
    if not body.strip():
        return [
            Finding(
                "ERROR",
                "body",
                "body-empty",
                "PR body is empty.",
                "Start from `lint_pr.py template`.",
            )
        ]

    out.extend(check_indentation(body))

    lines = body.splitlines()
    in_code, unclosed_fence = fence_mask_with_state(lines)
    if unclosed_fence:
        out.append(
            Finding(
                "ERROR",
                "body",
                "body-fence",
                "Unbalanced code fence — later text is swallowed.",
            )
        )

    sections = split_sections(body)

    for name, must_fill in TEMPLATE_SECTIONS:
        sec = find_section(sections, name)
        if sec is None:
            out.append(
                Finding(
                    "ERROR" if must_fill else "WARN",
                    "body",
                    "section-missing",
                    f"Template section `## {name}` is missing.",
                    "Keep every heading from .github/PULL_REQUEST_TEMPLATE.md, "
                    "even when a section is short.",
                )
            )
            continue
        if must_fill and sec.is_empty:
            out.append(
                Finding(
                    "ERROR",
                    "body",
                    "section-empty",
                    f"Required section `## {name}` is empty.",
                    "Fill in `## Description` and `## Checklist` before submitting.",
                )
            )

    positions = [
        sec.start
        for name, _ in TEMPLATE_SECTIONS
        if (sec := find_section(sections, name)) is not None
    ]
    if len(positions) > 1 and positions != sorted(positions):
        out.append(
            Finding(
                "WARN",
                "body",
                "section-order",
                "Template sections are out of template order.",
            )
        )

    if sections:
        head = COMMENT_RE.sub("", "\n".join(lines[: sections[0].start])).strip()
        if head:
            out.append(
                Finding(
                    "WARN",
                    "body",
                    "body-preamble",
                    f"{len(head.split())} word(s) sit above the first `##` heading.",
                    "Move that text into `## Description`.",
                )
            )

    checklist = find_section(sections, "Checklist")
    if checklist is not None:
        checked = [ln for ln in checklist.lines if CHECKED_RE.match(ln)]
        boxes = checked + [ln for ln in checklist.lines if UNCHECKED_RE.match(ln)]
        if not boxes:
            out.append(
                Finding(
                    "ERROR",
                    "body",
                    "checkbox-missing",
                    "`## Checklist` has no checkboxes.",
                    "Paste the list back from .github/PULL_REQUEST_TEMPLATE.md.",
                )
            )
        elif not checked:
            out.append(
                Finding(
                    "ERROR",
                    "body",
                    "checkbox-unticked",
                    "Nothing is ticked in `## Checklist`.",
                    "Tick every box that applies; an all-empty list reads as "
                    "'not self-reviewed'.",
                )
            )

    desc = find_section(sections, "Description")
    if desc is not None and not desc.is_empty:
        words = len(desc.text.split())
        if words > DESCRIPTION_MAX_WORDS:
            out.append(
                Finding(
                    "WARN",
                    "body",
                    "description-bloated",
                    f"`## Description` is {words} words.",
                    f"Trim to <= {DESCRIPTION_MAX_WORDS}: one sentence on what "
                    "changed, then a bullet per user-visible change.",
                )
            )
        bullets = [ln for ln in desc.lines if LIST_RE.match(ln)]
        if len(bullets) > 12:
            out.append(
                Finding(
                    "WARN",
                    "body",
                    "description-bullets",
                    f"{len(bullets)} bullets in `## Description`.",
                    "Group them; a reviewer scans, not reads.",
                )
            )

    tested = find_section(sections, "Testing")
    if tested is not None and tested.is_empty and not docs_only:
        out.append(
            Finding(
                "WARN",
                "body",
                "tested-empty",
                "`## Testing` is empty.",
                "List the exact commands you ran and their results.",
            )
        )

    comments = COMMENT_RE.findall(body)
    if len(comments) >= 3:
        out.append(
            Finding(
                "WARN",
                "body",
                "body-comments",
                f"{len(comments)} template `<!-- ... -->` hints are still in the body.",
                "Delete the hints once each section is written.",
            )
        )

    in_code = fence_mask(lines)
    wrapped = 0
    for i in range(len(lines) - 1):
        a, b = lines[i], lines[i + 1]
        if in_code[i] or in_code[i + 1]:
            continue
        if HEADING_RE.match(a) or HEADING_RE.match(b) or LIST_RE.match(b):
            continue
        if (
            a.strip()
            and b.strip()
            and 40 <= len(a.rstrip()) <= 90
            and not a.rstrip().endswith((".", ":", "|"))
        ):
            wrapped += 1
    if wrapped >= 6:
        out.append(
            Finding(
                "WARN",
                "body",
                "body-hardwrap",
                f"~{wrapped} hard-wrapped prose lines.",
                "GitHub reflows paragraphs; one line per paragraph survives "
                "edits and diffs better.",
            )
        )
    return out


def unwrap_paragraphs(text: str) -> str:
    """Join hard-wrapped prose into one line per paragraph."""
    lines = text.splitlines()
    in_code = fence_mask(lines)
    out: list[str] = []
    for i, line in enumerate(lines):
        joinable = (
            out
            and not in_code[i]
            and line.strip()
            and out[-1].strip()
            and not HEADING_RE.match(line)
            and not LIST_RE.match(line)
            and "|" not in line
            and not HEADING_RE.match(out[-1])
            and not LIST_RE.match(out[-1])
            and "|" not in out[-1]
            and not (i and in_code[i - 1])
        )
        if joinable:
            out[-1] = out[-1].rstrip() + " " + line.strip()
        else:
            out.append(line)
    return "\n".join(out)


def fix_body(body: str, *, unwrap: bool = False) -> str:
    """De-indent a uniformly-indented body and drop the template hints."""
    lines = body.replace("\r\n", "\n").splitlines()
    in_code = fence_mask(lines)

    mangled = any(
        f.code in ("body-indented-heading", "body-indented")
        for f in check_indentation(body)
    )
    if mangled:
        positives = [
            line_indent(line)
            for i, line in enumerate(lines)
            if not in_code[i] and line.strip() and line_indent(line) > 0
        ]
        offset = min(positives) if positives else 0
    else:
        offset = 0

    out = []
    for i, line in enumerate(lines):
        fixed = line[offset:] if offset and line_indent(line) >= offset else line
        if in_code[i]:
            out.append(fixed)
            continue
        if HEADING_RE.match(fixed) or fixed.strip().count("|") >= 2:
            fixed = fixed.lstrip()
        out.append(fixed.rstrip())

    text = strip_comments_outside_fences("\n".join(out))
    if unwrap:
        text = unwrap_paragraphs(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


TEMPLATE = """\
## Description

<one sentence: what this PR changes and where>

- <user-visible change 1>
- <user-visible change 2>

## Testing

- `<exact command run>` (<result>)

## Manual verification

Not needed

## Checklist

- [ ] My code follows the code style of this project.
- [ ] I have updated related documentation when needed.
- [ ] I have added tests to cover my changes when needed.
- [ ] All new and existing tests passed.
"""


def report(findings: list[Finding], *, title: str, quiet: bool, scope: str) -> int:
    errors = [f for f in findings if f.level == "ERROR"]
    warns = [f for f in findings if f.level == "WARN"]

    print(f"PR title: {title}")
    print(f"{len(errors)} error(s), {len(warns)} warning(s)\n")
    for f in errors + warns:
        mark = "E" if f.level == "ERROR" else "W"
        print(f"[{mark}] {f.where}/{f.code}: {f.message}")
        if f.fix and not quiet:
            for line in textwrap.wrap(f.fix, 74):
                print(f"      -> {line}")
    if not findings:
        print(
            f"clean — {scope} checked, nothing to fix against CONTRIBUTING.md\n"
            "and .github/PULL_REQUEST_TEMPLATE.md."
        )
    return 1 if errors else 0


def load(args: argparse.Namespace) -> tuple[str, str]:
    if args.pr:
        return fetch_pr(args.pr, repo=args.repo)
    title = args.title or ""
    body = ""
    if args.body_file:
        with open(args.body_file, encoding="utf-8") as fh:
            body = fh.read().replace("\r\n", "\n")
    return title, body


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--repo", default="RLinf/RPent", help="owner/name of the PR's repo")
    sub = p.add_subparsers(dest="cmd", required=True)

    lint = sub.add_parser("lint", help="check a PR or a draft")
    lint.add_argument("--pr", help="PR number (fetched with `gh pr view`)")
    lint.add_argument("--title", help="draft title")
    lint.add_argument("--body-file", help="file holding the draft body")
    lint.add_argument(
        "--docs-only", action="store_true", help="skip the testing-section warning"
    )
    lint.add_argument("--quiet", action="store_true", help="omit the suggested fixes")

    fix = sub.add_parser("fix", help="print a de-indented, hint-free body")
    fix.add_argument("--pr", help="PR number")
    fix.add_argument("--body-file", help="file holding the body")
    fix.add_argument(
        "--unwrap",
        action="store_true",
        help="also join hard-wrapped prose into one line per paragraph",
    )

    sub.add_parser("template", help="print the filled-in skeleton")

    args = p.parse_args(argv)

    if args.cmd == "template":
        print(TEMPLATE, end="")
        return 0

    if args.cmd == "fix":
        if args.pr:
            _, body = fetch_pr(args.pr, repo=args.repo)
        elif args.body_file:
            with open(args.body_file, encoding="utf-8") as fh:
                body = fh.read()
        else:
            sys.exit("error: fix needs --pr or --body-file")
        print(fix_body(body, unwrap=args.unwrap), end="")
        return 0

    check_t = args.pr is not None or args.title is not None
    check_b = args.pr is not None or args.body_file is not None
    if not check_t and not check_b:
        sys.exit("error: lint needs --pr, or --title and/or --body-file")

    title, body = load(args)
    findings: list[Finding] = []
    if check_t:
        findings += check_title(title)
    if check_b:
        findings += check_body(body, docs_only=args.docs_only)
    scope = " and ".join(
        part for part, on in (("title", check_t), ("body", check_b)) if on
    )
    return report(
        findings, title=title or "(none given)", quiet=args.quiet, scope=scope
    )


if __name__ == "__main__":
    raise SystemExit(main())
