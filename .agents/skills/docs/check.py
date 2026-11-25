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

"""Check changed RPent docs: build, markup, symbols, links, pairing, and writing.

Usage::

    python3 .agents/skills/docs/check.py              # docs changed against main
    python3 .agents/skills/docs/check.py FILE ...     # specific pages
    python3 .agents/skills/docs/check.py --no-build   # skip the Sphinx build

Page findings count only on lines the change adds (``--all-lines`` counts every
line); the build always covers both trees. The build runs in the active Python
when Sphinx and RPent's docs extensions are already installed, or provisions
``DOCS_VENV`` (default ``<repo>/.docs-venv``) from ``docs/requirements.txt``.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import check_doc_symbols  # noqa: E402
import route  # noqa: E402
from build_docs import build, ensure_venv  # noqa: E402

REQUIREMENTS = route.REPO_ROOT / "docs" / "requirements.txt"
VENV = Path(os.environ.get("DOCS_VENV") or route.REPO_ROOT / ".docs-venv")


def _ensure_docutils(argv: list[str] | None) -> None:
    """Re-exec into the docs virtualenv when ``docutils`` is not installed."""
    try:
        import docutils  # noqa: F401
    except (
        ImportError
    ):  # pragma: no cover - re-exec into docs venv when docutils missing
        explicit = bool(os.environ.get("DOCS_VENV"))
        python = ensure_venv(VENV, REQUIREMENTS, explicit=explicit)
        args = sys.argv[1:] if argv is None else argv
        os.execv(str(python), [str(python), __file__, *args])


LANGS = {"docs/source-en/": "docs/source-zh/", "docs/source-zh/": "docs/source-en/"}
PAIRED_ROOT_DOCS = {
    "README.md": "README.zh-CN.md",
    "README.zh-CN.md": "README.md",
    "CONTRIBUTING.md": "CONTRIBUTING_zh.md",
    "CONTRIBUTING_zh.md": "CONTRIBUTING.md",
}
RTD_LINK = re.compile(r"rpent\.readthedocs\.io/(?:en|zh-cn)/latest/rst_source/")
WARNING_AT = re.compile(
    r"^(?P<path>/[^:\s]+)(?::(?P<line>\d+))?: (?:WARNING|ERROR|SEVERE): "
)
LIST_MARKER = re.compile(r"^(?:\* -|[-*+]|#\.|\d+\.)\s+")
TABLE_BORDER = re.compile(r"^\s*=+(?:\s+=+)+\s*$")


def relative(path: str, root: Path | None = None) -> str:
    """A path relative to the repository when it lies inside it."""
    base = root or route.REPO_ROOT
    try:
        return Path(path).resolve().relative_to(base).as_posix()
    except ValueError:
        return path


def counterpart(path: str) -> str | None:
    """The other language's file for a docs page, README, or CONTRIBUTING guide."""
    if path in PAIRED_ROOT_DOCS:
        return PAIRED_ROOT_DOCS[path]
    for prefix, other in LANGS.items():
        if path.startswith(prefix):
            return other + path[len(prefix) :]
    return None


def check_parity(
    changes: list[route.Change] | tuple[route.Change, ...],
    root: Path | None = None,
) -> list[route.Finding]:
    base = root or route.REPO_ROOT
    changed = {c.path for c in changes}
    findings: list[route.Finding] = []
    for change in changes:
        other = counterpart(change.path)
        if other is None or not change.path.endswith((".rst", ".md")):
            continue
        exists = (base / other).exists()
        if change.status == "D" and exists and other not in changed:
            findings.append(
                route.Finding(
                    "major",
                    change.path,
                    None,
                    "docs/parity",
                    f"deleted, but {other} still exists",
                )
            )
        elif change.status != "D" and not exists:
            findings.append(
                route.Finding(
                    "major", change.path, None, "docs/parity", f"{other} does not exist"
                )
            )
        elif change.status != "D" and other not in changed:
            findings.append(
                route.Finding(
                    "minor",
                    change.path,
                    None,
                    "docs/parity",
                    f"changed while {other} did not; check whether it needs the same change",
                )
            )
    return findings


def toctree_entries(path: Path) -> set[str]:
    """The targets of every ``.. toctree::`` on a page, without captions."""
    entries: set[str] = set()
    lines = path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        if line.strip() != ".. toctree::":
            continue
        indent = len(line) - len(line.lstrip())
        for body in lines[i + 1 :]:
            stripped = body.strip()
            if stripped and len(body) - len(body.lstrip()) <= indent:
                break
            if stripped and not stripped.startswith(":"):
                target = re.search(r"<([^>]+)>$", stripped)
                entries.add(target.group(1) if target else stripped)
    return entries


def check_toctrees(pages: list[str], root: Path | None = None) -> list[route.Finding]:
    base = root or route.REPO_ROOT
    findings: list[route.Finding] = []
    pairs = {
        (page, counterpart(page))
        if page.startswith("docs/source-en/")
        else (counterpart(page), page)
        for page in pages
        if counterpart(page)
    }
    for en_page, zh_page in sorted(pairs):
        assert en_page is not None and zh_page is not None
        mine, theirs = base / en_page, base / zh_page
        if not (mine.exists() and theirs.exists()):
            continue
        en, zh = toctree_entries(mine), toctree_entries(theirs)
        if en != zh:
            only = "; ".join(
                f"only in {lang}: {', '.join(sorted(s))}"
                for lang, s in (("EN", en - zh), ("ZH", zh - en))
                if s
            )
            findings.append(
                route.Finding(
                    "major", en_page, None, "docs/parity", f"toctrees differ ({only})"
                )
            )
    return findings


def is_cjk(char: str) -> bool:
    return unicodedata.east_asian_width(char) in ("W", "F")


def check_hard_wrap(page: Path, rel: str) -> list[route.Finding]:
    """Unescaped Chinese prose continuing on the next line renders a visible space.

    RPent's STYLE_GUIDE permits either single-line Chinese paragraphs or a
    trailing backslash (``\\``) that escapes the newline; bare CJK wraps are flagged.
    """
    findings: list[route.Finding] = []
    import check_rst_markup

    source = page.read_text(encoding="utf-8")
    for block in check_rst_markup.prose_blocks(source):
        rows = block.text.split("\n")
        if any(TABLE_BORDER.match(row) for row in rows):
            continue
        for previous, current, number in zip(rows, rows[1:], block.lines[1:]):
            prev_stripped = previous.rstrip()
            if not prev_stripped or prev_stripped.endswith("\\"):
                continue
            head = previous.lstrip()
            marker = LIST_MARKER.match(head)
            column = len(previous) - len(head) + (marker.end() if marker else 0)
            text = current[column:]
            if (
                is_cjk(prev_stripped[-1])
                and len(current) - len(current.lstrip()) == column
                and text
                and is_cjk(text[0])
            ):
                findings.append(
                    route.Finding(
                        "minor",
                        rel,
                        number,
                        "docs/zh-hard-wrap",
                        "Chinese prose continues from the line above without '\\'; "
                        "join the lines or escape the line break",
                    )
                )
    return findings


def check_page(page: Path, rel: str, known: set[str]) -> list[route.Finding]:
    """Markup, symbols (on Development pages), RTD links, and Chinese wrapping."""
    import check_rst_markup

    findings = [
        route.Finding(
            "critical" if f.level == "ERROR" else "major",
            rel,
            f.line,
            "docs/markup",
            f.message,
        )
        for f in check_rst_markup.check_file(page)
    ]
    findings += [
        route.Finding(
            "major",
            rel,
            number,
            "docs/symbol",
            f"``{token}`` names nothing in the repository; rename, delete, or "
            "add it to ALLOWED in check_doc_symbols.py if it is illustrative",
        )
        for number, token in check_doc_symbols.literals(page)
        if rel.startswith(check_doc_symbols.DOC_ROOTS) and token not in known
    ]
    for number, line in enumerate(page.read_text(encoding="utf-8").splitlines(), 1):
        if RTD_LINK.search(line):
            findings.append(
                route.Finding(
                    "major",
                    rel,
                    number,
                    "docs/rtd-link",
                    "hardcoded Read the Docs link to an RPent page; use :doc: or :ref:",
                )
            )
    if rel.startswith("docs/source-zh/"):
        findings += check_hard_wrap(page, rel)
    return findings


def check_build() -> list[route.Finding]:
    explicit_venv = (
        Path(os.environ["DOCS_VENV"]) if os.environ.get("DOCS_VENV") else None
    )
    python = ensure_venv(
        explicit_venv or VENV, REQUIREMENTS, explicit=explicit_venv is not None
    )
    findings: list[route.Finding] = []
    with tempfile.TemporaryDirectory(prefix="rpent-docs-") as output:
        for lang in ("en", "zh"):
            source = route.REPO_ROOT / "docs" / f"source-{lang}"
            print(f"building {source.relative_to(route.REPO_ROOT)}", flush=True)
            for record in build(python, source, Path(output) / lang):
                head, *rest = record.splitlines()
                where = WARNING_AT.match(head)
                path, line = f"docs/source-{lang}", None
                if where:
                    path = relative(where["path"])
                    line = int(where["line"]) if where["line"] else None
                    head = head[where.end() :]
                findings.append(
                    route.Finding(
                        "critical",
                        path,
                        line,
                        "docs/build",
                        " ".join([head, *[r.strip() for r in rest]])[:300],
                    )
                )
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = route.check_parser(__doc__.splitlines()[0])
    parser.add_argument("--no-build", action="store_true", help="skip the Sphinx build")
    args = parser.parse_args(argv)
    selection = route.select("docs", args)
    changes = selection.changes
    pages = [
        c.path
        for c in changes
        if c.status != "D" and c.path.endswith(".rst") and c.path.startswith("docs/")
    ]
    findings = check_parity(changes) + check_toctrees(pages)
    if pages:
        _ensure_docutils(argv)
        import check_writing

        known = (
            check_doc_symbols.vocabulary(route.REPO_ROOT)
            | check_doc_symbols.ALLOWED
            | check_doc_symbols.EXTERNAL
            | check_doc_symbols.KNOWN_DRIFT
            if any(p.startswith(check_doc_symbols.DOC_ROOTS) for p in pages)
            else set()
        )
        names = check_writing.code_names(route.REPO_ROOT)
        added = {c.path for c in changes if c.status == "A"}
        for page in pages:
            path = route.REPO_ROOT / page
            page_findings = check_page(path, page, known)
            page_findings += check_writing.check_page(path, page, names, page in added)
            findings += selection.relevant(page_findings)
    if args.no_build:
        print("skipped: the Sphinx build (--no-build)")
    elif any(c.path.startswith("docs/") for c in changes):
        findings += check_build()
    return route.report("docs", findings)


if __name__ == "__main__":
    raise SystemExit(main())
