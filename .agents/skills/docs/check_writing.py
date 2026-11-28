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

"""Writing rules that a script can check on RPent docs, applied by page type.

Each page's type comes from where it lives (``rules/base.md``). These rules
catch the mechanical forms of the mistakes in ``rules/mistakes.md``: internal
code names on user-facing pages, missing simulator sections, code in headings,
and English and Chinese phrasing that the language rules ban. ``docs/check.py``
runs them on the lines a change adds.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import check_rst_markup
import route

TYPE_PREFIXES = (
    ("get_started/", "guide"),
    ("simulators/", "simulator"),
    ("real_world_robots/", "simulator"),
    ("real_world_demos/", "guide"),
    ("guides/", "guide"),
    ("toolkits/", "guide"),
    ("development/", "development"),
    ("leaderboard/", "reference"),
    ("resources/", "other"),
)
SIMULATOR_PAGE = re.compile(
    r"docs/source-(en|zh)/rst_source/simulators/(?!index\.rst|_)[^/]+\.rst"
)
SIMULATOR_REQUIRED = {
    "en": ("Overview", "Tasks", "Observation and Action", "Run a Task", "View Results"),
    "zh": ("概览", "任务", "观测与动作", "运行一个任务", "查看结果"),
}
SIMULATOR_INSTALL = {
    "en": ("Installation", "Installation and Resources"),
    "zh": ("安装", "安装与资源准备"),
}

#: Severity of an internal code name, by page type; other types are not checked.
CODE_NAME_SEVERITY = {"simulator": "major", "guide": "major"}

LITERAL = re.compile(r"``([^`\n]+)``")
SPHINX_API_ROLE = re.compile(
    r":(?:py:)?(?:meth|func|class|mod|attr|obj|data|exc):`([^`\n]+)`"
)
ROLE_TARGET = re.compile(r"<([^>]+)>")
#: A name that reads as code rather than a word or product: CamelCase with an
#: inner capital, or snake_case, at least four characters long.
CODE_LIKE = re.compile(
    r"^(?=.{4,})(?:[A-Za-z0-9]*[a-z][A-Z][A-Za-z0-9]*|[a-z0-9]+(?:_[a-z0-9]+)+)$"
)
FILE_NAME = re.compile(r"\.[A-Za-z]{1,12}$")
DEFINITION = re.compile(r"^\s*(?:class|def)\s+([A-Za-z_]\w*)", re.MULTILINE)
YAML_KEY = re.compile(r"^\s*([A-Za-z_]\w*)\s*:", re.MULTILINE)
UNDERLINE = re.compile(r"^([=\-~^\"'`#*+])\1{2,}\s*$")
MARKUP = re.compile(r"``[^`]*``|:\w+:`[^`]*`|`[^`]*`_{0,2}|\*\*|\|")

#: User-facing tool and planner names that appear legitimately in user guides.
USER_FACING_NAMES = frozenset(
    {
        "claude_code",
        "gemini_cli",
        "move_to",
        "move_to_pose",
        "set_gripper",
        "get_image",
        "get_state",
        "back_project",
        "segment",
        "geometry",
        "wam_act",
        "efe_submit",
        "efe_reset",
        "RPentEfeAgent",
        "history_length",
        "skills",
        "finish",
        "terminated",
    }
)

EN_PHRASES = re.compile(
    r"\b(simply|easily|obviously|basically|seamless(?:ly)?|powerful|robust|"
    r"comprehensive|cutting-edge|state-of-the-art|leverag(?:e|es|ed|ing)|crucial|"
    r"pivotal|delve|showcases?|empowers?|it is worth noting|please note|note that|"
    r"in order to|allows you to|enables you to|a wide range of|currently|not just)\b",
    re.IGNORECASE,
)
HAN = r"\u4e00-\u9fff"
ZH_PATTERNS = (
    (re.compile("您"), "address the reader as 你, or omit the subject"),
    (re.compile("[「」]"), "use “” for quotes"),
    (
        re.compile("值得注意的是|需要注意的是"),
        "state the fact, or move it into a note or warning",
    ),
    (re.compile("进行"), "use the verb directly: 进行训练 → 训练"),
    (
        re.compile("当[^，。；：！？]{1,30}时"),
        "drop 当……时 where ……后 or ……时 alone reads naturally",
    ),
    (
        re.compile("(?:不是|并非)[^。！？\n]{1,40}而是"),
        "state directly what it is instead of framing it as 不是……而是……",
    ),
    (
        re.compile(f"(?<=[{HAN}\u3001-\u301f\uff01-\uff5e])[,;:!?](?=\\s|$|[{HAN}])"),
        "use full-width punctuation in a Chinese sentence",
    ),
    (
        re.compile(f"[{HAN}][A-Za-z0-9]|[A-Za-z0-9][{HAN}]"),
        "put a space between Chinese and English or digits",
    ),
)


def page_type(rel: str) -> str | None:
    """The page type of a docs source file, from the section it lives in."""
    match = re.match(r"docs/source-(?:en|zh)/rst_source/(.+\.rst)$", rel)
    if not match:
        return None
    sub = match[1]
    name = sub.rsplit("/", 1)[-1]
    if name == "index.rst" or name.endswith("_index.rst"):
        return "index"
    if sub == "guides/cli.rst":
        return "reference"
    return next(
        (kind for prefix, kind in TYPE_PREFIXES if sub.startswith(prefix)), "other"
    )


def _decorator_is_tool(dec: ast.expr) -> bool:
    target = dec.func if isinstance(dec, ast.Call) else dec
    if isinstance(target, ast.Name):
        return target.id == "tool"
    if isinstance(target, ast.Attribute):
        return target.attr == "tool"
    return False


def _tool_names(root: Path) -> set[str]:
    """Registered user-facing tool names declared with ``@tool`` or ``Tool``."""
    names: set[str] = set()
    for folder in ("rpent", "robots", "components", "toolkits"):
        target = root / folder
        if not target.is_dir():
            continue
        for path in target.rglob("*.py"):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for dec in node.decorator_list:
                        if not _decorator_is_tool(dec):
                            continue
                        explicit: str | None = None
                        if isinstance(dec, ast.Call):
                            for kw in dec.keywords:
                                if (
                                    kw.arg == "name"
                                    and isinstance(kw.value, ast.Constant)
                                    and isinstance(kw.value.value, str)
                                ):
                                    explicit = kw.value.value
                        names.add(explicit or node.name)
                elif isinstance(node, ast.Call):
                    callee = (
                        node.func.id
                        if isinstance(node.func, ast.Name)
                        else node.func.attr
                        if isinstance(node.func, ast.Attribute)
                        else ""
                    )
                    if callee != "Tool":
                        continue
                    explicit = None
                    for kw in node.keywords:
                        if (
                            kw.arg == "name"
                            and isinstance(kw.value, ast.Constant)
                            and isinstance(kw.value.value, str)
                        ):
                            explicit = kw.value.value
                    if explicit is not None:
                        names.add(explicit)
                    elif node.args:
                        first = node.args[0]
                        if isinstance(first, ast.Constant) and isinstance(
                            first.value, str
                        ):
                            names.add(first.value)
                        elif isinstance(first, ast.Attribute):
                            names.add(first.attr)
                        elif isinstance(first, ast.Name):
                            names.add(first.id)
    return names


def _json_keys(obj: object) -> set[str]:
    """Recursively collect string object keys from parsed JSON data."""
    keys: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str):
                keys.add(k)
            keys.update(_json_keys(v))
    elif isinstance(obj, list):
        for item in obj:
            keys.update(_json_keys(item))
    return keys


def code_names(root: Path) -> frozenset[str]:
    """Classes and functions under ``rpent/``, ``robots/``, ``components/``, and ``toolkits/``, minus config/user names."""
    defined: set[str] = set()
    keys: set[str] = set()
    for folder in ("rpent", "robots", "components", "toolkits"):
        target = root / folder
        if not target.is_dir():
            continue
        for path in target.rglob("*.py"):
            defined.update(
                DEFINITION.findall(path.read_text(encoding="utf-8", errors="ignore"))
            )
        for pattern in ("*.yaml", "*.yml"):
            for path in target.rglob(pattern):
                keys.update(
                    YAML_KEY.findall(path.read_text(encoding="utf-8", errors="ignore"))
                )
        for path in target.rglob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            except json.JSONDecodeError:
                continue
            keys.update(_json_keys(data))
    allowed = keys | USER_FACING_NAMES | _tool_names(root)
    return frozenset(n for n in defined - allowed if CODE_LIKE.match(n))


def headings(lines: list[str]) -> list[tuple[int, str]]:
    """``(line, title)`` of every section title: a line underlined by punctuation."""
    found = []
    for i, (title, under) in enumerate(zip(lines, lines[1:]), 1):
        text = title.strip()
        if (
            text
            and not text.startswith("..")
            and UNDERLINE.match(under)
            and len(under.strip()) >= len(text) // 2
        ):
            found.append((i, text))
    return found


def prose_rows(source: str) -> list[tuple[int, str]]:
    """Every prose line with its 1-based line number, skipping verbatim blocks."""
    rows = []
    for block in check_rst_markup.prose_blocks(source):
        rows += list(zip(block.lines, block.text.split("\n")))
    return rows


def check_page(
    path: Path, rel: str, names: frozenset[str], new: bool
) -> list[route.Finding]:
    """The writing rules for one page, given its type and whether it is new."""
    kind = page_type(rel)
    lang = "zh" if rel.startswith("docs/source-zh/") else "en"
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    rows = prose_rows(source)
    findings: list[route.Finding] = []

    def add(severity: str, line: int | None, rule: str, message: str) -> None:
        findings.append(route.Finding(severity, rel, line, rule, message))

    if kind in CODE_NAME_SEVERITY:
        for number, text in rows:
            for token in LITERAL.findall(text):
                token = token.strip().removesuffix("()")
                internal = (
                    token.startswith(("rpent/", "robots/", "components/", "toolkits/"))
                    and ":" not in token
                    and not token.endswith((".yaml", ".yml", ".json"))
                ) or (
                    "/" not in token
                    and not FILE_NAME.search(token)
                    and any(part in names for part in token.split("."))
                )
                if internal:
                    add(
                        CODE_NAME_SEVERITY[kind],
                        number,
                        "docs/code-name",
                        f"``{token}`` is RPent internal code that a reader never "
                        "touches on this page; describe the CLI or user-visible behavior instead",
                    )
            for raw_role in SPHINX_API_ROLE.findall(text):
                match = ROLE_TARGET.search(raw_role)
                target = (
                    (match[1] if match else raw_role)
                    .strip()
                    .lstrip("~")
                    .removesuffix("()")
                )
                if target.startswith(
                    ("rpent.", "robots.", "components.", "toolkits.")
                ) or any(part in names for part in target.split(".")):
                    add(
                        CODE_NAME_SEVERITY[kind],
                        number,
                        "docs/code-name",
                        f"Sphinx API role targeting ``{target}`` exposes RPent internal "
                        "code on this page; describe the CLI or user-visible behavior instead",
                    )

    if kind not in ("reference", "development"):
        for number, title in headings(lines):
            if "``" in title:
                add(
                    "minor",
                    number,
                    "docs/heading",
                    "name the task or idea, not a code identifier",
                )

    if new and SIMULATOR_PAGE.match(rel):
        titles = {title for _, title in headings(lines)}
        missing = [s for s in SIMULATOR_REQUIRED[lang] if s not in titles]
        if not any(opt in titles for opt in SIMULATOR_INSTALL[lang]):
            missing.append(" / ".join(SIMULATOR_INSTALL[lang]))
        if missing:
            add(
                "major",
                None,
                "docs/structure",
                f"a simulator guide needs the sections: {', '.join(missing)}",
            )

    for number, text in rows:
        if lang == "en":
            for match in EN_PHRASES.finditer(text):
                add(
                    "minor",
                    number,
                    "docs/en-style",
                    f"cut or replace {match[0]!r} (rules/english.md)",
                )
        else:
            plain = MARKUP.sub(" ", text)
            for pattern, advice in ZH_PATTERNS:
                if pattern.search(plain):
                    add(
                        "minor",
                        number,
                        "docs/zh-style",
                        f"{advice} (rules/chinese.md)",
                    )
    return findings
