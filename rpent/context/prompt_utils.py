"""Python prompt rendering primitives."""

from __future__ import annotations

import textwrap
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from rpent.utils.templates import substitute_text


@dataclass(frozen=True)
class BulletList:
    """Render items as Markdown bullets."""

    items: tuple[Any, ...]

    def __init__(self, items: Sequence[Any]):
        """Create a bullet list from prompt nodes."""
        object.__setattr__(self, "items", tuple(items))


@dataclass(frozen=True)
class Numbered:
    """Render items as a numbered Markdown list."""

    items: tuple[Any, ...]

    def __init__(self, items: Sequence[Any]):
        """Create a numbered list from prompt nodes."""
        object.__setattr__(self, "items", tuple(items))


PromptNode = str | Mapping[str, Any] | Sequence[Any] | BulletList | Numbered


def format_prompt(
    prompt: PromptNode,
    *,
    variables: Mapping[str, object] | None = None,
) -> str:
    """Render a Python prompt tree into final prompt text."""
    vars_ = {k: str(v) for k, v in (variables or {}).items()}
    return _render(prompt, vars_, depth=0).strip() + "\n"


def _render(value: Any, variables: Mapping[str, str], *, depth: int) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return substitute_text(_clean_text(value), variables, strict=True)
    if isinstance(value, BulletList):
        return _render_list(value.items, variables, depth=depth, ordered=False)
    if isinstance(value, Numbered):
        return _render_list(value.items, variables, depth=depth, ordered=True)
    if isinstance(value, Mapping):
        return _render_mapping(value, variables, depth=depth)
    if isinstance(value, Sequence):
        return _render_list(value, variables, depth=depth, ordered=False)
    return str(value)


def _render_mapping(
    value: Mapping[str, Any],
    variables: Mapping[str, str],
    *,
    depth: int,
) -> str:
    parts: list[str] = []
    for title, body in value.items():
        rendered = _render(body, variables, depth=depth + 1).strip()
        if not rendered:
            continue
        if depth == 0:
            section = [
                "═" * 71,
                title,
                "═" * 71,
                "",
                rendered,
            ]
            parts.append("\n".join(section))
        else:
            parts.append(f"{'#' * min(depth + 2, 6)} {title}\n\n{rendered}")
    return "\n\n\n".join(parts)


def _render_list(
    items: Sequence[Any],
    variables: Mapping[str, str],
    *,
    depth: int,
    ordered: bool,
) -> str:
    rendered_items = [
        _render(item, variables, depth=depth + 1).strip() for item in items
    ]
    rendered_items = [item for item in rendered_items if item]
    blocks: list[str] = []
    for index, item in enumerate(rendered_items, 1):
        prefix = f"{index}. " if ordered else "- "
        item_lines = item.splitlines()
        lines = [prefix + item_lines[0]]
        lines.extend(" " * len(prefix) + line for line in item_lines[1:])
        blocks.append("\n".join(lines))
    separator = "\n\n" if ordered else "\n"
    return separator.join(blocks)


def _clean_text(text: str) -> str:
    return textwrap.dedent(text).strip()
