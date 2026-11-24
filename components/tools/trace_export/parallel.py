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

"""Remove ``explanation`` from the literal child actions of ``robo.parallel``.

A child action is a dictionary inside the call's ``actions`` list. The list and
its dictionaries may be bound to a module-level name once, but that name must
not be used anywhere else, so removing a key cannot change other code.
"""

from __future__ import annotations

import ast

from components.tools.trace_export.projection import (
    ANNOTATION,
    MAX_EXPLANATION_CHARS,
    is_robo_call,
)
from components.tools.trace_export.trace import require

_DYNAMIC = "dynamic parallel actions cannot be safely projected"


def _is_parallel(node: ast.AST) -> bool:
    return is_robo_call(node) and node.func.attr == "parallel"


def strip_parallel_annotations(tree: ast.Module) -> int:
    """Drop ``explanation`` keys from literal ``robo.parallel`` child actions.

    Raises ``TraceError`` if the actions are not a literal list of literal
    dictionaries, or if an annotation is not a short string literal.

    Args:
        tree: Parsed code cell, edited in place.

    Returns:
        The number of child annotations removed.
    """
    bindings = {
        n.targets[0].id: n.value
        for n in tree.body
        if isinstance(n, ast.Assign)
        and len(n.targets) == 1
        and isinstance(n.targets[0], ast.Name)
    }
    assigned: dict[str, int] = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            assigned[n.id] = assigned.get(n.id, 0) + 1
    allowed_loads: set[int] = set()
    used: set[str] = set()
    processed: set[int] = set()

    def literal(node: ast.AST, stack: tuple[str, ...] = ()) -> ast.AST:
        if not isinstance(node, ast.Name):
            return node
        require(
            node.id in bindings and assigned.get(node.id) == 1 and node.id not in stack,
            _DYNAMIC,
        )
        allowed_loads.add(id(node))
        used.add(node.id)
        return literal(bindings[node.id], (*stack, node.id))

    removed = 0
    for node in ast.walk(tree):
        if not _is_parallel(node):
            continue
        named = [k.value for k in node.keywords if k.arg == "actions"]
        require(
            all(k.arg == "actions" for k in node.keywords),
            "parallel accepts actions only",
        )
        require(len(node.args) + len(named) == 1, "parallel needs one actions list")
        actions = literal(node.args[0] if node.args else named[0])
        require(isinstance(actions, (ast.List, ast.Tuple)), _DYNAMIC)
        for item in actions.elts:
            child = literal(item)
            require(
                isinstance(child, ast.Dict)
                and all(
                    isinstance(k, ast.Constant) and isinstance(k.value, str)
                    for k in child.keys
                ),
                "parallel children must be literal ACT dictionaries",
            )
            if id(child) in processed:
                continue
            processed.add(id(child))
            pairs = list(zip(child.keys, child.values))
            require(
                len({k.value for k, _ in pairs}) == len(pairs),
                "duplicate parallel child fields",
            )
            for key, value in pairs:
                if key.value != ANNOTATION:
                    continue
                require(
                    isinstance(value, ast.Constant)
                    and isinstance(value.value, str)
                    and 1 <= len(value.value) <= MAX_EXPLANATION_CHARS,
                    "nonliteral parallel explanation cannot be projected",
                )
                removed += 1
            child.keys = [k for k, _ in pairs if k.value != ANNOTATION]
            child.values = [v for k, v in pairs if k.value != ANNOTATION]
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id in used
        ):
            require(
                id(node) in allowed_loads,
                "parallel annotation dictionary is reused outside its action group",
            )
    return removed
