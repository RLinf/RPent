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

"""Remove ``explanation`` annotations from recorded robo code.

Only side-effect-free annotations on unaliased public ``robo.<method>`` calls
are removed. Anything the exporter cannot prove harmless to execution raises
``TraceError``, so an exported row never holds code whose behavior changed.
"""

from __future__ import annotations

import ast
from collections.abc import Collection

from components.tools.trace_export.trace import require

PROJECTION = "robo-explanation-strip.v1"
MAX_EXPLANATION_CHARS = 800
ANNOTATION = "explanation"
_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
_DYNAMIC_CALLS = frozenset(
    {"exec", "eval", "globals", "locals", "compile", "__import__"}
)


def is_robo_call(node: ast.AST) -> bool:
    """Return whether ``node`` is a call of the form ``robo.<name>(...)``."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "robo"
    )


def _rebinds(scope: ast.AST, name: str) -> list[ast.Name]:
    return [
        n
        for n in ast.walk(scope)
        if isinstance(n, ast.Name)
        and n.id == name
        and isinstance(n.ctx, (ast.Store, ast.Del))
    ]


class Analysis:
    """Parent links and scope queries over one parsed code cell."""

    def __init__(self, tree: ast.Module):
        self.tree = tree
        self.parents = {
            child: node
            for node in ast.walk(tree)
            for child in ast.iter_child_nodes(node)
        }

    def scope_of(self, node: ast.AST) -> ast.AST:
        """Return the innermost function or lambda around ``node``, else the module."""
        while node in self.parents:
            node = self.parents[node]
            if isinstance(node, _SCOPES):
                return node
        return self.tree

    def stable_parameter(self, node: ast.AST) -> bool:
        """Return whether ``node`` names a function parameter that is never rebound."""
        if not isinstance(node, ast.Name):
            return False
        scope = self.scope_of(node)
        if scope is self.tree:
            return False
        args = scope.args
        names = {a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)}
        return node.id in names and not _rebinds(scope, node.id)

    def known_kwargs(self, node: ast.AST, scope: ast.AST, seen: tuple = ()) -> bool:
        """Return whether ``**node`` provably carries no annotation key."""
        if isinstance(node, ast.Dict):
            return all(
                isinstance(k, ast.Constant)
                and isinstance(k.value, str)
                and k.value != ANNOTATION
                for k in node.keys
            )
        if isinstance(node, ast.IfExp):
            return self.known_kwargs(node.body, scope, seen) and self.known_kwargs(
                node.orelse, scope, seen
            )
        if not isinstance(node, ast.Name) or node.id in seen:
            return False
        assignments = _rebinds(scope, node.id)
        if len(assignments) != 1:
            return False
        for use in ast.walk(scope):
            if not (
                isinstance(use, ast.Name)
                and use.id == node.id
                and isinstance(use.ctx, ast.Load)
            ):
                continue
            keyword = self.parents.get(use)
            call = self.parents.get(keyword)
            # Any other use could mutate or alias the dict, e.g. kw.update(...).
            if not (
                isinstance(keyword, ast.keyword)
                and keyword.arg is None
                and is_robo_call(call)
            ):
                return False
        assignment = self.parents[assignments[0]]
        return (
            isinstance(assignment, ast.Assign)
            and len(assignment.targets) == 1
            and self.scope_of(assignment) is scope
            and self.known_kwargs(assignment.value, scope, (*seen, node.id))
        )


def _check_binding(analysis: Analysis, node: ast.AST, methods: Collection[str]):
    """Reject code that shadows, aliases or dynamically reaches ``robo``."""
    if isinstance(node, ast.arg):
        require(node.arg != "robo", "shadowed robo parameter cannot be projected")
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        require(node.name != "robo", "shadowed robo definition cannot be projected")
    if isinstance(node, ast.alias):
        bound = node.asname or node.name.split(".")[0]
        require(bound != "robo", "shadowed robo import cannot be projected")
    if isinstance(node, ast.Name) and node.id == "robo":
        attr = analysis.parents.get(node)
        call = analysis.parents.get(attr)
        require(
            isinstance(node.ctx, ast.Load)
            and isinstance(attr, ast.Attribute)
            and attr.value is node
            and isinstance(call, ast.Call)
            and call.func is attr
            and attr.attr in methods,
            "aliased/dynamic robo access cannot be projected",
        )
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
        require(
            node.slice.value != ANNOTATION,
            "code depends on removed explanation field",
        )
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        require(
            node.func.id not in _DYNAMIC_CALLS,
            "dynamic namespace cannot be audited",
        )


def _evaluate_and_drop(value: ast.expr) -> ast.Call:
    """Keep an f-string's evaluation (it may call __format__ or raise), drop its value."""
    drop = ast.Lambda(
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg="_annotation")],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=ast.Dict(keys=[], values=[]),
    )
    return ast.Call(func=drop, args=[value], keywords=[])


def strip_call_annotations(
    analysis: Analysis, methods: Collection[str]
) -> dict[str, int]:
    """Drop ``explanation=`` from public robo calls, editing the tree in place.

    Raises ``TraceError`` if an annotation cannot be removed without risking a
    change in what the cell executes.

    Args:
        analysis: Parent links over the parsed code cell.
        methods: Public ``robo`` method names the cell may call.

    Returns:
        Counts of removed, forwarded and evaluation-preserved annotations.
    """
    counts = {
        "removed_keywords": 0,
        "removed_forwarded_keywords": 0,
        "preserved_annotation_evaluations": 0,
    }
    for node in ast.walk(analysis.tree):
        _check_binding(analysis, node, methods)
        if not is_robo_call(node):
            continue
        scope = analysis.scope_of(node)
        require(
            all(
                k.arg is not None or analysis.known_kwargs(k.value, scope)
                for k in node.keywords
            ),
            "dynamic robo kwargs cannot be projected",
        )
        if node.func.attr == "parallel":
            require(
                all(k.arg == "actions" for k in node.keywords),
                "parallel accepts actions only; outer annotations cannot be removed",
            )
        for keyword in node.keywords:
            if keyword.arg != ANNOTATION:
                continue
            counts["removed_keywords"] += 1
            if isinstance(keyword.value, ast.JoinedStr):
                keyword.arg = None
                keyword.value = _evaluate_and_drop(keyword.value)
                counts["preserved_annotation_evaluations"] += 1
            elif analysis.stable_parameter(keyword.value):
                counts["removed_forwarded_keywords"] += 1
            else:
                text = keyword.value
                require(
                    isinstance(text, ast.Constant) and isinstance(text.value, str),
                    "nonliteral explanation may change execution when removed",
                )
                require(
                    1 <= len(text.value) <= MAX_EXPLANATION_CHARS,
                    "invalid explanation annotation cannot be safely projected",
                )
        node.keywords = [k for k in node.keywords if k.arg != ANNOTATION]
    return counts
