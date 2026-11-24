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

"""Project a recorded code cell and check its namespace prefix."""

from __future__ import annotations

import ast
import builtins
import symtable
from collections.abc import Collection, Iterable
from typing import Any

from components.tools.trace_export.parallel import strip_parallel_annotations
from components.tools.trace_export.projection import (
    PROJECTION,
    Analysis,
    strip_call_annotations,
)
from components.tools.trace_export.trace import TraceError, require, sha256_hex


def project_code(code: str, methods: Collection[str]) -> tuple[str, dict[str, Any]]:
    """Return the projected form of a recorded cell and an audit record.

    Args:
        code: Python source exactly as the model submitted it.
        methods: Public ``robo`` method names the cell may call.

    Returns:
        The projected source and the counts and checksums that show only
        annotations changed.

    Raises:
        TraceError: If the source does not compile or cannot be projected safely.
    """
    try:
        tree = ast.parse(code)
        compile(tree, "<source>", "exec")
    except (SyntaxError, ValueError) as exc:
        raise TraceError(f"recorded code does not compile: {exc}") from exc
    counts = strip_call_annotations(Analysis(tree), methods)
    removed_children = strip_parallel_annotations(tree)
    expected = ast.dump(tree, include_attributes=False)
    ast.fix_missing_locations(tree)
    projected = ast.unparse(tree) + "\n"
    require(
        ast.dump(ast.parse(projected), include_attributes=False) == expected,
        "projection changed non-annotation AST",
    )
    compile(projected, "<projected>", "exec")
    return projected, {
        "rule": PROJECTION,
        **counts,
        "removed_child_annotations": removed_children,
        "original_sha256": sha256_hex(code.encode()),
        "projected_sha256": sha256_hex(projected.encode()),
        "non_annotation_ast_sha256": sha256_hex(expected.encode()),
    }


def _global_loads(scope: symtable.SymbolTable) -> set[str]:
    names = {
        s.get_name() for s in scope.get_symbols() if s.is_referenced() and s.is_global()
    }
    for child in scope.get_children():
        names |= _global_loads(child)
    return names


def namespace_check(
    code: str, defined: Iterable[str], injected: Collection[str] = ("robo",)
) -> set[str]:
    """Check that a cell only uses names the earlier cells already defined.

    Cells share one persistent namespace, so an exported row is only valid when
    every cell before it is present in the export.

    Args:
        code: Projected source of the cell.
        defined: Names bound by the preceding cells.
        injected: Names the runtime provides to every cell.

    Returns:
        The names defined after this cell runs.

    Raises:
        TraceError: If the cell reads a name that nothing defines.
    """
    known = set(defined)
    try:
        table = symtable.symtable(code, "<cell>", "exec")
    except SyntaxError as exc:
        raise TraceError(f"cell does not compile: {exc}") from exc
    assigned = {
        s.get_name()
        for s in table.get_symbols()
        if s.is_assigned() or s.is_imported() or s.is_namespace()
    }
    loads = {s.get_name() for s in table.get_symbols() if s.is_referenced()}
    loads |= _global_loads(table)
    missing = loads - known - assigned - set(dir(builtins)) - set(injected)
    require(not missing, "missing namespace definitions: " + ", ".join(sorted(missing)))
    return known | assigned
