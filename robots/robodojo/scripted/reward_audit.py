# Copyright 2026 The RPent Authors.
"""AST reward-boundary lint for trusted recipes, not a Python security sandbox.

The runtime RPC and payload allowlists enforce the bridge boundary. Static
inspection cannot prove arbitrary Python harmless; review recipes before freeze.
"""

import ast
from pathlib import Path
from typing import Any

DENIED = {
    "reward",
    "score",
    "is_success",
    "get_score",
    "get_reward_details",
    "official_success",
    "reward_progress",
    "completed_stages",
    "process_score",
    "official_episode_score",
    "official_episode_score_100",
    "success",
    "score_frac",
    "progress_proxy",
    "horizon",
    "evaluation",
    "score_source",
    "completed_predicates",
    "target_completed_stages",
    "reward_predicate_truth",
    "bottles_on_bin_bottom",
    "bottles_on_bin_bottom_count",
    "grippers_open",
    "arms_home",
}
IMPORTS = {
    "__future__",
    "argparse",
    "base64",
    "json",
    "math",
    "sys",
    "time",
    "pathlib",
    "numpy",
    "PIL",
}


def strict_audit(path: Path) -> dict[str, Any]:
    """Reject privileged symbols, foreign imports and reward-stage prompt lists."""
    findings = []
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except SyntaxError as exc:
        return {"passed": False, "findings": [{"line": exc.lineno, "error": str(exc)}]}
    for node in ast.walk(tree):
        name = (
            node.attr
            if isinstance(node, ast.Attribute)
            else node.id
            if isinstance(node, ast.Name)
            else node.arg
            if isinstance(node, ast.keyword)
            else node.value
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
            else None
        )
        if name in DENIED or name in {"eval", "exec", "__import__"}:
            findings.append({"line": node.lineno, "symbol": name})
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = (
                [n.name for n in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            if (isinstance(node, ast.ImportFrom) and node.level) or any(
                m.split(".")[0] not in IMPORTS for m in modules
            ):
                findings.append({"line": node.lineno, "imports": modules})
        if isinstance(node, ast.List):
            keys = [
                {k.value for k in item.keys if isinstance(k, ast.Constant)}
                for item in node.elts
                if isinstance(item, ast.Dict)
            ]
            if (
                sum(
                    "prompt" in k
                    and bool(k & {"stage", "target", "target_completed_stages"})
                    for k in keys
                )
                >= 2
            ):
                findings.append({"line": node.lineno, "symbol": "prompt+stage list"})
    return {"passed": not findings, "findings": findings}


def check_payload(obj: Any, path: str = "root") -> None:
    """Reject nested privileged fields before delivery to a worker."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in DENIED:
                raise RuntimeError(f"privileged field at {path}.{key}")
            check_payload(value, f"{path}.{key}")
    elif isinstance(obj, (tuple, list)):
        for item in obj:
            check_payload(item, path)
