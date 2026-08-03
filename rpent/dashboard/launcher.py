"""Small adapter between the dashboard launch form and CLI args."""
from __future__ import annotations

from typing import Any


def defaults_from_args(args: Any) -> dict[str, Any]:
    """Build the fixed Claude Code Session form defaults."""
    return {
        "planner": "claude_code",
        "model": args.model,
        "cuda-device": args.cuda_device,
        "max-turns": args.max_turns,
        "max-episode-steps": args.max_episode_steps,
        "planner-timeout-s": args.planner_timeout_s,
        "claude-code-max-budget-usd": args.claude_code_max_budget_usd,
    }


def apply_to_args(args: Any, payload: dict[str, Any]) -> None:
    """Apply the fixed Claude Code Session form values."""
    args.planner = "claude_code"
    args.model = payload.get("model") or None
    args.cuda_device = payload.get("cuda-device") or None
    args.max_turns = int(payload["max-turns"])
    args.max_episode_steps = int(payload["max-episode-steps"])
    timeout = payload.get("planner-timeout-s")
    args.planner_timeout_s = None if timeout in ("", None) else int(timeout)
    budget = payload.get("claude-code-max-budget-usd")
    args.claude_code_max_budget_usd = (
        None if budget in ("", None) else float(budget)
    )
