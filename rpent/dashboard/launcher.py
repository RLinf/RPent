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

"""Small adapter between the dashboard launch form and CLI args."""

from __future__ import annotations

from typing import Any


def defaults_from_args(args: Any) -> dict[str, Any]:
    """Build form defaults for the selected planner."""
    return {
        "planner": args.planner,
        "model": args.model,
        "cuda-device": args.cuda_device,
        "max-turns": args.max_turns,
        "max-episode-steps": args.max_episode_steps,
        "planner-timeout-s": args.planner_timeout_s,
        "reasoning-effort": args.reasoning_effort,
        "claude-code-max-budget-usd": args.claude_code_max_budget_usd,
        "no-images": args.no_images,
    }


def apply_to_args(args: Any, payload: dict[str, Any]) -> None:
    """Apply values submitted by the Dashboard launch form."""
    args.planner = payload["planner"]
    args.model = payload.get("model") or None
    args.cuda_device = payload.get("cuda-device") or None
    args.max_turns = int(payload["max-turns"])
    args.max_episode_steps = int(payload["max-episode-steps"])
    timeout = payload.get("planner-timeout-s")
    args.planner_timeout_s = None if timeout in ("", None) else int(timeout)
    args.reasoning_effort = payload.get("reasoning-effort", "none")
    args.no_images = bool(payload.get("no-images", False))
    if args.planner == "claude_code":
        budget = payload.get("claude-code-max-budget-usd")
        args.claude_code_max_budget_usd = (
            None if budget in ("", None) else float(budget)
        )
