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

"""Standalone async rollout and evaluation runner over any OpenAI/SGLang endpoint.

Uses the exact same ``EpisodeConversation`` + ``SlidingWindowCompaction`` wire
format and tool-validation loop as RLinf training so standalone evaluation and
data collection stay 100% consistent with RL rollouts.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

import httpx

from robots.rl.conversation import CompactionConfig, EpisodeConversation
from robots.rl.server import BINARY_EXECUTE_MEDIA_TYPE, unpack_execute_response
from rpent.utils.logging import get_logger

logger = get_logger("rl_rollout")


async def run_episode(
    *,
    model_url: str,
    model_name: str,
    sim_url: str,
    task: dict[str, Any],
    max_turns: int = 50,
    max_tokens: int = 16384,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 20,
    thinking: bool = True,
    compaction: CompactionConfig | None = None,
    image_size: int = 1024,
    max_actions_per_call: int = 3,
    record_artifacts: bool = False,
    output_dir: Path | None = None,
    request_timeout_s: float = 600.0,
) -> dict[str, Any]:
    """Run one standalone episode against ``model_url`` (OpenAI chat) and ``sim_url``."""
    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=request_timeout_s, trust_env=False) as client:
        base_seed = task.get("seed", task.get("init_state_id"))
        seeds = [base_seed, *(task.get("fallback_seeds") or ())]
        opened = None
        last_exc: Exception | None = None
        for seed in seeds:
            attempt_task = dict(task)
            if seed is not None:
                attempt_task["seed"] = seed
                if "init_state_id" in attempt_task:
                    attempt_task["init_state_id"] = seed
            try:
                open_resp = await client.post(
                    f"{sim_url.rstrip('/')}/sessions/open",
                    json={
                        "task": attempt_task,
                        "image_size": image_size,
                        "max_actions_per_call": max_actions_per_call,
                        "reasoning": thinking,
                        "record_artifacts": record_artifacts,
                        "output_dir": str(output_dir) if output_dir else None,
                    },
                )
                open_resp.raise_for_status()
                opened = open_resp.json()
                break
            except (httpx.HTTPError, OSError) as exc:
                last_exc = exc
        if opened is None:
            assert last_exc is not None
            raise last_exc
        sid = opened["session_id"]

        conv = EpisodeConversation(
            system_prompt=opened["system_prompt"],
            user_message=opened["user_message"],
            initial_observation_text=opened["initial_observation"],
            tools=opened["tools"],
            environment=opened["environment"],
            compaction=compaction,
        )
        done = bool(opened["done"])
        success = bool(opened["success"])
        truncated = bool(opened["truncated"])
        reason = "success" if success else "planner_completed"
        error_kind = None
        error_msg = None
        turns = 0

        try:
            while not done and turns < max_turns:
                conv.maybe_compact()
                payload: dict[str, Any] = {
                    "model": model_name,
                    "messages": conv.wire_messages(inline_base64=True),
                    "tools": conv.tools,
                    "parallel_tool_calls": False,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "chat_template_kwargs": {
                        "enable_thinking": thinking,
                        "preserve_thinking": True,
                    },
                }
                resp = await client.post(
                    f"{model_url.rstrip('/')}/chat/completions", json=payload
                )
                resp.raise_for_status()
                choice = resp.json()["choices"][0]
                turns += 1
                msg = choice["message"]
                finish = choice.get("finish_reason")
                tool_calls = msg.get("tool_calls") or []
                conv.append_assistant(
                    content=msg.get("content"),
                    reasoning_content=msg.get("reasoning_content"),
                    tool_calls=tool_calls or None,
                )
                if finish == "length":
                    error_kind, reason = "response_truncated", "response_truncated"
                    error_msg = "Token limit reached before completing response"
                    break
                if not tool_calls:
                    reason = "planner_completed"
                    break

                parsed_calls = [
                    {
                        "id": tc["id"],
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"],
                    }
                    for tc in tool_calls
                ]
                err_kind, err_msg, args, should_retry = conv.validate_or_retry(
                    parsed_calls
                )
                if err_kind is not None:
                    if should_retry:
                        continue
                    error_kind, error_msg, reason = err_kind, err_msg, err_kind
                    break

                exec_resp = await client.post(
                    f"{sim_url.rstrip('/')}/sessions/{sid}/execute",
                    json={"name": parsed_calls[0]["name"], "arguments": args},
                    headers={"Accept": BINARY_EXECUTE_MEDIA_TYPE},
                )
                exec_resp.raise_for_status()
                if BINARY_EXECUTE_MEDIA_TYPE in exec_resp.headers.get(
                    "content-type", ""
                ):
                    executed, images = unpack_execute_response(exec_resp.content)
                else:
                    executed = exec_resp.json()
                    images = []
                conv.append_tool_result(
                    tool_name=parsed_calls[0]["name"],
                    tool_call_id=parsed_calls[0]["id"],
                    text=executed["text"],
                    images=images,
                )
                success = bool(executed["success"])
                truncated = bool(executed["truncated"])
                done = bool(executed["done"])
                if executed.get("infrastructure_error"):
                    error_kind, reason = "tool_transport", "infrastructure_error"
                    error_msg = str(executed["infrastructure_error"])
                    break
            else:
                if not done and turns >= max_turns:
                    reason = "request_limit"
                elif success:
                    reason = "success"
                elif truncated:
                    reason = "environment_step_limit"
        finally:
            await client.post(f"{sim_url.rstrip('/')}/sessions/{sid}/close", json={})

    outcome = {
        "task": task,
        "init_state_id": opened.get("init_state_id"),
        "success": success,
        "truncated": truncated,
        "turns": turns,
        "termination_reason": "success" if success else reason,
        "error_kind": error_kind,
        "error": error_msg,
        "elapsed_s": round(time.perf_counter() - started, 2),
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "outcome.json").write_text(json.dumps(outcome, indent=2) + "\n")
    return outcome


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-url", required=True, help="OpenAI-compatible /v1 URL")
    parser.add_argument("--model-name", default="policy")
    parser.add_argument("--sim-url", required=True, help="SimPoolServer HTTP base URL")
    parser.add_argument("--tasks-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-turns", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--record-artifacts", action="store_true")
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.tasks_jsonl.read_text().splitlines()
        if line.strip()
    ]
    if args.limit is not None:
        rows = rows[: args.limit]
    compaction = CompactionConfig()
    sem = asyncio.Semaphore(args.concurrency)

    async def _run_all() -> list[dict[str, Any]]:
        async def _one(idx: int, row: dict[str, Any]) -> dict[str, Any]:
            task = (
                json.loads(row["answer"]) if isinstance(row.get("answer"), str) else row
            )
            async with sem:
                return await run_episode(
                    model_url=args.model_url,
                    model_name=args.model_name,
                    sim_url=args.sim_url,
                    task=task,
                    max_turns=args.max_turns,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    compaction=compaction,
                    record_artifacts=args.record_artifacts,
                    output_dir=args.output_dir / f"ep_{idx:04d}",
                )

        return await asyncio.gather(*(_one(i, r) for i, r in enumerate(rows)))

    results = asyncio.run(_run_all())
    solved = sum(bool(r["success"]) for r in results)
    summary = {
        "episodes": len(results),
        "solved": solved,
        "success_rate": solved / max(len(results), 1),
        "results": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    logger.info(
        "Standalone rollout complete: %d/%d solved (%.1f%%)",
        solved,
        len(results),
        100.0 * summary["success_rate"],
    )


if __name__ == "__main__":
    main()
