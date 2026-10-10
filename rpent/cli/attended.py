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

"""CLI orchestration for one operator-attended attempt."""

import json
import shutil
from contextlib import ExitStack
from pathlib import Path

from rpent.cli.tui import next_user_line
from rpent.evaluation.result import record_operator_outcome
from rpent.session.transcript import _serialize_messages
from rpent.utils.logging import get_logger

logger = get_logger("agent")


def _finalize_operator_verdict(
    result, *, toolkit, memory, recipe_tag, output_dir, state_output_dir, session_number
):
    """Archive the interrupted attempt and prepare its next-session handoff."""
    finish = toolkit.finalize_direct_verdict()
    error = result.error
    if error:
        logger.info("Planner stopped after operator verdict: %s", error)
        result.stats["planner_error_at_operator_verdict"] = error
    verdict = finish["operator_verdict"]
    handoff = ""
    if verdict in {"success", "failure"}:
        evidence = record_operator_outcome(
            memory_root=memory.root,
            recipe_tag=recipe_tag,
            run_name=Path(output_dir).name,
            session_number=session_number,
            state_output_dir=state_output_dir,
            verdict=finish,
            messages=_serialize_messages(result.messages),
            planner_error=error,
        )
        handoff = (
            f"\nThe operator marked the preceding attempt as {verdict}. "
            f"Before requesting any reset or motion, read {evidence}, "
            "inspect its evidence and write a grounded outcome summary "
            "to your memory inbox wip notes. Distinguish observations "
            "from hypotheses; never label a failed attempt as successful. "
            "Then request_scene_reset and wait for a new operator "
            "confirmation before attempting the task again.\n"
        )
        if verdict == "failure":
            error = None
    return finish, error, handoff


def _publish_success_recipe(
    memory, *, recipe_tag, output_dir, state_output_dir, auto_merge
):
    """Preserve successful-session artifacts before another session overwrites them."""
    for name in (f"{recipe_tag}.json", f"{recipe_tag}_recipe.jsonl"):
        shutil.copy2(Path(output_dir) / name, Path(state_output_dir) / name)
    if auto_merge:
        memory.merge_memory(
            cell_tag=recipe_tag, run_state_dir=state_output_dir, solved=True
        )


def _solve_attended(
    planner, *, operator_input, state_output_dir, keep_mcp_alive=False, **kwargs
):
    """Own the transport for the whole attended attempt, including human waits."""
    with ExitStack() as cleanup:
        if keep_mcp_alive:
            from rpent.planner.utils.http_mcp_server import HttpMcpServer

            server = HttpMcpServer(kwargs["toolkit"])
            kwargs["mcp_server"] = server
            cleanup.callback(server.stop)
            # Release pending tool handlers before shutting down HTTP.
            cleanup.callback(kwargs["toolkit"].cancel_active_and_wait)
            server.start()
        return _solve_attended_turns(
            planner,
            operator_input=operator_input,
            state_output_dir=state_output_dir,
            **kwargs,
        )


def _solve_attended_turns(planner, *, operator_input, state_output_dir, **kwargs):
    """Keep an unjudged attempt alive across planner replies and human pauses."""
    toolkit = kwargs["toolkit"]
    original_message = kwargs["user_message"].removesuffix(
        operator_input.feedback_prompt()
    )
    messages = []
    usage = {}
    turn_number = 0
    while True:
        continue_revision = operator_input.continue_revision
        result = planner.solve(**kwargs)
        turn_number += 1
        messages.extend(result.messages)
        for key in (
            "total_input_tokens",
            "total_output_tokens",
            "total_cached_input_tokens",
            "tool_calls",
        ):
            usage[key] = usage.get(key, 0) + result.stats.get(key, 0)
        # Preserve every reply before the planner reuses its output filenames.
        archive = Path(state_output_dir) / f"dialogue_{turn_number:03d}.json"
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_text(
            json.dumps(
                {
                    "messages": _serialize_messages(result.messages),
                    "stats": result.stats,
                    "finish": result.finish_result,
                    "error": result.error,
                },
                indent=2,
                default=str,
            )
        )
        if result.error or result.finish_result or toolkit.direct_verdict_requested:
            break
        if operator_input.closed:
            toolkit.request_direct_verdict("abort", "Interactive input closed")
            break
        resumed_operator_request = operator_input.pending_kind is not None
        if resumed_operator_request:
            logger.info(
                "Waiting for the current operator request; keeping this attempt alive."
            )
        toolkit.wait_active()
        if toolkit.direct_verdict_requested:
            break
        if operator_input.closed:
            toolkit.request_direct_verdict("abort", "Interactive input closed")
            break
        resumed_operator_request |= (
            operator_input.continue_revision != continue_revision
        )
        if resumed_operator_request:
            reply = (
                "The pending operator request has returned or the operator already answered continue. "
                "Read its recorded result and continue this attempt without waiting for the same reply again."
            )
        else:
            logger.info(
                "Waiting for feedback or /continue; /success, /failure and /abort remain available."
            )
            reply = next_user_line(kwargs["input_queue"])
            if toolkit.direct_verdict_requested:
                break
            if reply is None:
                toolkit.request_direct_verdict("abort", "Interactive input closed")
                break
        if not resumed_operator_request:
            observation = toolkit.execute_tool("view_env_state", {}).data
            if observation.get("error"):
                result.error = (
                    f"Observation refresh failed after feedback: {observation['error']}"
                )
                break
        messages.append({"role": "user", "content": reply})
        kwargs["user_message"] = (
            original_message
            + "\nContinue the SAME attempt, not a new exploration session. Robot state and "
            "scene confirmation are retained; do not reset merely because the conversation resumed. "
            f"Current attempt state and operator results are recorded under {state_output_dir}. "
            "If the operator changed the scene, obtain fresh observations and localization. "
            "Previous conversation:\n"
            + json.dumps(_serialize_messages(messages), ensure_ascii=False, default=str)
            + operator_input.feedback_prompt()
        )
    result.messages = messages
    result.stats.update(usage)
    return result
