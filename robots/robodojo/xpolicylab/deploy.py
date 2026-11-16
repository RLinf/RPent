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

"""Official evaluator entry: execute RPent actions and retain native scoring."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from rpent.evaluation import write_json_atomic


def eval_one_episode(TASK_ENV: Any, model_client: Any) -> None:
    """Run one episode; the official run_eval owns aggregate results and videos."""
    context = {
        "task": TASK_ENV.task_name,
        "seed": int(TASK_ENV.eval_seed),
        "layout": int(TASK_ENV.env_seeds[0]),
    }
    output = Path(TASK_ENV.save_dir) / "rpent" / f"layout_{context['layout']}"
    output.mkdir(parents=True, exist_ok=False)
    ended = False

    def feedback() -> None:
        obs = TASK_ENV.get_obs()
        obs["rpent_episode"] = context
        obs["episode_status"] = {
            "step": int(TASK_ENV.take_action_cnt[0]),
            "step_limit": int(TASK_ENV.step_lim),
            "ended": ended,
        }
        if ended:
            obs["episode_status"]["success"] = bool(TASK_ENV.success[0])
        model_client.call(func_name="update_obs", obs=obs)

    try:
        ended = TASK_ENV.is_episode_end()
        # The official environment calls model.reset() at its episode boundary.
        feedback()
        report = model_client.call(func_name="episode_report")
        deadline = time.monotonic() + report["timeout_s"] + 60
        with (output / "actions.jsonl").open("w") as log:
            while True:
                if time.monotonic() > deadline:
                    raise TimeoutError("RPent episode wall-clock limit exceeded")
                actions = model_client.call(func_name="get_action")
                if not actions:
                    report = model_client.call(func_name="episode_report")
                    if report["status"] in {"completed", "failed"}:
                        break
                    continue
                if len(actions) != 1:
                    raise ValueError("RPent requires one acknowledged action at a time")
                action = actions[0]
                if action.get("_rpent_capture") is True:
                    feedback()
                    continue
                if ended:
                    raise RuntimeError(
                        "Motion requested after native episode termination"
                    )
                before = int(TASK_ENV.take_action_cnt[0])
                TASK_ENV.take_action(action)
                ended = TASK_ENV.is_episode_end()
                after = int(TASK_ENV.take_action_cnt[0])
                if after != before + 1:
                    raise RuntimeError("Official action counter did not advance by one")
                log.write(
                    json.dumps(
                        {
                            "step_before": before,
                            "step_after": after,
                            "action": {
                                k: [float(x) for x in v] for k, v in action.items()
                            },
                        }
                    )
                    + "\n"
                )
                log.flush()
                feedback()
        write_json_atomic(output / "planner-report.json", report)
        if report["status"] != "completed":
            raise RuntimeError(report.get("error") or "Planner did not finish")
        if not ended:
            # The official final check still decides success for an early finish.
            TASK_ENV.success[0] = False
            TASK_ENV.is_episode_end()
        score = 0.0
        if hasattr(TASK_ENV, "get_score"):
            TASK_ENV.get_score()
            score = float(TASK_ENV.reward_manager.get_score()[0])
        success = bool(TASK_ENV.success[0])
        if success:
            score = 100.0
        write_json_atomic(
            output / "native-result.json",
            {
                **context,
                "success": success,
                "score": score,
                "step": int(TASK_ENV.take_action_cnt[0]),
                "step_limit": int(TASK_ENV.step_lim),
            },
        )
    except Exception as exc:
        write_json_atomic(
            output / "infrastructure-error.json", {"error": str(exc), **context}
        )
        try:
            model_client.call(func_name="abort")
        finally:
            model_client.close()
            TASK_ENV.close()
        # Do not let the official reset-retry loop count an infrastructure error.
        raise SystemExit(2) from exc
