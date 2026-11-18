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

"""Clean and export recorded API planner turns for offline VLM training."""

import json
import mimetypes
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from rpent.flywheel.flywheel import has_confirmed_environment_success
from rpent.flywheel.reader import iter_jsonl, iter_tasks
from rpent.flywheel.vlm import VLMMessage, VLMSample
from rpent.flywheel.vlm_api import TurnConversionError, api_turn_to_sample
from rpent.planner.utils.sdk import json_object


def _tool_data(message: VLMMessage) -> dict:
    """Read the JSON object from the first text block of a tool result."""
    for block in message.content:
        if block["type"] == "text":
            value = block["text"]
            if value.startswith("["):
                try:
                    items = json.loads(value)
                except ValueError:
                    return {}
                if isinstance(items, list) and len(items) == 1:
                    value = items[0]
            return json_object(value) or {}
    return {}


def _image_groups(sample: VLMSample, prior_calls: dict[str, dict]) -> dict:
    """Map image positions to known observations for history filtering."""
    calls = {}
    reads = {}
    groups = {}
    scope = 0
    latest_step = -1
    for index, message in enumerate(sample.messages):
        calls.update((call["id"], call) for call in message.tool_calls)
        call_id = message.tool_call_id
        call = calls.get(call_id, {})
        name = call.get("name")
        group = (scope, None, None)
        native_paths = set()
        images = [
            (offset, b)
            for offset, b in enumerate(message.content)
            if b["type"] == "image"
        ]
        if (
            message.role == "tool"
            and name
            and name not in {"back_project", "query_world_map"}
        ):
            record = prior_calls.get(call_id, {})
            raw = (record.get("raw_result") or {}).get("data") or {}
            data = _tool_data(message)
            if name == "reset" and data and "error" not in data:
                scope += 1
                latest_step = -1
                group = (scope, None, None)
            step = data.get("step")
            if type(step) is not int or step < 0:
                step = None
            state_return = name == "view_env_state"
            if (
                name != "view_env_state"
                and isinstance(raw, dict)
                and "images" in raw
                and "state" in data
            ):
                native = raw["images"]
                if all(
                    isinstance(i, dict) and i.get("$type") == "artifact_ref"
                    for i in native
                ):
                    state_return = True
                    native_paths = {i["path"] for i in native}
            if state_return or name == "read_image":
                requested = call["arguments"].get("step", -1)
                explicit = (
                    name in {"view_env_state", "read_image"}
                    and isinstance(requested, int)
                    and requested >= 0
                )
                old_read = explicit and (
                    step is None
                    or step < latest_step
                    or (name == "read_image" and latest_step < 0)
                )
                if not explicit and step is not None and step < latest_step:
                    scope += 1
                    latest_step = -1
                if old_read:
                    group = None
                else:
                    observation = data.get("observation_id")
                    if not explicit and state_return:
                        observation = observation or record.get("after_observation_id")
                    if step is not None:
                        latest_step = max(latest_step, step)
                    group = (scope, observation, step)
                if name == "read_image":
                    reads[call_id] = group
        for offset, block in images:
            if block["artifact"].path in native_paths:
                continue
            source = block.get("source_call_id")
            association = reads.get(source) if source is not None else group
            if association is None:
                continue
            run, observation, step = association
            observation = block["observation_id"] or observation
            if observation is not None or step is not None:
                groups[index, offset] = (run, observation, step)
    return groups


def filter_vlm_sample(
    sample: VLMSample,
    *,
    image_history: int = 2,
    reasoning: str = "drop",
    prior_calls: dict[str, dict] | None = None,
) -> VLMSample:
    """Keep recent observation images and optionally remove marked assistant thinking."""
    if image_history < 0 or reasoning not in {"drop", "keep"}:
        raise ValueError("Expected image_history >= 0 and reasoning 'drop' or 'keep'")
    groups = _image_groups(sample, prior_calls or {})
    ids = defaultdict(set)
    for run, observation, step in groups.values():
        if observation is not None and step is not None:
            ids[run, step].add(observation)

    keys = {}
    for position, (run, observation, step) in groups.items():
        if observation is None and (aliases := ids.get((run, step))):
            if len(aliases) != 1:
                continue
            observation = next(iter(aliases))
        keys[position] = (run, observation if observation is not None else step)
    newest = list(dict.fromkeys(reversed(keys.values())))
    keep = set(newest[:image_history])
    messages = []
    for index, message in enumerate(sample.messages):
        content = []
        for offset, block in enumerate(message.content):
            key = keys.get((index, offset))
            if key is not None and key not in keep:
                continue
            if block["type"] == "image" and key is not None and isinstance(key[1], str):
                block = {**block, "observation_id": key[1]}
            content.append(block)
        if not content and message.content and message.role == "user":
            continue
        messages.append(
            replace(
                message,
                content=content,
                reasoning=None
                if message.role == "assistant" and reasoning == "drop"
                else message.reasoning,
            )
        )
    target = (
        sample.target if reasoning == "keep" else replace(sample.target, reasoning=None)
    )
    return replace(sample, messages=messages, target=target)


def _copy_images(
    row: dict,
    root: Path,
    output: Path,
    source_paths: dict[str, str],
    images_by_digest: dict[str, str],
) -> list[str]:
    """Store each distinct image once and replace source paths with dataset paths."""
    artifacts = [
        block["artifact"]
        for message in row["messages"]
        for block in message["content"]
        if block["type"] == "image"
    ]
    paths = {artifact["path"] for artifact in artifacts}
    missing = sorted(
        path
        for path in paths
        if path not in source_paths and not (root / path).is_file()
    )
    if missing:
        return missing
    for artifact in artifacts:
        path = artifact["path"]
        if path not in source_paths:
            source = root / path
            if not source.resolve().is_relative_to(root / "artifacts"):
                raise ValueError(f"Image artifact outside trajectory: {source}")
            data = source.read_bytes()
            digest = sha256(data).hexdigest()
            if digest not in images_by_digest:
                suffix = mimetypes.guess_extension(artifact["media_type"]) or ".bin"
                target = f"images/{digest}{suffix}"
                (output / target).write_bytes(data)
                images_by_digest[digest] = target
            source_paths[path] = images_by_digest[digest]
        artifact["path"] = source_paths[path]
    return []


def export_vlm_samples(
    input_root: Path | str,
    output_dir: Path | str,
    *,
    pattern: str = "*",
    image_history: int = 2,
    reasoning: str = "drop",
    include_unsuccessful: bool = False,
) -> dict[str, Any]:
    """Export selected episodes to one JSONL file with portable image references."""
    if image_history < 0 or reasoning not in {"drop", "keep"}:
        raise ValueError("Expected image_history >= 0 and reasoning 'drop' or 'keep'")
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Output directory already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "image_history": image_history,
        "reasoning": reasoning,
        "include_unsuccessful": include_unsuccessful,
        "episode_count": 0,
        "selected_episode_count": 0,
        "exported_episode_count": 0,
        "turn_count": 0,
        "sample_count": 0,
        "skipped_backend_counts": Counter(),
        "excluded": [],
    }
    with TemporaryDirectory(
        prefix=f".{destination.name}-", dir=destination.parent
    ) as tmp:
        output = Path(tmp)
        (output / "images").mkdir()
        images_by_digest = {}
        with (output / "samples.jsonl").open("w", encoding="utf-8") as rows:
            for root, task in iter_tasks(input_root, pattern):
                selected = include_unsuccessful or has_confirmed_environment_success(
                    task
                )
                report["episode_count"] += 1
                if selected:
                    report["selected_episode_count"] += 1
                samples_before = report["sample_count"]
                prior_calls = {}
                source_paths = {}
                for planner in iter_jsonl(root / "planners.jsonl"):
                    report["turn_count"] += 1
                    source = {
                        "episode_id": task["task_id"],
                        "turn_id": planner["planner_id"],
                    }
                    backend = planner["backend"]
                    if backend != "api":
                        report["skipped_backend_counts"][backend] += 1
                    if not selected:
                        report["excluded"].append(
                            {**source, "reason": "no_confirmed_environment_success"}
                        )
                        continue
                    if backend != "api":
                        report["excluded"].append(
                            {
                                **source,
                                "reason": "unsupported_backend",
                                "backend": backend,
                            }
                        )
                        continue
                    try:
                        sample = api_turn_to_sample(
                            task, planner, prior_calls=prior_calls
                        )
                        sample = filter_vlm_sample(
                            sample,
                            image_history=image_history,
                            reasoning=reasoning,
                            prior_calls=prior_calls,
                        )
                    except (ValueError, KeyError) as exc:
                        report["excluded"].append(
                            {
                                **source,
                                "reason": exc.reason
                                if isinstance(exc, TurnConversionError)
                                else "invalid_turn",
                                "detail": str(exc),
                            }
                        )
                        continue
                    finally:
                        for call in planner.get("tool_calls", []):
                            sdk_id = call.get("source_ids", {}).get(
                                "pydantic_ai.tool_call_id"
                            )
                            if sdk_id:
                                prior_calls[sdk_id] = (
                                    {} if sdk_id in prior_calls else call
                                )
                    row = asdict(sample)
                    missing = _copy_images(
                        row, root, output, source_paths, images_by_digest
                    )
                    if missing:
                        report["excluded"].append(
                            {
                                **source,
                                "reason": "missing_images",
                                "paths": missing,
                            }
                        )
                        continue
                    rows.write(
                        json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                    )
                    report["sample_count"] += 1
                if report["sample_count"] > samples_before:
                    report["exported_episode_count"] += 1
        report["image_count"] = len(images_by_digest)
        report["excluded_episode_counts"] = {
            "no_confirmed_environment_success": report["episode_count"]
            - report["selected_episode_count"]
        }
        counts = Counter(item["reason"] for item in report["excluded"])
        report["excluded_turn_counts"] = {
            reason: counts[reason]
            for reason in (
                "no_confirmed_environment_success",
                "unsupported_backend",
                "incomplete_response",
                "invalid_tool_arguments",
                "missing_images",
                "invalid_turn",
            )
        }
        (output / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        output.rename(destination)
    return report
