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

"""Validate, aggregate API transcripts, and safely update leaderboard results."""

from __future__ import annotations

import argparse
import copy
import difflib
import json
import math
import os
import re
import sys
import tempfile
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "docs/_static/leaderboard/results.json"
ROBOTWIN_MANIFEST_PATH = REPO_ROOT / "robots/robotwin/eval/demo_randomized.json"
ROBOCASA_MANIFEST_PATH = REPO_ROOT / "robots/robocasa/eval/target50.json"

_RATE_PATTERN = re.compile(r"\A\d{1,3}\.\d{2}\Z")
_JSON_TOKEN_PATTERN = re.compile(r'"(?:\\.|[^"\\])*"|[{}\[\]:,]|[^\s{}\[\]:,]+')
_COST_NUMERIC_FIELDS = (
    "mean_elapsed_seconds",
    "mean_total_tokens",
    "total_output_tokens",
)
_ROBOCASA_SPLIT_TO_VIEW = (
    ("atomic", "robocasa-atomic-seen"),
    ("composite_seen", "robocasa-composite-seen"),
    ("composite_unseen", "robocasa-composite-unseen"),
)


class LeaderboardValidationError(ValueError):
    """Raised when leaderboard data fails schema or consistency validation."""


class _Dataset(dict):
    """Dict subclass carrying original inline/multiline array formatting styles."""

    _array_styles: dict[tuple[str, ...], bool] | None = None


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_nonnegative_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value) and value >= 0


def _is_nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _round_two_decimals(
    numerator: int | float | Decimal,
    denominator: int = 1,
    scale: int = 1,
) -> Decimal:
    value = (Decimal(str(numerator)) * scale) / Decimal(denominator)
    return value.quantize(Decimal("0.01"), ROUND_HALF_UP)


def _index_by_id(items: Any, kind: str) -> dict[str, dict[str, Any]]:
    if not isinstance(items, list):
        raise LeaderboardValidationError(f"Leaderboard: {kind} must be a list")
    indexed: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict) or not _is_nonempty_str(item.get("id")):
            raise LeaderboardValidationError(f"Leaderboard: invalid id in {kind}")
        item_id = item["id"]
        if item_id in indexed:
            raise LeaderboardValidationError(
                f"Leaderboard: duplicate ID {item_id!r} in {kind}"
            )
        indexed[item_id] = item
    return indexed


def _validate_bilingual(
    value: Any,
    context: str,
    *,
    required: bool = True,
) -> None:
    if value is None and not required:
        return
    if not isinstance(value, dict):
        if required or not _is_nonempty_str(value):
            raise LeaderboardValidationError(f"Leaderboard: invalid {context}")
        return
    for language in ("en", "zh"):
        if not _is_nonempty_str(value.get(language)):
            raise LeaderboardValidationError(
                f"Leaderboard: missing or empty {language!r} text in {context}"
            )


def _validate_references(
    identifiers: Any,
    pool: dict[str, Any],
    context: str,
    *,
    allow_empty: bool = False,
) -> None:
    if not isinstance(identifiers, list):
        raise LeaderboardValidationError(f"Leaderboard: {context} must be a list")
    if not allow_empty and not identifiers:
        raise LeaderboardValidationError(f"Leaderboard: {context} must not be empty")
    for identifier in identifiers:
        if not isinstance(identifier, str) or identifier not in pool:
            raise LeaderboardValidationError(
                f"Leaderboard: unknown reference {identifier!r} in {context}"
            )


def validate_data(data: dict[str, Any]) -> None:
    """Validate leaderboard dataset schema, references, and numeric invariants."""
    if not isinstance(data, dict):
        raise LeaderboardValidationError(
            "Leaderboard: top-level JSON must be an object"
        )
    schema_version = data.get("schema_version")
    if not (_is_int(schema_version) and schema_version == 4):
        raise LeaderboardValidationError(
            f"Leaderboard: expected schema_version 4, got {schema_version!r}"
        )

    configs = _index_by_id(data.get("configurations"), "configurations")
    benchmarks = _index_by_id(data.get("benchmarks"), "benchmarks")

    all_views: list[dict[str, Any]] = []
    all_notes: list[dict[str, Any]] = []
    for benchmark_id, benchmark in benchmarks.items():
        if not _is_nonempty_str(benchmark.get("name")):
            raise LeaderboardValidationError(
                f"Leaderboard: missing name in benchmark {benchmark_id!r}"
            )
        _validate_bilingual(
            benchmark.get("heading"),
            f"benchmarks[{benchmark_id}].heading",
        )
        _validate_bilingual(
            benchmark.get("summary"),
            f"benchmarks[{benchmark_id}].summary",
            required=False,
        )
        benchmark_views = benchmark.get("views")
        benchmark_notes = benchmark.get("notes", [])
        if not isinstance(benchmark_views, list) or not benchmark_views:
            raise LeaderboardValidationError(
                f"Leaderboard: invalid views in benchmark {benchmark_id!r}"
            )
        if not isinstance(benchmark_notes, list):
            raise LeaderboardValidationError(
                f"Leaderboard: invalid notes in benchmark {benchmark_id!r}"
            )
        local_views = _index_by_id(benchmark_views, f"views[{benchmark_id}]")
        default_view = benchmark.get("default_view")
        if not isinstance(default_view, str) or default_view not in local_views:
            raise LeaderboardValidationError(
                f"Leaderboard: invalid default_view in benchmark {benchmark_id!r}"
            )
        for view in benchmark_views:
            _validate_bilingual(view.get("label"), f"views[{view['id']}].label")
        all_views.extend(benchmark_views)
        all_notes.extend(benchmark_notes)

    views = _index_by_id(all_views, "views")
    results = _index_by_id(data.get("results"), "results")
    cost_results = data.get("cost_results")
    if not isinstance(cost_results, dict):
        raise LeaderboardValidationError("Leaderboard: cost_results must be an object")
    _index_by_id(cost_results.get("rows"), "cost_results")
    sources = _index_by_id(data.get("sources"), "sources")

    footer_notes = data.get("footer_notes", [])
    if not isinstance(footer_notes, list):
        raise LeaderboardValidationError("Leaderboard: footer_notes must be a list")
    _index_by_id(all_notes + footer_notes, "notes")

    palette = data.get("palette")
    if not isinstance(palette, dict):
        raise LeaderboardValidationError("Leaderboard: palette must be an object")
    palette_roles = ("best", "rpent", "reference")
    for theme_name in ("light", "dark"):
        theme = palette.get(theme_name)
        if not isinstance(theme, dict) or not all(
            _is_nonempty_str(theme.get(role)) for role in palette_roles
        ):
            raise LeaderboardValidationError(
                f"Leaderboard: invalid palette theme {theme_name!r}"
            )

    for config_id, config in configs.items():
        kind = config.get("kind")
        if kind not in ("rpent", "external"):
            raise LeaderboardValidationError(
                f"Leaderboard: invalid kind {kind!r} in {config_id!r}"
            )
        label_fields = (
            ("display_name", "model", "backend")
            if kind == "rpent"
            else ("display_name", "model")
        )
        if not any(_is_nonempty_str(config.get(field)) for field in label_fields):
            raise LeaderboardValidationError(
                f"Leaderboard: missing display label in configuration {config_id!r}"
            )

    for note in all_notes + footer_notes:
        _validate_bilingual(note.get("text"), f"notes[{note['id']}].text")
    for note in all_notes:
        _validate_references(
            note.get("configuration_ids"),
            configs,
            f"notes[{note['id']}].configuration_ids",
            allow_empty=True,
        )
    _validate_references(
        cost_results.get("source_ids"),
        sources,
        "cost_results.source_ids",
    )

    collection_specs = (
        (data["results"], "view_id", views, None, True),
        (
            cost_results["rows"],
            "benchmark_id",
            benchmarks,
            cost_results["source_ids"],
            False,
        ),
    )
    for (
        collection,
        group_key,
        valid_groups,
        default_sources,
        is_result,
    ) in collection_specs:
        seen_pairs: set[tuple[str, str]] = set()
        for record in collection:
            record_id = record["id"]
            config_id = record.get("configuration_id")
            group_id = record.get(group_key)
            if not isinstance(config_id, str) or config_id not in configs:
                raise LeaderboardValidationError(
                    f"Leaderboard: unknown configuration_id in {record_id!r}"
                )
            if not isinstance(group_id, str) or group_id not in valid_groups:
                raise LeaderboardValidationError(
                    f"Leaderboard: unknown {group_key} in {record_id!r}"
                )
            pair = (group_id, config_id)
            if pair in seen_pairs:
                raise LeaderboardValidationError(
                    f"Leaderboard: duplicate record for {pair!r}"
                )
            seen_pairs.add(pair)

            record_sources = record.get("source_ids", default_sources)
            _validate_references(record_sources, sources, f"sources in {record_id!r}")
            if "derived_from" in record:
                _validate_references(
                    record["derived_from"],
                    results,
                    f"derived_from in {record_id!r}",
                    allow_empty=True,
                )

            for field_name in _COST_NUMERIC_FIELDS:
                metric_value = record.get(field_name)
                if metric_value is not None and not _is_nonnegative_number(
                    metric_value
                ):
                    raise LeaderboardValidationError(
                        f"Leaderboard: invalid {field_name} in {record_id!r}"
                    )

            rate = record.get("rate")
            if rate is not None:
                if not isinstance(rate, str) or _RATE_PATTERN.fullmatch(rate) is None:
                    raise LeaderboardValidationError(
                        f"Leaderboard: invalid rate {rate!r} in {record_id!r}"
                    )
                rate_decimal = Decimal(rate)
                if not (Decimal(0) <= rate_decimal <= Decimal(100)):
                    raise LeaderboardValidationError(
                        f"Leaderboard: rate out of range in {record_id!r}"
                    )

            if not is_result:
                if "mean_elapsed_seconds" not in record:
                    raise LeaderboardValidationError(
                        f"Leaderboard: missing mean_elapsed_seconds in {record_id!r}"
                    )
                continue

            if "rate" not in record:
                raise LeaderboardValidationError(
                    f"Leaderboard: missing rate key in {record_id!r}"
                )
            episodes = record.get("episodes")
            successes = record.get("successes")
            if episodes is not None and not (_is_int(episodes) and episodes >= 0):
                raise LeaderboardValidationError(
                    f"Leaderboard: invalid episodes in {record_id!r}"
                )
            if successes is not None and not (_is_int(successes) and successes >= 0):
                raise LeaderboardValidationError(
                    f"Leaderboard: invalid successes in {record_id!r}"
                )
            if _is_int(successes) and _is_int(episodes):
                if successes > episodes:
                    raise LeaderboardValidationError(
                        f"Leaderboard: successes > episodes in {record_id!r}"
                    )
                is_estimated = record.get("successes_estimated") is True
                is_derived = record.get("status") == "derived"
                skip_ratio_check = (
                    rate is None
                    or episodes == 0
                    or is_estimated
                    or is_derived
                    or group_id == "robocasa"
                )
                if not skip_ratio_check:
                    expected_rate = str(_round_two_decimals(successes, episodes, 100))
                    if rate != expected_rate:
                        raise LeaderboardValidationError(
                            f"Leaderboard: rate {rate!r} != expected "
                            f"{expected_rate!r} in {record_id!r}"
                        )
            _validate_bilingual(
                record.get("evaluation_note"),
                f"results[{record_id}].evaluation_note",
                required=False,
            )


def _scan_array_styles(raw_text: str) -> dict[tuple[str, ...], bool]:
    """Record whether each non-empty JSON array was formatted on a single line."""
    tokens = list(_JSON_TOKEN_PATTERN.finditer(raw_text))
    styles: dict[tuple[str, ...], bool] = {}

    def walk(index: int, path: tuple[str, ...]) -> int:
        token = tokens[index].group(0)
        if token == "{":
            index += 1
            while tokens[index].group(0) != "}":
                key = json.loads(tokens[index].group(0))
                index = walk(index + 2, (*path, key))
                if tokens[index].group(0) == ",":
                    index += 1
            return index + 1
        if token == "[":
            array_start = tokens[index].start()
            index += 1
            item_index = 0
            while tokens[index].group(0) != "]":
                is_id_object = (
                    tokens[index].group(0) == "{"
                    and index + 3 < len(tokens)
                    and tokens[index + 1].group(0) == '"id"'
                    and tokens[index + 3].group(0).startswith('"')
                )
                if is_id_object:
                    segment = f"id={json.loads(tokens[index + 3].group(0))}"
                else:
                    segment = str(item_index)
                index = walk(index, (*path, segment))
                item_index += 1
                if tokens[index].group(0) == ",":
                    index += 1
            if item_index > 0:
                styles[path] = "\n" not in raw_text[array_start : tokens[index].end()]
            return index + 1
        return index + 1

    if tokens:
        walk(0, ())
    return styles


def load_data(path: Path | str = DATA_PATH) -> dict[str, Any]:
    """Read, parse, and validate a leaderboard ``results.json`` file."""
    raw_text = Path(path).read_text(encoding="utf-8")
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise LeaderboardValidationError(f"Leaderboard: invalid JSON: {exc}") from exc
    validate_data(parsed)
    dataset = _Dataset(parsed)
    dataset._array_styles = _scan_array_styles(raw_text)
    return dataset


def dump_data(data: dict[str, Any]) -> str:
    """Serialize leaderboard data preserving compact row and array layout."""
    saved_styles = data._array_styles if isinstance(data, _Dataset) else None

    def compact_json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False)

    def format_value(value: Any, indent: int, path: tuple[str, ...]) -> str:
        pad = " " * indent
        child_pad = " " * (indent + 2)
        if isinstance(value, dict):
            if not value:
                return "{}"
            rows = [
                f"{child_pad}{compact_json(key)}: "
                f"{format_value(item, indent + 2, (*path, key))}"
                for key, item in value.items()
            ]
            return "{\n" + ",\n".join(rows) + f"\n{pad}}}"
        if isinstance(value, list):
            if not value:
                return "[]"
            if path in {("results",), ("cost_results", "rows")}:
                rows = [f"{child_pad}{compact_json(item)}" for item in value]
                return "[\n" + ",\n".join(rows) + f"\n{pad}]"

            all_primitives = all(not isinstance(item, (dict, list)) for item in value)
            is_inline_ref_list = (
                bool(path)
                and path[-1] in {"source_ids", "configuration_ids"}
                and all(isinstance(item, str) for item in value)
                and len(compact_json(value)) <= 88
            )
            if saved_styles is not None and path in saved_styles:
                inline = saved_styles[path]
            else:
                inline = is_inline_ref_list
            if all_primitives and inline:
                return compact_json(value)

            formatted_items: list[str] = []
            for item_index, item in enumerate(value):
                if isinstance(item, dict) and isinstance(item.get("id"), str):
                    segment = f"id={item['id']}"
                else:
                    segment = str(item_index)
                formatted_items.append(
                    child_pad + format_value(item, indent + 2, (*path, segment))
                )
            return "[\n" + ",\n".join(formatted_items) + f"\n{pad}]"
        return compact_json(value)

    return format_value(data, 0, ()) + "\n"


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            delete=False,
            newline="\n",
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except BaseException:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise


def write_data(path: Path | str, data: dict[str, Any]) -> None:
    """Validate and atomically write leaderboard data preserving layout."""
    validate_data(data)
    _atomic_write(Path(path), dump_data(data))


def load_transcript_final_record(path: Path | str) -> tuple[dict[str, Any], int]:
    """Read an append-mode ``transcript_*.json`` file.

    Returns ``(final_record, excluded_in_file_attempts)``, where the count
    reflects only earlier JSON records appended inside the same file (not
    externally quarantined retry directories).
    """
    raw_text = Path(path).read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    cursor = 0
    length = len(raw_text)
    records: list[dict[str, Any]] = []
    while cursor < length:
        while cursor < length and raw_text[cursor].isspace():
            cursor += 1
        if cursor >= length:
            break
        try:
            parsed, cursor = decoder.raw_decode(raw_text, cursor)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSON in {path}: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"Transcript entry in {path} must be an object")
        records.append(parsed)
    if not records:
        raise ValueError(f"Empty transcript file: {path}")
    return records[-1], len(records) - 1


def _classify_robotwin_termination(agent_error: str | None, success: bool) -> str:
    """Classify a RoboTwin episode outcome.

    Aligns with ``robots/robocasa/eval/result.py::_termination_reason`` while
    also matching ``episode timed out`` messages emitted by RoboTwin.
    """
    if success or not (agent_error and agent_error.strip()):
        return "completed"
    lowered = agent_error.strip().lower()
    if "timed out" in lowered and any(
        marker in lowered for marker in ("planner", "agent sdk", "codex sdk", "episode")
    ):
        return "planner_timeout"
    if "planner response truncated" in lowered:
        return "planner_response_truncated"
    return "infrastructure_error"


def _parse_cell_spec(spec: str) -> tuple[str, int]:
    """Parse a ``<task>_s<seed>`` cell identifier."""
    if not isinstance(spec, str) or "_s" not in spec:
        raise ValueError(f"Invalid cell spec {spec!r}; expected '<task>_s<seed>'")
    task_name, seed_text = spec.rsplit("_s", 1)
    if not task_name.strip() or not seed_text.isdigit():
        raise ValueError(f"Invalid cell spec {spec!r}; expected '<task>_s<seed>'")
    return task_name, int(seed_text)


def _attach_record_metadata(
    body: dict[str, Any],
    id_prefix: str,
    configuration_id: str | None,
    source_ids: list[str] | None,
    protocol_id: str | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {}
    if configuration_id is not None:
        record["id"] = f"{id_prefix}--{configuration_id}"
    record.update(body)
    if configuration_id is not None:
        record["configuration_id"] = configuration_id
    if source_ids is not None:
        record["source_ids"] = list(source_ids)
    if protocol_id is not None:
        record["protocol_id"] = protocol_id
    return record


def _summarize_usage(
    entries: list[tuple[Path, dict[str, Any], int]],
    *,
    benchmark_id: str,
    total_episodes: int,
    expected_model: str | None = None,
    configuration_id: str | None = None,
    source_ids: list[str] | None = None,
    extra_summary: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not entries:
        raise ValueError("Cannot summarize usage with zero completed transcripts")

    elapsed_sum = Decimal(0)
    input_tokens_sum = 0
    output_tokens_sum = 0
    cached_input_tokens_sum = 0
    in_file_retries_sum = 0
    observed_model: str | None = None

    for path, record, in_file_retries in entries:
        model = record.get("model")
        if not _is_nonempty_str(model):
            raise ValueError(f"Missing or invalid model in {path}")
        if observed_model is None:
            observed_model = model
        elif model != observed_model:
            raise ValueError(
                f"Inconsistent model in {path}: {model!r} != {observed_model!r}"
            )
        if expected_model is not None and model != expected_model:
            raise ValueError(
                f"Unexpected model in {path}: {model!r} != {expected_model!r}"
            )

        elapsed_s = record.get("elapsed_s")
        stats = record.get("stats")
        if not (_is_nonnegative_number(elapsed_s) and isinstance(stats, dict)):
            raise ValueError(f"Invalid elapsed_s or stats in {path}")

        for container in (record, stats):
            for key in ("planner", "backend"):
                if container.get(key) not in (None, "api"):
                    raise ValueError(f"Non-API {key} {container.get(key)!r} in {path}")

        input_tokens = stats.get("total_input_tokens")
        output_tokens = stats.get("total_output_tokens")
        cached_tokens = stats.get("total_cached_input_tokens", 0)
        if not all(
            _is_int(count) and count >= 0
            for count in (input_tokens, output_tokens, cached_tokens)
        ):
            raise ValueError(f"Invalid token counts in {path}")

        elapsed_sum += Decimal(str(elapsed_s))
        input_tokens_sum += input_tokens
        output_tokens_sum += output_tokens
        cached_input_tokens_sum += cached_tokens
        in_file_retries_sum += in_file_retries

    cost_episodes = len(entries)
    mean_elapsed_seconds = float(_round_two_decimals(elapsed_sum, cost_episodes))
    mean_total_tokens = float(
        _round_two_decimals(input_tokens_sum + output_tokens_sum, cost_episodes)
    )

    cost_body = {
        "benchmark_id": benchmark_id,
        "mean_elapsed_seconds": mean_elapsed_seconds,
        "total_output_tokens": output_tokens_sum,
    }
    cost_row = _attach_record_metadata(
        cost_body,
        benchmark_id,
        configuration_id,
        source_ids,
    )
    token_summary: dict[str, Any] = {
        "model": observed_model,
        "episodes": total_episodes,
        "cost_episodes": cost_episodes,
    }
    if extra_summary:
        token_summary.update(extra_summary)
    token_summary.update(
        {
            "excluded_in_file_attempts": in_file_retries_sum,
            "mean_elapsed_seconds": mean_elapsed_seconds,
            "total_input_tokens": input_tokens_sum,
            "total_output_tokens": output_tokens_sum,
            "total_cached_input_tokens": cached_input_tokens_sum,
            "mean_total_tokens": mean_total_tokens,
        }
    )
    return [cost_row], token_summary


def aggregate_robotwin(
    input_dir: Path | str,
    *,
    manifest_path: Path | str = ROBOTWIN_MANIFEST_PATH,
    planner: str = "api",
    expected_model: str | None = None,
    count_as_failure: list[str] | None = None,
    configuration_id: str | None = None,
    source_ids: list[str] | None = None,
    protocol_id: str | None = None,
) -> dict[str, Any]:
    """Aggregate RoboTwin C2R ``api`` transcripts against ``demo_randomized.json``."""
    if planner != "api":
        raise ValueError(f"Unsupported planner {planner!r}: only 'api' is supported")

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    tasks = manifest.get("tasks")
    expected_task_config = manifest.get("task_config")
    if not isinstance(tasks, dict) or not tasks:
        raise ValueError(f"Invalid RoboTwin manifest: {manifest_path}")

    expected_pairs: set[tuple[str, int]] = set()
    for task_name, seed_items in tasks.items():
        if not isinstance(seed_items, list):
            raise ValueError(f"Invalid seed list for task {task_name!r}")
        for entry in seed_items:
            seed = entry.get("seed") if isinstance(entry, dict) else entry
            if not _is_int(seed):
                raise ValueError(f"Invalid seed {seed!r} for task {task_name!r}")
            pair = (task_name, seed)
            if pair in expected_pairs:
                raise ValueError(f"Duplicate manifest pair {pair!r}")
            expected_pairs.add(pair)

    counted_failure_pairs: set[tuple[str, int]] = set()
    for raw_spec in count_as_failure or []:
        pair = _parse_cell_spec(raw_spec)
        if pair not in expected_pairs:
            raise ValueError(f"--count-as-failure cell {raw_spec!r} is not in manifest")
        if pair in counted_failure_pairs:
            raise ValueError(f"Duplicate --count-as-failure cell {raw_spec!r}")
        counted_failure_pairs.add(pair)

    seen_pairs: set[tuple[str, int]] = set()
    entries: list[tuple[Path, dict[str, Any], int]] = []
    successes = 0
    task_failures = 0
    episode_timeouts = 0

    for path in sorted(Path(input_dir).rglob("transcript_*.json")):
        record, in_file_retries = load_transcript_final_record(path)
        task_name = record.get("task_name")
        seed_values = [
            record[key]
            for key in ("requested_seed", "initial_native_seed", "seed")
            if key in record and record[key] is not None
        ]
        valid_seed = (
            bool(seed_values)
            and all(_is_int(value) for value in seed_values)
            and len(set(seed_values)) == 1
        )
        if not (_is_nonempty_str(task_name) and valid_seed):
            raise ValueError(f"Missing or inconsistent task_name/seed in {path}")
        seed = seed_values[0]
        if (
            expected_task_config is not None
            and record.get("task_config") is not None
            and record["task_config"] != expected_task_config
        ):
            raise ValueError(
                f"Unexpected task_config {record['task_config']!r} in {path}"
            )

        pair = (task_name, seed)
        if pair not in expected_pairs:
            raise ValueError(f"Unexpected episode pair {pair!r} in {path}")
        if pair in seen_pairs:
            raise ValueError(f"Duplicate episode pair {pair!r} in {path}")
        if pair in counted_failure_pairs:
            raise ValueError(
                f"Cell {task_name}_s{seed} passed to --count-as-failure "
                f"already has a transcript: {path}"
            )
        seen_pairs.add(pair)

        env_success = record.get("environment_success")
        if not isinstance(env_success, bool):
            raise ValueError(f"Non-boolean environment_success in {path}")

        if env_success:
            successes += 1
        else:
            error_value = record.get("error")
            if error_value is not None and not isinstance(error_value, str):
                raise ValueError(f"Non-string error in {path}")
            termination_reason = _classify_robotwin_termination(
                error_value,
                success=False,
            )
            if termination_reason == "planner_timeout":
                episode_timeouts += 1
            elif termination_reason in ("completed", "planner_response_truncated"):
                task_failures += 1
            else:
                raise ValueError(
                    f"Unretried infrastructure error in {path}: {error_value!r}"
                )
        entries.append((path, record, in_file_retries))

    missing_pairs = sorted(expected_pairs - seen_pairs - counted_failure_pairs)
    if missing_pairs:
        raise ValueError(
            f"Missing {len(missing_pairs)} RoboTwin episode(s): {missing_pairs[:3]}"
        )

    task_failures += len(counted_failure_pairs)
    total_episodes = len(expected_pairs)
    counted_failure_labels = sorted(
        f"{task_name}_s{seed}" for task_name, seed in counted_failure_pairs
    )

    counts_body = {
        "successes": successes,
        "task_failures": task_failures,
        "episode_timeouts": episode_timeouts,
    }
    result_body = {
        "view_id": "robotwin",
        "rate": str(_round_two_decimals(successes, total_episodes, 100)),
        "episodes": total_episodes,
        **counts_body,
    }
    result_record = _attach_record_metadata(
        result_body,
        "robotwin",
        configuration_id,
        source_ids,
        protocol_id,
    )
    extra_summary = {
        **counts_body,
        "counted_as_failure_cells": counted_failure_labels,
    }
    cost_rows, token_summary = _summarize_usage(
        entries,
        benchmark_id="robotwin",
        total_episodes=total_episodes,
        expected_model=expected_model,
        configuration_id=configuration_id,
        source_ids=source_ids,
        extra_summary=extra_summary,
    )
    return {
        "results": [result_record],
        "cost_results": cost_rows,
        "token_summary": token_summary,
    }


def aggregate_robocasa(
    input_dir: Path | str,
    summary_path: Path | str,
    *,
    manifest_path: Path | str = ROBOCASA_MANIFEST_PATH,
    planner: str = "api",
    expected_model: str | None = None,
    configuration_id: str | None = None,
    source_ids: list[str] | None = None,
    protocol_id: str | None = None,
) -> dict[str, Any]:
    """Aggregate RoboCasa Target50 results from ``validate_target50`` + transcripts."""
    if planner != "api":
        raise ValueError(f"Unsupported planner {planner!r}: only 'api' is supported")

    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    if (
        not isinstance(summary, dict)
        or summary.get("protocol_id") != "robocasa-harness-vla-v2"
    ):
        raise ValueError("Invalid Target50 protocol_id in summary")

    expected_cells = summary.get("expected_cells")
    overall = summary.get("overall")
    splits = summary.get("splits")
    if not (
        _is_int(expected_cells)
        and expected_cells > 0
        and summary.get("valid_cells") == expected_cells
    ):
        raise ValueError("Target50 summary is incomplete or invalid")
    if not (
        isinstance(overall, dict)
        and overall.get("metric") == "task_weighted_success_rate"
        and _is_nonnegative_number(overall.get("success_rate"))
        and overall["success_rate"] <= 1
    ):
        raise ValueError("Invalid Target50 overall summary")
    if not isinstance(splits, dict):
        raise ValueError("Target50 summary missing splits")

    overall_body = {
        "view_id": "robocasa",
        "rate": str(_round_two_decimals(overall["success_rate"], 1, 100)),
        "episodes": expected_cells,
    }
    results_out = [
        _attach_record_metadata(
            overall_body,
            "robocasa",
            configuration_id,
            source_ids,
            protocol_id,
        )
    ]

    split_expected_counts: dict[str, int] = {}
    for split_name, view_id in _ROBOCASA_SPLIT_TO_VIEW:
        split_summary = splits.get(split_name)
        if not isinstance(split_summary, dict):
            raise ValueError(f"Target50 summary missing split {split_name!r}")
        split_successes = split_summary.get("successes")
        split_valid = split_summary.get("valid_cells")
        split_expected = split_summary.get("expected_cells")
        if not (
            _is_int(split_expected)
            and split_expected > 0
            and split_valid == split_expected
            and _is_int(split_successes)
            and 0 <= split_successes <= split_expected
        ):
            raise ValueError(f"Invalid Target50 split summary for {split_name!r}")
        split_expected_counts[split_name] = split_expected
        split_body = {
            "view_id": view_id,
            "rate": str(_round_two_decimals(split_successes, split_expected, 100)),
            "successes": split_successes,
            "episodes": split_expected,
        }
        results_out.append(
            _attach_record_metadata(
                split_body,
                view_id,
                configuration_id,
                source_ids,
                protocol_id,
            )
        )
    if sum(split_expected_counts.values()) != expected_cells:
        raise ValueError("Target50 split expected_cells sum mismatch")

    root = Path(input_dir)
    all_transcripts = sorted(root.rglob("transcript_*.json"))
    if len(all_transcripts) != expected_cells:
        raise ValueError(
            f"Expected {expected_cells} transcripts, got {len(all_transcripts)}"
        )

    manifest_splits = json.loads(Path(manifest_path).read_text(encoding="utf-8")).get(
        "splits", {}
    )
    entries: list[tuple[Path, dict[str, Any], int]] = []
    matched_paths: set[Path] = set()

    for split_name, split_expected in split_expected_counts.items():
        manifest_split = manifest_splits.get(split_name)
        if not isinstance(manifest_split, dict):
            raise ValueError(f"Manifest missing split {split_name!r}")
        expected_split_cells = [
            (task_name, seed)
            for task_name in manifest_split.get("tasks", [])
            for seed in manifest_split.get("seeds", [])
        ]
        if len(expected_split_cells) != split_expected:
            raise ValueError(f"Manifest split {split_name!r} cell count mismatch")
        for task_name, seed in expected_split_cells:
            cell_dir = root / split_name / f"{task_name}_s{seed}"
            result_file = cell_dir / "result.json"
            result_model: str | None = None
            if result_file.is_file():
                planner_info = json.loads(result_file.read_text(encoding="utf-8")).get(
                    "planner"
                )
                if isinstance(planner_info, dict):
                    backend = planner_info.get("backend")
                    if backend not in (None, "api"):
                        raise ValueError(
                            f"Non-API planner backend {backend!r} in {result_file}"
                        )
                    if isinstance(planner_info.get("model"), str):
                        result_model = planner_info["model"]

            cell_transcripts = sorted(cell_dir.glob("transcript_*.json"))
            if len(cell_transcripts) != 1:
                raise ValueError(f"Expected 1 transcript in {cell_dir}")
            transcript_path = cell_transcripts[0]
            record, in_file_retries = load_transcript_final_record(transcript_path)
            observed_pair = (record.get("task_name"), record.get("seed"))
            if observed_pair != (task_name, seed):
                raise ValueError(
                    f"Transcript {observed_pair!r} != expected {(task_name, seed)!r}"
                )
            if result_model is not None and record.get("model") != result_model:
                raise ValueError(
                    f"Transcript model {record.get('model')!r} != "
                    f"result.json model {result_model!r} in {cell_dir}"
                )
            matched_paths.add(transcript_path.resolve())
            entries.append((transcript_path, record, in_file_retries))

    if matched_paths != {path.resolve() for path in all_transcripts}:
        raise ValueError("Found stray transcripts outside manifest cell directories")

    cost_rows, token_summary = _summarize_usage(
        entries,
        benchmark_id="robocasa",
        total_episodes=expected_cells,
        expected_model=expected_model,
        configuration_id=configuration_id,
        source_ids=source_ids,
    )
    return {
        "results": results_out,
        "cost_results": cost_rows,
        "token_summary": token_summary,
    }


def _require_new_entry(
    item: Any,
    seen_ids: set[str],
    label: str,
) -> None:
    if not isinstance(item, dict) or not _is_nonempty_str(item.get("id")):
        raise LeaderboardValidationError(f"Leaderboard: invalid {label} entry")
    item_id = item["id"]
    if item_id in seen_ids:
        raise LeaderboardValidationError(
            f"Leaderboard: cannot overwrite existing {label} {item_id!r}"
        )
    seen_ids.add(item_id)


def upsert_data(data: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Append new validated entries to a dataset without overwriting existing data."""
    validate_data(data)
    allowed_keys = {
        "configurations",
        "results",
        "cost_results",
        "sources",
        "benchmark_notes",
    }
    if not isinstance(payload, dict) or not (set(payload) <= allowed_keys):
        raise LeaderboardValidationError("Leaderboard: invalid upsert payload")

    updated = _Dataset(copy.deepcopy(dict(data)))
    if isinstance(data, _Dataset) and data._array_styles is not None:
        updated._array_styles = dict(data._array_styles)

    seen_sources = {source["id"] for source in updated["sources"]}
    new_sources = payload.get("sources", [])
    if not isinstance(new_sources, list):
        raise LeaderboardValidationError("Leaderboard: upsert sources must be a list")
    for source in new_sources:
        _require_new_entry(source, seen_sources, "source")
        updated["sources"].append(copy.deepcopy(source))

    configs = updated["configurations"]
    seen_configs = {config["id"] for config in configs}
    new_configs = payload.get("configurations", [])
    if not isinstance(new_configs, list):
        raise LeaderboardValidationError(
            "Leaderboard: upsert configurations must be a list"
        )
    for raw_config in new_configs:
        _require_new_entry(raw_config, seen_configs, "configuration")
        config = copy.deepcopy(raw_config)
        anchor = config.pop("insert_after", None)
        if anchor is not None:
            if not isinstance(anchor, str):
                raise LeaderboardValidationError(
                    f"Leaderboard: insert_after must be a string, got {anchor!r}"
                )
            anchor_index = next(
                (i for i, existing in enumerate(configs) if existing["id"] == anchor),
                None,
            )
            if anchor_index is None:
                raise LeaderboardValidationError(
                    f"Leaderboard: unknown insert_after {anchor!r}"
                )
            configs.insert(anchor_index + 1, config)
        elif config.get("kind") == "rpent":
            first_external = next(
                (
                    i
                    for i, existing in enumerate(configs)
                    if existing.get("kind") != "rpent"
                ),
                len(configs),
            )
            configs.insert(first_external, config)
        else:
            configs.append(config)

    benchmark_by_id = {
        benchmark["id"]: benchmark for benchmark in updated["benchmarks"]
    }
    seen_notes = {
        note["id"]
        for benchmark in updated["benchmarks"]
        for note in benchmark.get("notes", [])
    }
    seen_notes |= {note["id"] for note in updated.get("footer_notes", [])}
    benchmark_notes = payload.get("benchmark_notes", {})
    if not isinstance(benchmark_notes, dict):
        raise LeaderboardValidationError(
            "Leaderboard: benchmark_notes must be an object"
        )
    for benchmark_id, notes_to_add in benchmark_notes.items():
        if (
            not isinstance(benchmark_id, str)
            or benchmark_id not in benchmark_by_id
            or not isinstance(notes_to_add, list)
        ):
            raise LeaderboardValidationError(
                f"Leaderboard: invalid benchmark_notes[{benchmark_id!r}]"
            )
        target_notes = benchmark_by_id[benchmark_id].setdefault("notes", [])
        for note in notes_to_add:
            _require_new_entry(note, seen_notes, "note")
            target_notes.append(copy.deepcopy(note))

    table_targets = (
        ("results", updated["results"], "view_id"),
        ("cost_results", updated["cost_results"]["rows"], "benchmark_id"),
    )
    for field_name, target_list, group_key in table_targets:
        items_to_add = payload.get(field_name, [])
        if not isinstance(items_to_add, list):
            raise LeaderboardValidationError(
                f"Leaderboard: upsert {field_name} must be a list"
            )
        seen_ids = {item["id"] for item in target_list}
        seen_pairs = {
            (item[group_key], item["configuration_id"]) for item in target_list
        }
        for item in items_to_add:
            _require_new_entry(item, seen_ids, field_name)
            group_value = item.get(group_key)
            config_value = item.get("configuration_id")
            if not (isinstance(group_value, str) and isinstance(config_value, str)):
                raise LeaderboardValidationError(
                    f"Leaderboard: invalid keys in {field_name}"
                )
            pair = (group_value, config_value)
            if pair in seen_pairs:
                raise LeaderboardValidationError(
                    f"Leaderboard: cannot overwrite {field_name} {pair!r}"
                )
            seen_pairs.add(pair)
            target_list.append(copy.deepcopy(item))

    validate_data(updated)
    return updated


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate a leaderboard results.json dataset.",
    )
    validate_parser.add_argument("--data", type=Path, default=DATA_PATH)

    aggregate_parser = subparsers.add_parser(
        "aggregate",
        help="Aggregate API planner transcripts for RoboTwin or RoboCasa.",
    )
    aggregate_parser.add_argument(
        "--benchmark",
        choices=("robotwin", "robocasa"),
        required=True,
    )
    aggregate_parser.add_argument("--input-dir", type=Path, required=True)
    aggregate_parser.add_argument("--planner", default="api")
    aggregate_parser.add_argument("--manifest", type=Path, default=None)
    aggregate_parser.add_argument("--target50-summary", type=Path, default=None)
    aggregate_parser.add_argument("--expected-model", default=None)
    aggregate_parser.add_argument(
        "--count-as-failure",
        action="append",
        dest="count_as_failure",
        default=None,
        help="RoboTwin cell '<task>_s<seed>' counted as task failure after retry cap.",
    )
    aggregate_parser.add_argument("--configuration-id", default=None)
    aggregate_parser.add_argument("--protocol-id", default=None)
    aggregate_parser.add_argument(
        "--source-id",
        action="append",
        dest="source_ids",
        default=None,
    )
    aggregate_parser.add_argument("--output", type=Path, default=None)

    upsert_parser = subparsers.add_parser(
        "upsert",
        help="Append validated entries to results.json without overwriting.",
    )
    upsert_parser.add_argument("--data", type=Path, default=DATA_PATH)
    upsert_parser.add_argument("--payload", type=Path, required=True)
    mode = upsert_parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--in-place", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the leaderboard validation, aggregation, and upsert CLI."""
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "validate":
            dataset = load_data(args.data)
            print(
                f"OK: {args.data} (v{dataset['schema_version']}, "
                f"{len(dataset['results'])} rows)"
            )
            return 0

        if args.command == "aggregate":
            shared_kwargs = {
                "planner": args.planner,
                "expected_model": args.expected_model,
                "configuration_id": args.configuration_id,
                "source_ids": args.source_ids,
                "protocol_id": args.protocol_id,
            }
            if args.benchmark == "robotwin":
                if args.target50_summary is not None:
                    raise ValueError(
                        "--target50-summary is only valid for --benchmark robocasa"
                    )
                manifest_path = args.manifest or ROBOTWIN_MANIFEST_PATH
                aggregated = aggregate_robotwin(
                    args.input_dir,
                    manifest_path=manifest_path,
                    count_as_failure=args.count_as_failure,
                    **shared_kwargs,
                )
            else:
                if args.target50_summary is None:
                    raise ValueError(
                        "--target50-summary is required for --benchmark robocasa"
                    )
                if args.count_as_failure:
                    raise ValueError(
                        "--count-as-failure is only valid for --benchmark robotwin"
                    )
                manifest_path = args.manifest or ROBOCASA_MANIFEST_PATH
                aggregated = aggregate_robocasa(
                    args.input_dir,
                    args.target50_summary,
                    manifest_path=manifest_path,
                    **shared_kwargs,
                )
            output_text = json.dumps(aggregated, indent=2, ensure_ascii=False) + "\n"
            if args.output is not None:
                _atomic_write(args.output, output_text)
            else:
                print(output_text, end="")
            return 0

        payload = json.loads(args.payload.read_text(encoding="utf-8"))
        updated = upsert_data(load_data(args.data), payload)
        if args.dry_run:
            old_lines = args.data.read_text(encoding="utf-8").splitlines(keepends=True)
            new_lines = dump_data(updated).splitlines(keepends=True)
            diff = difflib.unified_diff(old_lines, new_lines, str(args.data), "updated")
            print("".join(diff), end="")
        else:
            write_data(args.data, updated)
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
