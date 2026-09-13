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

"""Validate benchmark records and generate the bilingual RST result tables.

Only the Python standard library is required. JSON is the editable source;
the marked list-table blocks are generated documentation, not another dataset.
"""

from __future__ import annotations

import argparse
import json
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1]
DATA_PATH = DOCS / "_static/benchmarks/results.json"
LANGUAGES = ("en", "zh")


def load_data(path: Path | str | None = None) -> dict:
    """Read the source without importing plotting or documentation packages."""
    return json.loads(Path(path or DATA_PATH).read_text(encoding="utf-8"))


def indexed(items: list[dict], name: str) -> dict[str, dict]:
    """Index records while rejecting IDs that would silently overwrite records."""
    result = {}
    for item in items:
        key = item.get("id")
        if not isinstance(key, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", key):
            raise ValueError(f"Invalid {name} ID: {key!r}")
        if key in result:
            raise ValueError(f"Duplicate {name} ID: {key}")
        result[key] = item
    return result


def _translated(value: dict, name: str) -> None:
    if not isinstance(value, dict) or any(
        not isinstance(value.get(language), str) or not value[language].strip()
        for language in LANGUAGES
    ):
        raise ValueError(f"Missing English/Chinese text: {name}")


def _references(values: list[str], choices: dict, name: str) -> None:
    if not isinstance(values, list) or len(values) != len(set(values)):
        raise ValueError(f"Invalid or duplicate references: {name}")
    for value in values:
        if value not in choices:
            raise ValueError(f"Unknown reference {value!r}: {name}")


def _rate(record: dict) -> Decimal:
    return Decimal(record["rate"])


def validate_data(data: dict) -> None:
    """Check scope, references, exact precision, counts, and aggregate coverage."""
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported benchmark schema_version")
    configurations = indexed(data["configurations"], "configuration")
    benchmarks = indexed(data["benchmarks"], "benchmark")
    sources = indexed(data["sources"], "source")
    protocols = indexed(data["protocols"], "protocol")
    views = indexed(data["views"], "view")
    records = indexed(data["results"], "result")
    tables = indexed(data["tables"], "table")
    planners = [
        key for key, value in configurations.items() if value["kind"] == "rpent"
    ]
    if not planners:
        raise ValueError("No RPent planner configurations")
    for configuration in configurations.values():
        if configuration["kind"] not in ("rpent", "external"):
            raise ValueError(f"Invalid configuration kind: {configuration['id']}")
        if not isinstance(configuration["model"], str) or not configuration["model"]:
            raise ValueError(f"Missing model/method: {configuration['id']}")
        if (
            configuration["reasoning"] is not None
            and type(configuration["reasoning"]) is not bool
        ):
            raise ValueError(
                f"Reasoning must be boolean or null: {configuration['id']}"
            )
        if configuration["kind"] == "rpent" and not configuration["backend"]:
            raise ValueError(f"Missing planner backend: {configuration['id']}")
        if configuration["kind"] == "external" and configuration.get(
            "metadata_status"
        ) not in ("not_reported", "not_applicable"):
            raise ValueError(
                f"External metadata must distinguish unreported from inapplicable: {configuration['id']}"
            )
    for benchmark in benchmarks.values():
        _translated(benchmark["title"], f"benchmark {benchmark['id']}")
    for source in sources.values():
        _translated(source["label"], f"source {source['id']}")
        if not source["url"].startswith("https://") or not source["anchor"]:
            raise ValueError(f"Invalid source URL/anchor: {source['id']}")
    for protocol in protocols.values():
        _translated(protocol["label"], f"protocol {protocol['id']}")
        _translated(protocol["description"], f"protocol {protocol['id']}")

    pairs = {}
    for record in records.values():
        rid = record["id"]
        if (
            record["view_id"] not in views
            or record["configuration_id"] not in configurations
        ):
            raise ValueError(f"Unknown result view/configuration: {rid}")
        pair = (record["view_id"], record["configuration_id"])
        if pair in pairs:
            raise ValueError(f"Duplicate model within an evaluation scope: {pair}")
        pairs[pair] = record
        if record["protocol_id"] not in protocols:
            raise ValueError(f"Unknown protocol: {rid}")
        _references(record["source_ids"], sources, f"sources for {rid}")
        if not record["source_ids"]:
            raise ValueError(f"Missing source: {rid}")
        if record["status"] == "not_reported":
            if any(
                record[field] is not None for field in ("rate", "successes", "episodes")
            ):
                raise ValueError(
                    f"Unreported results must have null rate/counts: {rid}"
                )
            continue
        if record["status"] != "reported":
            raise ValueError(f"Invalid status: {rid}")
        if not isinstance(record["rate"], str) or not re.fullmatch(
            r"\d+(?:\.\d+)?", record["rate"]
        ):
            raise ValueError(f"Rate must preserve its decimal string: {rid}")
        rate = _rate(record)
        if not Decimal(0) <= rate <= Decimal(100):
            raise ValueError(f"Rate outside 0–100: {rid}")
        successes, episodes = record["successes"], record["episodes"]
        if episodes is not None and (type(episodes) is not int or episodes <= 0):
            raise ValueError(f"Invalid episode denominator: {rid}")
        if successes is not None:
            if (
                type(successes) is not int
                or episodes is None
                or not 0 <= successes <= episodes
            ):
                raise ValueError(f"Invalid success count: {rid}")
            count_rate = (Decimal(successes) * 100 / episodes).quantize(
                rate, rounding=ROUND_HALF_UP
            )
            if count_rate != rate:
                raise ValueError(f"Count and displayed rate disagree: {rid}")

    assigned_records = []
    for view in views.values():
        vid = view["id"]
        if view["benchmark_id"] not in benchmarks:
            raise ValueError(f"Unknown benchmark: {vid}")
        for field in (
            "title",
            "label",
            "subtitle",
            "scope",
            "table_section",
            "table_row",
            "source_note",
        ):
            _translated(view[field], f"{vid}.{field}")
        _references(view["source_ids"], sources, f"view sources for {vid}")
        _references(view["record_ids"], records, f"records for {vid}")
        _references(view["chart_record_ids"], records, f"chart for {vid}")
        if view["asset_slug"] != vid:
            raise ValueError(f"Asset slug must match stable view ID: {vid}")
        assigned_records.extend(view["record_ids"])
        for rid in view["record_ids"]:
            if records[rid]["view_id"] != vid:
                raise ValueError(
                    f"Record belongs to a different evaluation scope: {rid}"
                )
        for field in ("chart_record_ids", "overview_record_ids"):
            if field not in view:
                continue
            _references(view[field], records, f"{vid}.{field}")
            for rid in view[field]:
                if (
                    rid not in view["record_ids"]
                    or records[rid]["status"] != "reported"
                ):
                    raise ValueError(
                        f"Chart contains unreported or cross-scope record: {vid}/{rid}"
                    )
        reported_planners = {
            rid
            for rid in view["record_ids"]
            if records[rid]["configuration_id"] in planners
            and records[rid]["status"] == "reported"
        }
        if not reported_planners.issubset(view["chart_record_ids"]):
            raise ValueError(f"Chart omits a reported RPent configuration: {vid}")
        maximum = max(
            (
                _rate(records[rid])
                for rid in view["record_ids"]
                if records[rid]["status"] == "reported"
            ),
            default=Decimal(0),
        )
        # Decimal ceiling also handles rates such as 90.1 without float rounding.
        axis_max = max(
            10, int((maximum / 10).to_integral_value(rounding="ROUND_CEILING")) * 10
        )
        if view["axis_max"] != axis_max:
            raise ValueError(f"Stale axis limit for {vid}: expected {axis_max}")
        if "aggregate_of" in view:
            _references(view["aggregate_of"], views, f"aggregate {vid}")
            for configuration_id in planners:
                aggregate = pairs.get((vid, configuration_id))
                if aggregate is None or aggregate["status"] != "reported":
                    continue
                items = [
                    pairs.get((item, configuration_id)) for item in view["aggregate_of"]
                ]
                if any(item is None or item["status"] != "reported" for item in items):
                    raise ValueError(
                        f"Incomplete coverage cannot define Overall: {vid}/{configuration_id}"
                    )
                if aggregate["episodes"] != sum(item["episodes"] for item in items):
                    raise ValueError(
                        f"Overall denominator differs from covered items: {vid}/{configuration_id}"
                    )
                mean = sum(_rate(item) for item in items) / len(items)
                if mean.quantize(_rate(aggregate), rounding=ROUND_HALF_UP) != _rate(
                    aggregate
                ):
                    raise ValueError(
                        f"Overall rate differs from its covered items: {vid}/{configuration_id}"
                    )
    if sorted(assigned_records) != sorted(records):
        raise ValueError("Every result must belong to exactly one view")

    _references(data["overview_views"], views, "overview views")
    for vid in data["overview_views"]:
        if not views[vid].get("overview_record_ids"):
            raise ValueError(f"Missing overview bars: {vid}")
        if views[vid]["overview_record_ids"] != views[vid]["chart_record_ids"]:
            raise ValueError(
                f"Default overview and detail charts must use the same bars: {vid}"
            )
    table_results = []
    for table in tables.values():
        _translated(table["title"], f"table {table['id']}")
        if table["kind"] == "main":
            _references(table["view_ids"], views, f"table {table['id']}")
            for vid in table["view_ids"]:
                if "aggregate_of" in views[vid] and set(
                    views[vid]["aggregate_of"]
                ) != set(table["view_ids"]) - {vid}:
                    raise ValueError(
                        f"Overall must cover every other item in its main table: {vid}"
                    )
                for cid in planners:
                    if (vid, cid) not in pairs:
                        raise ValueError(
                            f"Missing explicit reported/not-reported cell: {vid}/{cid}"
                        )
                    table_results.append(pairs[(vid, cid)]["id"])
        elif table["kind"] == "baseline":
            _references(table["record_ids"], records, f"table {table['id']}")
            for rid in table["record_ids"]:
                if (
                    configurations[records[rid]["configuration_id"]]["kind"]
                    != "external"
                ):
                    raise ValueError(f"Planner in external baseline appendix: {rid}")
                _translated(records[rid]["coverage"], f"coverage for {rid}")
            table_results.extend(table["record_ids"])
        elif table["kind"] != "configurations":
            raise ValueError(f"Unknown table kind: {table['id']}")
    if sorted(table_results) != sorted(records):
        raise ValueError(
            "Every result must be represented exactly once in the detailed tables"
        )


def formatted_rate(record: dict, language: str) -> str:
    if record["status"] == "not_reported":
        return "Not reported" if language == "en" else "未报告"
    text = f"{record['rate']}%"
    if record["successes"] is not None:
        text += f" ({record['successes']}/{record['episodes']})"
    return text


def render_table(data: dict, table: dict, language: str) -> str:
    """Produce the existing list-table style, including baseline indentation."""
    configurations = indexed(data["configurations"], "configuration")
    planners = [item for item in configurations.values() if item["kind"] == "rpent"]
    records = indexed(data["results"], "result")
    views = indexed(data["views"], "view")
    sources = indexed(data["sources"], "source")
    pairs = {
        (item["view_id"], item["configuration_id"]): item for item in records.values()
    }
    indent = "   " if table["kind"] == "baseline" else ""
    if table["kind"] == "configurations":
        rows = [table["headers"][language]]
        for item in planners:
            reasoning = (
                ("On" if language == "en" else "开启")
                if item["reasoning"]
                else ("Off" if language == "en" else "关闭")
            )
            if item["reasoning"] is None:
                reasoning = "Not reported" if language == "en" else "未报告"
            effort = (
                f"``{item['effort']}``"
                if item["effort"]
                else ("Not reported" if language == "en" else "未报告")
            )
            rows.append([item["backend"], item["model"], reasoning, effort])
        widths = table["widths"]
    elif table["kind"] == "main":
        rows = [
            [table["headers"][language]]
            + [
                f"{item['model']} / ``{item['effort']}``"
                if item["effort"]
                else item["model"]
                for item in planners
            ]
        ]
        for vid in table["view_ids"]:
            rows.append(
                [views[vid]["table_row"][language]]
                + [
                    formatted_rate(pairs[(vid, item["id"])], language)
                    for item in planners
                ]
            )
        widths = [31] + [23] * len(planners)
    else:
        rows = [table["headers"][language]]
        for rid in table["record_ids"]:
            record = records[rid]
            references = [
                f":ref:`{sources[sid]['label'][language]} <{sources[sid]['anchor']}>`"
                for sid in record["source_ids"]
            ]
            rows.append(
                [
                    configurations[record["configuration_id"]]["model"],
                    record["coverage"][language],
                    formatted_rate(record, language),
                    "; ".join(references),
                ]
            )
        widths = table["widths"]
    lines = [
        f"{indent}.. list-table:: {table['title'][language]}",
        f"{indent}   :header-rows: 1",
        f"{indent}   :widths: {' '.join(map(str, widths))}",
        "",
    ]
    for row in rows:
        lines.append(f"{indent}   * - {row[0]}")
        lines.extend(f"{indent}     - {cell}" for cell in row[1:])
    return "\n".join(lines) + "\n"


def _block(table: dict, body: str) -> str:
    indent = "   " if table["kind"] == "baseline" else ""
    return f"{indent}.. benchmark-data-begin: {table['id']}\n\n{body}\n{indent}.. benchmark-data-end: {table['id']}\n"


def _pattern(table: dict) -> re.Pattern:
    indent = "   " if table["kind"] == "baseline" else ""
    return re.compile(
        rf"^{indent}\.\. benchmark-data-begin: {re.escape(table['id'])}\n.*?^{indent}\.\. benchmark-data-end: {re.escape(table['id'])}\n",
        re.MULTILINE | re.DOTALL,
    )


def generated_tables_check(data: dict) -> int:
    """Check both generated RST pages without writing; return result-cell count."""
    for language in LANGUAGES:
        path = DOCS / f"source-{language}/rst_source/benchmarks.rst"
        text = path.read_text(encoding="utf-8")
        for table in data["tables"]:
            matches = list(_pattern(table).finditer(text))
            if len(matches) != 1:
                raise ValueError(
                    f"Missing or duplicate generated block: {language}/{table['id']}"
                )
            if matches[0][0] != _block(table, render_table(data, table, language)):
                raise ValueError(
                    f"Stale table: {language}/{table['id']}; run benchmark_data.py --write"
                )
        for source in data["sources"]:
            if source["url"] not in text or f".. _{source['anchor']}:" not in text:
                raise ValueError(
                    f"Missing source URL/anchor: {language}/{source['id']}"
                )
    return len(data["results"]) * len(LANGUAGES)


def write_generated_tables(data: dict) -> None:
    """Update only established marked regions; fail rather than replace prose."""
    for language in LANGUAGES:
        path = DOCS / f"source-{language}/rst_source/benchmarks.rst"
        text = path.read_text(encoding="utf-8")
        for table in data["tables"]:
            pattern = _pattern(table)
            if len(list(pattern.finditer(text))) != 1:
                raise ValueError(
                    f"Missing or duplicate generated block: {language}/{table['id']}"
                )
            replacement = _block(table, render_table(data, table, language))
            text = pattern.sub(lambda _: replacement, text)
        path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--check",
        action="store_true",
        help="Validate records and reject stale generated tables",
    )
    action.add_argument(
        "--write",
        action="store_true",
        help="Regenerate only the marked bilingual table blocks",
    )
    args = parser.parse_args()
    data = load_data()
    validate_data(data)
    if args.write:
        write_generated_tables(data)
    checked = generated_tables_check(data)
    print(
        f"Validated {len(data['views'])} evaluation scopes, {len(data['results'])} records, and {checked} bilingual result cells across {len(data['tables']) * 2} generated tables."
    )


if __name__ == "__main__":
    main()
