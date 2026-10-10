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

"""Build the leaderboard's fallback tables and README images from one dataset."""

import json
import math
from pathlib import Path

from docutils import nodes
from jinja2 import Environment, FileSystemLoader
from sphinx.application import Sphinx
from sphinx.errors import ExtensionError
from sphinx.util.docutils import SphinxDirective
from sphinx.util.osutil import relative_uri

ASSETS = Path(__file__).resolve().parents[1] / "_static" / "leaderboard"
TEMPLATES = Environment(
    loader=FileSystemLoader(Path(__file__).parent / "templates"), autoescape=True
)
COPY = {
    "en": {
        "performance": "Performance",
        "costs": "Time & Token Costs",
        "method": "Method",
        "rate": "Success rate",
        "time": "Mean time / episode (s)",
        "mean_total_tokens": "Mean input + output tokens / episode",
        "total_output_tokens": "Total output tokens",
        "scope": "Rankings apply to the methods and evaluation coverage shown.",
    },
    "zh": {
        "performance": "评测成绩",
        "costs": "耗时与 Token 开销",
        "method": "方法",
        "rate": "成功率",
        "time": "平均每回合耗时（秒）",
        "mean_total_tokens": "平均每回合输入＋输出 token",
        "total_output_tokens": "总输出 token",
        "scope": "排名仅限图中方法及评测范围。",
    },
}


def load_data(path: Path = ASSETS / "results.json") -> dict:
    """Read results and reject broken references before publishing a build."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["schema_version"] != 4:
        raise ExtensionError("Leaderboard: expected schema_version 4")

    def index(items: list[dict], kind: str) -> dict:
        result = {item["id"]: item for item in items}
        if len(result) != len(items):
            raise ExtensionError(f"Leaderboard: duplicate ID in {kind}")
        return result

    configs = index(data["configurations"], "configurations")
    benchmarks = index(data["benchmarks"], "benchmarks")
    views = index(
        [view for benchmark in benchmarks.values() for view in benchmark["views"]],
        "views",
    )
    results = index(data["results"], "results")
    index(data["cost_results"]["rows"], "cost_results")
    sources = index(data["sources"], "sources")
    notes = [
        note for benchmark in benchmarks.values() for note in benchmark.get("notes", [])
    ]
    index(notes, "notes")
    for benchmark in benchmarks.values():
        if benchmark["default_view"] not in {view["id"] for view in benchmark["views"]}:
            raise ExtensionError(
                f"Leaderboard: invalid default view in {benchmark['id']}"
            )
    for note in notes:
        if any(key not in configs for key in note["configuration_ids"]):
            raise ExtensionError(f"Leaderboard: unknown configuration in {note['id']}")
    for collection, group_key, groups, source_ids in (
        (data["results"], "view_id", views, []),
        (
            data["cost_results"]["rows"],
            "benchmark_id",
            benchmarks,
            data["cost_results"]["source_ids"],
        ),
    ):
        pairs = set()
        for record in collection:
            record_id = record["id"]
            if (
                record["configuration_id"] not in configs
                or record[group_key] not in groups
            ):
                raise ExtensionError(
                    f"Leaderboard: unknown configuration or {group_key} in {record_id}"
                )
            pair = (record[group_key], record["configuration_id"])
            if pair in pairs:
                raise ExtensionError(f"Leaderboard: duplicate result for {pair}")
            pairs.add(pair)
            record_sources = record.get("source_ids", source_ids)
            if not record_sources or any(key not in sources for key in record_sources):
                raise ExtensionError(
                    f"Leaderboard: unknown or missing source in {record_id}"
                )
            for key in (
                "rate",
                "mean_elapsed_seconds",
                "mean_total_tokens",
                "total_output_tokens",
            ):
                value = record.get(key)
                if value is None:
                    continue
                try:
                    number = float(value)
                except (TypeError, ValueError) as error:
                    raise ExtensionError(
                        f"Leaderboard: invalid {key} in {record_id}"
                    ) from error
                if not (
                    math.isfinite(number)
                    and number >= 0
                    and (key != "rate" or number <= 100)
                ):
                    raise ExtensionError(f"Leaderboard: invalid {key} in {record_id}")
            if any(key not in results for key in record.get("derived_from", [])):
                raise ExtensionError(
                    f"Leaderboard: unknown derived result in {record_id}"
                )
    return data


def context(data: dict, language: str) -> dict:
    """Prepare the same reported results for localized tables and chart images."""

    def translate(value):
        return (
            value.get(language, value.get("en", ""))
            if isinstance(value, dict)
            else value
        )

    configs = {item["id"]: item for item in data["configurations"]}
    records_by_view = {}
    for record in data["results"]:
        records_by_view.setdefault(record["view_id"], []).append(record)

    def label(config):
        if config.get("display_name"):
            return config["display_name"]
        if config["kind"] == "external":
            return config["model"]
        return "RPent / " + (config.get("model") or config.get("backend") or "—")

    def detail(config):
        if config.get("perception_model"):
            return config["perception_model"] + (
                " · visual localization" if language == "en" else " · 视觉定位"
            )
        reasoning = config.get("reasoning")
        return " · ".join(
            filter(
                None,
                [
                    config.get("backend"),
                    config.get("effort"),
                    "reasoning"
                    if reasoning is True
                    else "no-reasoning"
                    if reasoning is False
                    else None,
                ],
            )
        )

    sections = []
    for section in data["benchmarks"]:
        notes = [
            {"id": note["id"], "number": number, "text": translate(note["text"])}
            for number, note in enumerate(section.get("notes", []), 1)
        ]
        references = {}
        for note, rendered in zip(section.get("notes", []), notes):
            for config_id in note["configuration_ids"]:
                references.setdefault(config_id, []).append(rendered)
        columns = section["views"]
        rows = []
        for config in data["configurations"]:
            cells = [
                next(
                    (
                        record
                        for record in records_by_view.get(view["id"], [])
                        if record["configuration_id"] == config["id"]
                    ),
                    None,
                )
                for view in columns
            ]
            if not any(cell is not None and cell["rate"] is not None for cell in cells):
                continue
            rows.append(
                {
                    "label": label(config),
                    "detail": detail(config),
                    "cells": cells,
                    "notes": references.get(config["id"], []),
                }
            )
        overview = [
            record
            for record in records_by_view.get(section["default_view"], [])
            if record["rate"] is not None
        ]
        overview.sort(key=lambda record: (-float(record["rate"]), record["id"]))
        bars = [
            {
                **record,
                "label": label(configs[record["configuration_id"]]),
                "detail": detail(configs[record["configuration_id"]]),
                "width": float(record["rate"]) * 2.35,
                "color_role": "best"
                if float(record["rate"]) == float(overview[0]["rate"])
                else "reference"
                if configs[record["configuration_id"]]["kind"] == "external"
                else "rpent",
            }
            for record in overview
        ]
        sections.append(
            {
                **section,
                "columns": [translate(view["label"]) for view in columns],
                "rows": rows,
                "bars": bars,
                "notes": notes,
            }
        )
    order = {
        key: index
        for index, key in enumerate(item["id"] for item in data["configurations"])
    }
    costs = []
    for section in sections:
        rows = sorted(
            (
                record
                for record in data["cost_results"]["rows"]
                if record["benchmark_id"] == section["id"]
            ),
            key=lambda record: (
                order.get(record["configuration_id"], math.inf),
                record["id"],
            ),
        )
        if rows:
            costs.append(
                {
                    "id": section["id"],
                    "name": section["name"],
                    "token_metrics": [
                        {"key": key, "label": COPY[language][key]}
                        for key in ("total_output_tokens", "mean_total_tokens")
                        if any(record.get(key) is not None for record in rows)
                    ],
                    "rows": [
                        {
                            **record,
                            "label": label(configs[record["configuration_id"]]),
                            "detail": detail(configs[record["configuration_id"]]),
                        }
                        for record in rows
                    ],
                }
            )
    return {
        "sections": sections,
        "costs": costs,
        "language": language,
        "copy": COPY[language],
    }


class LeaderboardDirective(SphinxDirective):
    """Render an accessible fallback and mount the locally hosted renderer."""

    required_arguments = 1

    def run(self) -> list[nodes.Node]:
        section = self.arguments[0]
        if section not in ("performance", "time-token-costs"):
            raise self.error(f"Unknown leaderboard section: {section}")
        self.env.note_dependency(str(ASSETS / "results.json"))
        self.env.note_dependency(
            str(Path(__file__).parent / "templates" / "leaderboard.html")
        )
        language = "zh" if self.config.language.startswith("zh") else "en"
        values = context(load_data(), language)
        values["section"] = section
        values["asset_base"] = relative_uri(
            self.env.app.builder.get_target_uri(self.env.docname),
            "_static/leaderboard/",
        )
        html = TEMPLATES.get_template("leaderboard.html").render(**values)
        return [nodes.raw("", html, format="html")]


def write_images(app: Sphinx, exception: Exception | None) -> None:
    """Publish README chart images alongside each successful HTML build."""
    if exception is not None or app.builder.format != "html":
        return
    output = Path(app.outdir) / "_static" / "leaderboard"
    output.mkdir(parents=True, exist_ok=True)
    data = load_data()
    for language in COPY:
        values = context(data, language)
        panels = values["sections"]
        y = 100
        for offset in range(0, len(panels), 2):
            pair = panels[offset : offset + 2]
            height = 70 + max(len(panel["bars"]) for panel in pair) * 46
            for index, panel in enumerate(pair):
                panel.update(x=30 + index * 640, y=y, height=height)
            y += height + 28
        values["height"] = y + 45
        for theme in ("light", "dark"):
            values["dark"] = theme == "dark"
            values["palette"] = data["palette"][theme]
            svg = TEMPLATES.get_template("leaderboard.svg").render(**values)
            (output / f"leaderboard-{language}-{theme}.svg").write_text(
                svg, encoding="utf-8"
            )


def setup(app: Sphinx) -> dict:
    """Register the directive and derived image output for both documentation trees."""
    app.add_directive("rpent-leaderboard", LeaderboardDirective)
    app.connect("build-finished", write_images)
    return {
        "version": "3",
        "env_version": 3,
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
