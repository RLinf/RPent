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

"""Scientific-data regressions for the benchmark documentation source in CI."""

from copy import deepcopy

import pytest

from docs.tools import benchmark_data


def record(data, record_id):
    return next(item for item in data["results"] if item["id"] == record_id)


def view(data, view_id):
    return next(item for item in data["views"] if item["id"] == view_id)


def test_missing_result_cannot_be_zero():
    data = benchmark_data.load_data()
    record(data, "libero-pro--gpt-6-astra-low")["rate"] = "0.0"
    with pytest.raises(ValueError, match="Unreported results must have null"):
        benchmark_data.validate_data(data)


def test_counts_must_agree_at_reported_precision():
    data = benchmark_data.load_data()
    record(data, "long-task--gpt-6-astra-low")["successes"] = 84
    with pytest.raises(ValueError, match="Count and displayed rate disagree"):
        benchmark_data.validate_data(data)


def test_six_item_baseline_cannot_enter_full_pro_chart():
    data = benchmark_data.load_data()
    view(data, "libero-pro")["chart_record_ids"].append("libero-pro-six-non-long--rats")
    with pytest.raises(ValueError, match="cross-scope record"):
        benchmark_data.validate_data(data)


def test_partial_astra_results_cannot_define_full_overall():
    data = benchmark_data.load_data()
    aggregate = record(data, "libero-pro--gpt-6-astra-low")
    aggregate.update(status="reported", rate="78.5", episodes=800)
    view(data, "libero-pro")["chart_record_ids"].append(aggregate["id"])
    view(data, "libero-pro")["overview_record_ids"].append(aggregate["id"])
    with pytest.raises(ValueError, match="Incomplete coverage cannot define Overall"):
        benchmark_data.validate_data(data)


def test_overall_scope_cannot_be_redefined_as_long_only():
    data = benchmark_data.load_data()
    aggregate_view = view(data, "libero-pro")
    aggregate_view["aggregate_of"] = ["long-task", "long-swap"]
    for cid, rate in (("gpt-5-5-xhigh", "50.5"), ("opus-4-8-max", "66.5")):
        record(data, f"libero-pro--{cid}").update(rate=rate, episodes=200)
    aggregate_view["axis_max"] = 70
    with pytest.raises(ValueError, match="Overall must cover every other item"):
        benchmark_data.validate_data(data)


def test_full_overall_requires_full_denominator():
    data = benchmark_data.load_data()
    record(data, "libero-pro--gpt-5-5-xhigh")["episodes"] = 600
    with pytest.raises(ValueError, match="Overall denominator differs"):
        benchmark_data.validate_data(data)


def test_duplicate_model_result_cannot_replace_the_original():
    data = benchmark_data.load_data()
    duplicate = deepcopy(record(data, "long-swap--gpt-6-astra-low"))
    duplicate["id"] += "-other-attempt"
    data["results"].append(duplicate)
    with pytest.raises(ValueError, match="Duplicate model within an evaluation scope"):
        benchmark_data.validate_data(data)


def test_new_score_updates_fixed_axis_and_preserves_precision():
    data = benchmark_data.load_data()
    new = record(data, "spatial-task--gpt-6-astra-low")
    new.update(status="reported", rate="100.0", successes=100, episodes=100)
    scope = view(data, "spatial-task")
    scope["chart_record_ids"].append(new["id"])
    scope["axis_max"] = 100
    benchmark_data.validate_data(data)
    table = next(item for item in data["tables"] if item["id"] == "libero-pro")
    for language in ("en", "zh"):
        assert "100.0% (100/100)" in benchmark_data.render_table(data, table, language)
        assert "96.0% (384/400)" not in benchmark_data.render_table(
            data, table, language
        )


def test_new_model_adds_a_column_without_reordering_existing_models():
    data = benchmark_data.load_data()
    data["configurations"].append(
        {
            "id": "future-model",
            "kind": "rpent",
            "backend": "Future backend",
            "model": "Future model",
            "reasoning": False,
            "effort": None,
        }
    )
    for table in data["tables"]:
        if table["kind"] != "main":
            continue
        for vid in table["view_ids"]:
            previous = record(data, f"{vid}--gpt-6-astra-low")
            new = {
                **previous,
                "id": f"{vid}--future-model",
                "configuration_id": "future-model",
                "status": "not_reported",
                "rate": None,
                "successes": None,
                "episodes": None,
            }
            data["results"].append(new)
            view(data, vid)["record_ids"].append(new["id"])
    benchmark_data.validate_data(data)
    table = next(item for item in data["tables"] if item["id"] == "libero-pro")
    rendered = benchmark_data.render_table(data, table, "en")
    assert rendered.index("GPT-5.5 / ``xhigh``") < rendered.index("Opus-4.8 / ``max``")
    assert rendered.index("Opus-4.8 / ``max``") < rendered.index(
        "GPT-6 Astra / ``low``"
    )
    assert rendered.index("GPT-6 Astra / ``low``") < rendered.index("Future model")
    assert ":widths: 31 23 23 23 23" in rendered


def test_stale_axis_is_rejected():
    data = benchmark_data.load_data()
    view(data, "robotwin")["axis_max"] = 100
    with pytest.raises(ValueError, match="Stale axis limit"):
        benchmark_data.validate_data(data)
