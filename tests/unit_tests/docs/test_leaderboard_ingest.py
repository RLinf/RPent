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

"""Unit tests for the standalone leaderboard ingestion and validation module."""

from __future__ import annotations

import ast
import copy
import difflib
import json
import runpy
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
INGEST_PATH = REPO_ROOT / "docs/_ext/leaderboard_ingest.py"
M = runpy.run_path(str(INGEST_PATH))
ERR = M["LeaderboardValidationError"]


def _w(path: Path, obj: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


def _bi(s: str) -> dict[str, str]:
    return {"en": s, "zh": s}


def _res(vid: str, cid: str, rate: Any, src: str = "s1", **kw: Any) -> dict[str, Any]:
    base = {"id": f"{vid}--{cid}", "view_id": vid, "configuration_id": cid}
    return {**base, "rate": rate, "source_ids": [src], **kw}


def _cost(bid: str, cid: str, el: Any, tok: Any, **kw: Any) -> dict[str, Any]:
    base = {"id": f"{bid}--{cid}", "benchmark_id": bid, "configuration_id": cid}
    return {**base, "mean_elapsed_seconds": el, "total_output_tokens": tok, **kw}


def _dataset() -> dict[str, Any]:
    rt_note = {"id": "n1", "configuration_ids": ["cfg-rpent"], "text": _bi("N")}
    rc_views = [
        {"id": "robocasa", "label": _bi("Overall")},
        {"id": "robocasa-atomic-seen", "label": _bi("A")},
    ]
    rt_b = {
        "id": "robotwin",
        "name": "RoboTwin",
        "heading": _bi("C2R"),
        "default_view": "robotwin",
        "views": [{"id": "robotwin", "label": _bi("C2R")}],
        "notes": [rt_note],
    }
    rc_b = {
        "id": "robocasa",
        "name": "RoboCasa",
        "heading": _bi("T50"),
        "default_view": "robocasa",
        "views": rc_views,
    }
    return {
        "schema_version": 4,
        "footer_notes": [{"id": "fn1", "text": _bi("Footer")}],
        "configurations": [
            {"id": "cfg-rpent", "kind": "rpent", "backend": "API", "model": "M1"},
            {"id": "cfg-ext", "kind": "external", "model": "Ext1"},
        ],
        "benchmarks": [rt_b, rc_b],
        "results": [_res("robotwin", "cfg-rpent", "50.00", successes=1, episodes=2)],
        "cost_results": {
            "source_ids": ["s1"],
            "rows": [_cost("robotwin", "cfg-rpent", 12.5, 1000)],
        },
        "sources": [{"id": "s1", "note": "English-only source note."}],
        "palette": {
            "light": {"best": "#1", "rpent": "#2", "reference": "#3"},
            "dark": {"best": "#4", "rpent": "#5", "reference": "#6"},
        },
    }


def _tr(
    task: str,
    seed: int,
    ok: Any = True,
    err: Any = None,
    with_err: bool = True,
    el: float = 10.0,
    tok: int = 20,
    rt_seeds: bool = False,
) -> dict[str, Any]:
    st = {
        "total_input_tokens": 100,
        "total_output_tokens": tok,
        "total_cached_input_tokens": 40,
    }
    sd_fields = (
        {"requested_seed": seed, "initial_native_seed": seed}
        if rt_seeds
        else {"seed": seed}
    )
    rec: dict[str, Any] = {
        "task_name": task,
        "task_config": "demo_randomized",
        **sd_fields,
        "elapsed_s": el,
        "environment_success": ok,
        "stats": st,
    }
    if with_err:
        rec["error"] = err
    return rec


def test_module_uses_only_stdlib():
    tree = ast.parse(INGEST_PATH.read_text(encoding="utf-8"))
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods.update(a.name.split(".")[0] for a in n.names)
        elif isinstance(n, ast.ImportFrom) and n.module:
            mods.add(n.module.split(".")[0])
    assert mods <= sys.stdlib_module_names


def test_real_results_roundtrip_and_compact_upsert():
    raw = M["DATA_PATH"].read_text(encoding="utf-8")
    data = M["load_data"](M["DATA_PATH"])
    assert M["dump_data"](data) == raw

    cfg = {"id": "sim-api", "kind": "rpent", "backend": "API", "model": "Sim"}
    cfg["insert_after"] = "deepseek-v4-1-flash-api-xhigh"
    res = _res("robotwin", "sim-api", "60.00", "src-sim", successes=150, episodes=250)
    cost = _cost("robotwin", "sim-api", 123.45, 500000, source_ids=["src-sim"])
    payload = {
        "sources": [{"id": "src-sim", "note": "Synthetic API run."}],
        "configurations": [cfg],
        "results": [res],
        "cost_results": [cost],
    }
    after = M["dump_data"](M["upsert_data"](data, payload))
    diff = list(difflib.unified_diff(raw.splitlines(), after.splitlines(), lineterm=""))
    rem = [ln for ln in diff if ln.startswith("-") and not ln.startswith("---")]
    add = [ln for ln in diff if ln.startswith("+") and not ln.startswith("+++")]
    assert len(rem) <= 3 and len(add) < 25


def test_validate_rules_a1_and_cost_guards():
    base = _dataset()
    M["validate_data"](base)
    with_none = copy.deepcopy(base)
    with_none["results"][0]["rate"] = None
    with_none["cost_results"]["rows"][0]["mean_elapsed_seconds"] = None
    M["validate_data"](with_none)

    for patch in (
        {"successes_estimated": True},
        {"status": "derived"},
        {"id": "robocasa--cfg-rpent", "view_id": "robocasa"},
    ):
        ex = copy.deepcopy(base)
        ex["results"][0].update(successes=1, episodes=3, rate="99.00", **patch)
        M["validate_data"](ex)

    def _clr(cfg: dict[str, Any], cid: str, kind: str) -> None:
        cfg.clear()
        cfg.update(id=cid, kind=kind)

    bad_rates = ("97", "97.00\n", "100.01", 50.0, True)
    bad_nums = (True, "12", -1, float("nan"))
    bad = [
        lambda d: d.update(schema_version=True),
        lambda d: d["benchmarks"][0].pop("name"),
        lambda d: d["results"][0].pop("rate"),
        lambda d: d["cost_results"]["rows"][0].pop("mean_elapsed_seconds"),
        lambda d: d["benchmarks"][0].update(default_view=[]),
        lambda d: d["results"][0].update(view_id=[]),
        lambda d: d["results"][0].update(derived_from=[[]]),
        lambda d: d["results"][0].update(derived_from=None),
        lambda d: d["cost_results"]["rows"][0].update(benchmark_id=[]),
        lambda d: d["palette"]["dark"].pop("reference"),
        lambda d: _clr(d["configurations"][0], "cfg-rpent", "rpent"),
        lambda d: _clr(d["configurations"][1], "cfg-ext", "external"),
        lambda d: d["benchmarks"][0]["heading"].update(zh=" "),
        lambda d: d["benchmarks"][0].update(summary=123),
        lambda d: d["results"][0].update(evaluation_note={"en": "ok", "zh": ""}),
        lambda d: d["results"][0].update(evaluation_note=""),
        lambda d: d["results"][0].update(successes=1, episodes=3, rate="33.34"),
        *(lambda d, r=r: d["results"][0].update(rate=r) for r in bad_rates),
        *(
            lambda d, v=v: d["cost_results"]["rows"][0].update(total_output_tokens=v)
            for v in bad_nums
        ),
    ]
    for mutate in bad:
        broken = copy.deepcopy(base)
        mutate(broken)
        with pytest.raises(ERR):
            M["validate_data"](broken)


def test_robotwin_append_and_aggregation(tmp_path: Path):
    mdata = {
        "task_config": "demo_randomized",
        "tasks": {"a": [{"seed": 1}, {"seed": 2}], "b": [{"seed": 3}]},
    }
    mf = _w(tmp_path / "mf.json", mdata)
    run = tmp_path / "rt"
    run.mkdir()
    t1 = run / "transcript_a_s1.json"
    p1 = json.dumps(_tr("a", 1, False, "HTTP 500", rt_seeds=True))
    p2 = json.dumps(_tr("a", 1, with_err=False, el=10.0, tok=100, rt_seeds=True))
    t1.write_text(p1 + p2, encoding="utf-8")
    final_rec, retries = M["load_transcript_final_record"](t1)
    assert final_rec["environment_success"] is True and retries == 1
    _w(
        run / "transcript_a_s2.json",
        _tr(
            "a", 2, False, "planner response truncated", el=20.0, tok=200, rt_seeds=True
        ),
    )
    _w(
        run / "transcript_b_s3.json",
        _tr(
            "b",
            3,
            False,
            "planner timed out after 120s",
            el=30.0,
            tok=300,
            rt_seeds=True,
        ),
    )

    kw = {
        "manifest_path": mf,
        "configuration_id": "cfg-rpent",
        "source_ids": ["s1"],
        "protocol_id": "p",
    }
    agg = M["aggregate_robotwin"](run, **kw)
    res, cost = agg["results"][0], agg["cost_results"][0]
    assert (res["rate"], res["task_failures"], res["episode_timeouts"]) == (
        "33.33",
        1,
        1,
    )
    assert "mean_total_tokens" not in cost
    assert (cost["mean_elapsed_seconds"], cost["total_output_tokens"]) == (20.0, 600)
    assert agg["token_summary"]["excluded_prior_attempts"] == 1

    out_f = tmp_path / "agg_rt.json"
    cli = [
        "aggregate",
        "--benchmark",
        "robotwin",
        "--input-dir",
        str(run),
        "--manifest",
        str(mf),
    ]
    assert M["main"]([*cli, "--output", str(out_f)]) == 0
    assert (
        json.loads(out_f.read_text(encoding="utf-8"))["results"][0]["rate"] == "33.33"
    )
    assert M["main"]([*cli, "--planner", "codex"]) == 1

    bad_sd = {**_tr("b", 3, rt_seeds=True), "initial_native_seed": 9}
    bad_pl = {**_tr("b", 3, rt_seeds=True), "planner": "codex"}
    bad_recs = (
        _tr("b", 3, False, "ModelHTTPError: status_code: 500", rt_seeds=True),
        _tr("b", 3, False, 500, rt_seeds=True),
        _tr("b", 3, None, rt_seeds=True),
        bad_sd,
        bad_pl,
    )
    for bad in bad_recs:
        _w(run / "transcript_b_s3.json", bad)
        with pytest.raises(ValueError):
            M["aggregate_robotwin"](run, manifest_path=mf)


def test_robocasa_four_views_stray_guards_and_cli(tmp_path: Path):
    sp = lambda sc: {"successes": sc, "valid_cells": 1, "expected_cells": 1}  # noqa: E731
    sdata = {
        "protocol_id": "robocasa-harness-vla-v2",
        "valid_cells": 3,
        "expected_cells": 3,
        "splits": {"atomic": sp(1), "composite_seen": sp(1), "composite_unseen": sp(0)},
        "overall": {
            "metric": "task_weighted_success_rate",
            "tasks": 3,
            "success_rate": 0.666667,
        },
    }
    summary = _w(tmp_path / "summary.json", sdata)
    cells = (
        ("atomic", "T1", 10.0, 100),
        ("composite_seen", "T2", 20.0, 200),
        ("composite_unseen", "T3", 30.0, 300),
    )
    mf = _w(
        tmp_path / "t50.json",
        {"splits": {s: {"tasks": [t], "seeds": [1]} for s, t, _, _ in cells}},
    )
    root = tmp_path / "rc"
    for split, task, el, tok in cells:
        _w(root / split / f"{task}_s1/transcript_rc.json", _tr(task, 1, el=el, tok=tok))

    kw = {"manifest_path": mf, "configuration_id": "cfg-rpent", "source_ids": ["s1"]}
    agg = M["aggregate_robocasa"](root, summary, **kw)
    rates = {r["view_id"]: r["rate"] for r in agg["results"]}
    assert rates == {
        "robocasa": "66.67",
        "robocasa-atomic-seen": "100.00",
        "robocasa-composite-seen": "100.00",
        "robocasa-composite-unseen": "0.00",
    }
    assert "mean_total_tokens" not in agg["cost_results"][0]
    assert agg["cost_results"][0]["total_output_tokens"] == 600
    cli = [
        "aggregate",
        "--benchmark",
        "robocasa",
        "--input-dir",
        str(root),
        "--target50-summary",
        str(summary),
    ]
    assert M["main"]([*cli, "--manifest", str(mf)]) == 0
    assert M["main"]([*cli, "--planner", "claude_code"]) == 1

    assert M["ROBOCASA_MANIFEST_PATH"].is_file()
    with pytest.raises(ValueError):
        M["aggregate_robocasa"](root, summary)  # default manifest expects 340 cells

    _w(root / "atomic/T1_s1/result.json", {"planner": {"backend": "codex"}})
    with pytest.raises(ValueError, match="Non-API"):
        M["aggregate_robocasa"](root, summary, manifest_path=mf)
    (root / "atomic/T1_s1/result.json").unlink()

    _w(root / "atomic/T1_s1/transcript_rc.json", _tr("Wrong", 1))
    with pytest.raises(ValueError, match="!="):
        M["aggregate_robocasa"](root, summary, manifest_path=mf)

    (root / "composite_unseen/T3_s1/transcript_rc.json").unlink()
    _w(root / "stray/transcript_s.json", _tr("T3", 1))
    with pytest.raises(ValueError):
        M["aggregate_robocasa"](root, summary, manifest_path=mf)


def test_upsert_anchor_no_overwrite_and_cli(tmp_path: Path, capsys):
    ds = _dataset()
    ds["benchmarks"][1]["notes"] = []
    ds["sources"][0]["extra"] = [{"id": {"nested": 1}}]
    data_file = tmp_path / "results.json"
    M["write_data"](data_file, ds)
    assert M["main"](["validate", "--data", str(data_file)]) == 0
    assert "OK:" in capsys.readouterr().out

    cfg2 = {
        "id": "cfg-rpent-2",
        "kind": "rpent",
        "model": "M2",
        "insert_after": "cfg-rpent",
    }
    note = {"id": "rc-n1", "configuration_ids": ["cfg-rpent"], "text": _bi("RC")}
    pdata = {
        "sources": [{"id": "s2", "note": "Run 2"}],
        "configurations": [cfg2],
        "benchmark_notes": {"robocasa": [note]},
        "results": [_res("robocasa", "cfg-rpent", "55.00", "s2", episodes=340)],
        "cost_results": [_cost("robocasa", "cfg-rpent", 40.0, 500, source_ids=["s2"])],
    }
    payload = _w(tmp_path / "payload.json", pdata)
    before = data_file.read_text(encoding="utf-8")
    base = ["upsert", "--data", str(data_file), "--payload", str(payload)]
    assert M["main"]([*base, "--dry-run"]) == 0
    assert '+    {"id": "robocasa--cfg-rpent"' in capsys.readouterr().out
    assert data_file.read_text(encoding="utf-8") == before

    assert M["main"]([*base, "--in-place"]) == 0
    assert '\n          "id": "rc-n1",' in data_file.read_text(encoding="utf-8")
    updated = M["load_data"](data_file)
    assert [c["id"] for c in updated["configurations"]] == [
        "cfg-rpent",
        "cfg-rpent-2",
        "cfg-ext",
    ]
    assert M["main"]([*base, "--in-place"]) == 1
    assert "error:" in capsys.readouterr().err

    with pytest.raises(ERR):
        M["upsert_data"]({"schema_version": 3}, {})
    bad_u = {"results": [{"id": "x", "view_id": [], "configuration_id": "cfg-rpent"}]}
    with pytest.raises(ERR):
        M["upsert_data"](updated, bad_u)
    assert "s2" in M["dump_data"](
        M["upsert_data"](_dataset(), {"sources": [{"id": "s2", "note": "x"}]})
    )
    for flag in ("--allow-update-config", "--json"):
        with pytest.raises(SystemExit):
            M["main"]([*base, "--dry-run", flag])
