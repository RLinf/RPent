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

_RATE_RE = re.compile(r"\A\d{1,3}\.\d{2}\Z")
_TOK_RE = re.compile(r'"(?:\\.|[^"\\])*"|[{}\[\]:,]|[^\s{}\[\]:,]+')
_COST_NUMS = ("mean_elapsed_seconds", "mean_total_tokens", "total_output_tokens")


class LeaderboardValidationError(ValueError):
    """Raised when leaderboard data fails schema or consistency validation."""


class _Dataset(dict):
    _styles: dict[tuple[str, ...], bool] | None = None


def _must(ok: bool, msg: str, exc: type[ValueError] = ValueError) -> None:
    if not ok:
        raise exc(msg)


def _chk(ok: bool, msg: str) -> None:
    _must(ok, f"Leaderboard: {msg}", LeaderboardValidationError)


def _int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _num(v: Any) -> bool:
    ok = isinstance(v, (int, float)) and not isinstance(v, bool)
    return ok and math.isfinite(v) and v >= 0


def _str(v: Any) -> bool:
    return isinstance(v, str) and bool(v.strip())


def _q2(num: int | float | Decimal, den: int = 1, scale: int = 1) -> Decimal:
    v = (Decimal(str(num)) * scale) / Decimal(den)
    return v.quantize(Decimal("0.01"), ROUND_HALF_UP)


def validate_data(data: dict[str, Any]) -> None:
    """Validate leaderboard dataset schema, references, and numeric invariants."""
    _chk(isinstance(data, dict), "top-level JSON must be an object")
    sv = data.get("schema_version")
    _chk(_int(sv) and sv == 4, f"expected schema_version 4, got {sv!r}")

    def idx(items: Any, kind: str) -> dict[str, dict[str, Any]]:
        _chk(isinstance(items, list), f"{kind} must be a list")
        out: dict[str, dict[str, Any]] = {}
        for x in items:
            _chk(isinstance(x, dict) and _str(x.get("id")), f"bad id in {kind}")
            _chk(x["id"] not in out, f"duplicate ID {x['id']!r} in {kind}")
            out[x["id"]] = x
        return out

    def bi(val: Any, ctx: str, req: bool = True) -> None:
        if val is None and not req:
            return
        if not isinstance(val, dict):
            _chk(not req and _str(val), f"bad {ctx}")
            return
        _chk(all(_str(val.get(k)) for k in ("en", "zh")), f"bad en/zh in {ctx}")

    def refs(ids: Any, pool: dict[str, Any], ctx: str, nonempty: bool = True) -> None:
        _chk(isinstance(ids, list) and (bool(ids) or not nonempty), f"bad {ctx}")
        for k in ids:
            _chk(isinstance(k, str) and k in pool, f"unknown {k!r} in {ctx}")

    configs = idx(data.get("configurations"), "configurations")
    benchmarks = idx(data.get("benchmarks"), "benchmarks")
    all_views, all_notes = [], []
    for bid, b in benchmarks.items():
        _chk(_str(b.get("name")), f"missing name in {bid}")
        bi(b.get("heading"), f"benchmarks[{bid}].heading")
        bi(b.get("summary"), f"benchmarks[{bid}].summary", False)
        bv, bn = b.get("views"), b.get("notes", [])
        _chk(isinstance(bv, list) and bool(bv) and isinstance(bn, list), f"bad {bid}")
        lv, dv = idx(bv, f"views[{bid}]"), b.get("default_view")
        _chk(isinstance(dv, str) and dv in lv, f"bad default_view in {bid}")
        for v in bv:
            bi(v.get("label"), f"views[{v['id']}].label")
        all_views.extend(bv)
        all_notes.extend(bn)

    views, results = idx(all_views, "views"), idx(data.get("results"), "results")
    costs = data.get("cost_results")
    _chk(isinstance(costs, dict), "cost_results must be an object")
    idx(costs.get("rows"), "cost_results")
    sources = idx(data.get("sources"), "sources")
    fnotes = data.get("footer_notes", [])
    _chk(isinstance(fnotes, list), "footer_notes must be a list")
    idx(all_notes + fnotes, "notes")

    pal = data.get("palette")
    _chk(isinstance(pal, dict), "palette must be an object")
    roles = ("best", "rpent", "reference")
    for t in ("light", "dark"):
        tm = pal.get(t)
        _chk(isinstance(tm, dict) and all(_str(tm.get(r)) for r in roles), f"bad {t}")

    for cid, cfg in configs.items():
        k = cfg.get("kind")
        _chk(k in ("rpent", "external"), f"invalid kind {k!r} in {cid}")
        req = ("display_name", "model", *(["backend"] if k == "rpent" else []))
        _chk(any(_str(cfg.get(f)) for f in req), f"no label in {cid}")

    for n in all_notes + fnotes:
        bi(n.get("text"), f"notes[{n['id']}].text")
    for n in all_notes:
        refs(n.get("configuration_ids"), configs, n["id"], False)
    refs(costs.get("source_ids"), sources, "cost_results.source_ids")

    specs = (
        (data["results"], "view_id", views, None, True),
        (costs["rows"], "benchmark_id", benchmarks, costs["source_ids"], False),
    )
    for coll, gkey, groups, def_srcs, is_res in specs:
        pairs: set[tuple[str, str]] = set()
        for r in coll:
            rid, cid, gid = r["id"], r.get("configuration_id"), r.get(gkey)
            _chk(isinstance(cid, str) and cid in configs, f"bad config in {rid}")
            _chk(isinstance(gid, str) and gid in groups, f"bad {gkey} in {rid}")
            _chk((gid, cid) not in pairs, f"duplicate ({gid}, {cid})")
            pairs.add((gid, cid))
            refs(r.get("source_ids", def_srcs), sources, f"sources in {rid}")
            if "derived_from" in r:
                refs(r["derived_from"], results, f"derived {rid}", False)
            for k in _COST_NUMS:
                _chk(r.get(k) is None or _num(r[k]), f"invalid {k} in {rid}")
            rate = r.get("rate")
            if rate is not None:
                ok_r = isinstance(rate, str) and _RATE_RE.fullmatch(rate) is not None
                _chk(ok_r, f"bad rate {rate!r} in {rid}")
                _chk(Decimal(0) <= Decimal(rate) <= Decimal(100), f"rate range: {rid}")
            if not is_res:
                _chk("mean_elapsed_seconds" in r, f"missing elapsed in {rid}")
                continue
            _chk("rate" in r, f"missing rate key in {rid}")
            ep, sc = r.get("episodes"), r.get("successes")
            _chk(ep is None or (_int(ep) and ep >= 0), f"bad episodes in {rid}")
            _chk(sc is None or (_int(sc) and sc >= 0), f"bad successes in {rid}")
            if _int(sc) and _int(ep):
                _chk(sc <= ep, f"successes > episodes in {rid}")
                est = r.get("successes_estimated") is True
                skip = rate is None or ep == 0 or est or r.get("status") == "derived"
                if not skip and gid != "robocasa":
                    exp = str(_q2(sc, ep, 100))
                    _chk(rate == exp, f"rate {rate!r} != {exp!r} in {rid}")
            bi(r.get("evaluation_note"), f"results[{rid}].note", False)


def _scan_styles(raw: str) -> dict[tuple[str, ...], bool]:
    toks, styles = list(_TOK_RE.finditer(raw)), {}

    def walk(i: int, p: tuple[str, ...]) -> int:
        c = toks[i].group(0)
        if c == "{":
            i += 1
            while toks[i].group(0) != "}":
                i = walk(i + 2, (*p, json.loads(toks[i].group(0))))
                i += toks[i].group(0) == ","
            return i + 1
        if c == "[":
            s, i, n = toks[i].start(), i + 1, 0
            while toks[i].group(0) != "]":
                ok = (
                    toks[i].group(0) == "{"
                    and i + 3 < len(toks)
                    and toks[i + 1].group(0) == '"id"'
                    and toks[i + 3].group(0).startswith('"')
                )
                seg = f"id={json.loads(toks[i + 3].group(0))}" if ok else str(n)
                i, n = walk(i, (*p, seg)), n + 1
                i += toks[i].group(0) == ","
            if n > 0:
                styles[p] = "\n" not in raw[s : toks[i].end()]
            return i + 1
        return i + 1

    if toks:
        walk(0, ())
    return styles


def load_data(path: Path | str = DATA_PATH) -> dict[str, Any]:
    """Read, parse, and validate a leaderboard ``results.json`` file."""
    raw = Path(path).read_text(encoding="utf-8")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LeaderboardValidationError(f"Leaderboard: invalid JSON: {exc}") from exc
    validate_data(parsed)
    ds = _Dataset(parsed)
    ds._styles = _scan_styles(raw)
    return ds


def dump_data(data: dict[str, Any]) -> str:
    """Serialize leaderboard data preserving compact row and array layout."""
    st = data._styles if isinstance(data, _Dataset) else None

    def j(x: Any) -> str:
        return json.dumps(x, ensure_ascii=False)

    def fmt(v: Any, ind: int, p: tuple[str, ...]) -> str:
        pad, sub = " " * ind, " " * (ind + 2)
        if isinstance(v, dict):
            if not v:
                return "{}"
            rows = [f"{sub}{j(k)}: {fmt(x, ind + 2, (*p, k))}" for k, x in v.items()]
            return "{\n" + ",\n".join(rows) + f"\n{pad}}}"
        if isinstance(v, list):
            if not v:
                return "[]"
            if p in {("results",), ("cost_results", "rows")}:
                return "[\n" + ",\n".join(f"{sub}{j(x)}" for x in v) + f"\n{pad}]"
            prim = all(not isinstance(x, (dict, list)) for x in v)
            ok = bool(p) and p[-1] in {"source_ids", "configuration_ids"}
            dflt = ok and all(isinstance(x, str) for x in v) and len(j(v)) <= 88
            if prim and (st[p] if (st is not None and p in st) else dflt):
                return j(v)
            items = []
            for i, x in enumerate(v):
                has_id = isinstance(x, dict) and isinstance(x.get("id"), str)
                items.append(
                    sub + fmt(x, ind + 2, (*p, "id=" + x["id"] if has_id else str(i)))
                )
            return "[\n" + ",\n".join(items) + f"\n{pad}]"
        return j(v)

    return fmt(data, 0, ()) + "\n"


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False, newline="\n"
        ) as fh:
            tmp = Path(fh.name)
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
        raise


def write_data(path: Path | str, data: dict[str, Any]) -> None:
    """Validate and atomically write leaderboard data preserving layout."""
    validate_data(data)
    _atomic_write(Path(path), dump_data(data))


def load_transcript_final_record(path: Path | str) -> tuple[dict[str, Any], int]:
    """Read an append-mode ``transcript_*.json`` and return ``(final, retries)``."""
    raw = Path(path).read_text(encoding="utf-8")
    dec, i, end, recs = json.JSONDecoder(), 0, len(raw), []
    while i < end:
        while i < end and raw[i].isspace():
            i += 1
        if i >= end:
            break
        try:
            obj, i = dec.raw_decode(raw, i)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSON in {path}: {exc}") from exc
        _must(isinstance(obj, dict), f"Transcript entry in {path} must be an object")
        recs.append(obj)
    _must(bool(recs), f"Empty transcript file: {path}")
    return recs[-1], len(recs) - 1


def _meta(
    body: dict[str, Any], pfx: str, cid: str | None, srcs: Any, pid: Any = None
) -> dict[str, Any]:
    out = ({"id": f"{pfx}--{cid}"} if cid is not None else {}) | body
    for k, v in (("configuration_id", cid), ("source_ids", srcs), ("protocol_id", pid)):
        if v is not None:
            out[k] = list(v) if k == "source_ids" else v
    return out


def _sum_usage(
    entries: list[tuple[Path, dict[str, Any], int]],
    bid: str,
    cid: str | None,
    srcs: list[str] | None,
    extra: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    el_s, it_s, ot_s, ct_s, rt_s = Decimal(0), 0, 0, 0, 0
    for path, rec, retries in entries:
        el, st = rec.get("elapsed_s"), rec.get("stats")
        _must(_num(el) and isinstance(st, dict), f"Bad elapsed_s/stats in {path}")
        for obj in (rec, st):
            for k in ("planner", "backend"):
                _must(obj.get(k) in (None, "api"), f"Non-API {k} in {path}")
        it, ot = st.get("total_input_tokens"), st.get("total_output_tokens")
        ct = st.get("total_cached_input_tokens", 0)
        _must(all(_int(x) and x >= 0 for x in (it, ot, ct)), f"Bad tokens: {path}")
        el_s += Decimal(str(el))
        it_s, ot_s, ct_s, rt_s = it_s + it, ot_s + ot, ct_s + ct, rt_s + retries
    n = len(entries)
    mean_el, mean_tok = float(_q2(el_s, n)), float(_q2(it_s + ot_s, n))
    c_body = {"benchmark_id": bid, "mean_elapsed_seconds": mean_el}
    c_body["total_output_tokens"] = ot_s
    tok_summary = {
        "episodes": n,
        **(extra or {}),
        "excluded_prior_attempts": rt_s,
        "mean_elapsed_seconds": mean_el,
        "total_input_tokens": it_s,
        "total_output_tokens": ot_s,
        "total_cached_input_tokens": ct_s,
        "mean_total_tokens": mean_tok,
    }
    return [_meta(c_body, bid, cid, srcs)], tok_summary


def aggregate_robotwin(
    input_dir: Path | str,
    *,
    manifest_path: Path | str = ROBOTWIN_MANIFEST_PATH,
    planner: str = "api",
    configuration_id: str | None = None,
    source_ids: list[str] | None = None,
    protocol_id: str | None = None,
) -> dict[str, Any]:
    """Aggregate RoboTwin C2R ``api`` transcripts against ``demo_randomized.json``."""
    _must(planner == "api", f"Unsupported planner {planner!r}: only 'api' supported")
    mf = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    tasks, exp_cfg = mf.get("tasks"), mf.get("task_config")
    _must(isinstance(tasks, dict) and bool(tasks), f"Invalid manifest: {manifest_path}")
    expected: set[tuple[str, int]] = set()
    for tname, items in tasks.items():
        _must(isinstance(items, list), f"Invalid seeds for {tname!r}")
        for e in items:
            sd = e.get("seed") if isinstance(e, dict) else e
            _must(_int(sd) and (tname, sd) not in expected, f"Bad/dup {(tname, sd)}")
            expected.add((tname, sd))

    seen: set[tuple[str, int]] = set()
    entries: list[tuple[Path, dict[str, Any], int]] = []
    succ = fail = timeout = 0
    for path in sorted(Path(input_dir).rglob("transcript_*.json")):
        rec, retries = load_transcript_final_record(path)
        tname = rec.get("task_name")
        seeds = [
            rec[k]
            for k in ("requested_seed", "initial_native_seed", "seed")
            if k in rec and rec[k] is not None
        ]
        ok_sd = bool(seeds) and all(_int(v) for v in seeds) and len(set(seeds)) == 1
        _must(_str(tname) and ok_sd, f"Missing/inconsistent task_name/seed in {path}")
        sd = seeds[0]
        if exp_cfg is not None and rec.get("task_config") is not None:
            _must(rec["task_config"] == exp_cfg, f"Bad task_config in {path}")
        pair = (tname, sd)
        _must(pair in expected and pair not in seen, f"Bad/dup pair {pair} in {path}")
        seen.add(pair)
        env_ok = rec.get("environment_success")
        _must(isinstance(env_ok, bool), f"Non-bool environment_success in {path}")
        if env_ok:
            succ += 1
        else:
            err = rec.get("error")
            _must(err is None or isinstance(err, str), f"Non-str error in {path}")
            low = (err or "").strip().lower()
            if not low or "planner response truncated" in low:
                fail += 1
            elif "timed out" in low and ("planner" in low or "episode" in low):
                timeout += 1
            else:
                raise ValueError(f"Unretried infrastructure error in {path}: {err!r}")
        entries.append((path, rec, retries))

    missing = sorted(expected - seen)
    _must(not missing, f"Missing {len(missing)} RoboTwin episode(s): {missing[:3]}")
    ep = len(expected)
    extra = {"successes": succ, "task_failures": fail, "episode_timeouts": timeout}
    body = {"view_id": "robotwin", "rate": str(_q2(succ, ep, 100)), "episodes": ep}
    res = _meta(body | extra, "robotwin", configuration_id, source_ids, protocol_id)
    costs, tok = _sum_usage(entries, "robotwin", configuration_id, source_ids, extra)
    return {"results": [res], "cost_results": costs, "token_summary": tok}


def aggregate_robocasa(
    input_dir: Path | str,
    summary_path: Path | str,
    *,
    manifest_path: Path | str = ROBOCASA_MANIFEST_PATH,
    planner: str = "api",
    configuration_id: str | None = None,
    source_ids: list[str] | None = None,
    protocol_id: str | None = None,
) -> dict[str, Any]:
    """Aggregate RoboCasa Target50 results from ``validate_target50`` + transcripts."""
    _must(planner == "api", f"Unsupported planner {planner!r}: only 'api' supported")
    s = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    ok_s = isinstance(s, dict) and s.get("protocol_id") == "robocasa-harness-vla-v2"
    _must(ok_s, "Invalid Target50 protocol_id")
    exp_n, ov, splits = s.get("expected_cells"), s.get("overall"), s.get("splits")
    _must(_int(exp_n) and exp_n > 0 and s.get("valid_cells") == exp_n, "Bad summary")
    ok_ov = isinstance(ov, dict) and ov.get("metric") == "task_weighted_success_rate"
    _must(ok_ov and _num(ov.get("success_rate")) and ov["success_rate"] <= 1, "Bad ov")
    _must(isinstance(splits, dict), "Summary missing splits")

    cid, srcs, pid = configuration_id, source_ids, protocol_id
    ov_b = {"view_id": "robocasa", "rate": str(_q2(ov["success_rate"], 1, 100))}
    res_out = [_meta(ov_b | {"episodes": exp_n}, "robocasa", cid, srcs, pid)]
    split_exp: dict[str, int] = {}
    rc_views = (
        ("atomic", "robocasa-atomic-seen"),
        ("composite_seen", "robocasa-composite-seen"),
        ("composite_unseen", "robocasa-composite-unseen"),
    )
    for sn, vid in rc_views:
        sp = splits.get(sn)
        _must(isinstance(sp, dict), f"Missing split {sn!r}")
        sc, vn, en = (sp.get(k) for k in ("successes", "valid_cells", "expected_cells"))
        ok_sp = _int(en) and en > 0 and vn == en and _int(sc) and 0 <= sc <= en
        _must(ok_sp, f"Invalid split {sn!r}")
        split_exp[sn] = en
        sb = {"view_id": vid, "rate": str(_q2(sc, en, 100)), "successes": sc}
        res_out.append(_meta(sb | {"episodes": en}, vid, cid, srcs, pid))
    _must(sum(split_exp.values()) == exp_n, "Split expected_cells sum mismatch")

    root = Path(input_dir)
    all_t = sorted(root.rglob("transcript_*.json"))
    _must(len(all_t) == exp_n, f"Expected {exp_n} transcripts, got {len(all_t)}")
    ms = json.loads(Path(manifest_path).read_text(encoding="utf-8")).get("splits", {})
    entries: list[tuple[Path, dict[str, Any], int]] = []
    matched: set[Path] = set()
    for sn, en in split_exp.items():
        sp = ms.get(sn)
        _must(isinstance(sp, dict), f"Manifest missing split {sn!r}")
        cells = [(t, sd) for t in sp.get("tasks", []) for sd in sp.get("seeds", [])]
        _must(len(cells) == en, f"Manifest split {sn!r} cell mismatch")
        for task, seed in cells:
            cdir = root / sn / f"{task}_s{seed}"
            res_f = cdir / "result.json"
            if res_f.is_file():
                pl = json.loads(res_f.read_text(encoding="utf-8")).get("planner")
                if isinstance(pl, dict):
                    _must(pl.get("backend") in (None, "api"), f"Non-API {res_f}")
            found = sorted(cdir.glob("transcript_*.json"))
            _must(len(found) == 1, f"Expected 1 transcript in {cdir}")
            rec, retries = load_transcript_final_record(found[0])
            got = (rec.get("task_name"), rec.get("seed"))
            _must(got == (task, seed), f"Transcript {got!r} != {(task, seed)!r}")
            matched.add(found[0].resolve())
            entries.append((found[0], rec, retries))
    _must(matched == {p.resolve() for p in all_t}, "Found stray transcripts")
    costs, tok = _sum_usage(entries, "robocasa", cid, srcs)
    return {"results": res_out, "cost_results": costs, "token_summary": tok}


def upsert_data(data: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Append new validated entries to a dataset without overwriting existing data."""
    validate_data(data)
    keys = {"configurations", "results", "cost_results", "sources", "benchmark_notes"}
    _chk(isinstance(payload, dict) and set(payload) <= keys, "invalid upsert payload")
    updated = _Dataset(copy.deepcopy(dict(data)))
    if isinstance(data, _Dataset) and data._styles is not None:
        updated._styles = dict(data._styles)

    def req_new(item: Any, seen: set[str], label: str) -> None:
        _chk(isinstance(item, dict) and _str(item.get("id")), f"bad {label} entry")
        _chk(item["id"] not in seen, f"cannot overwrite {label} {item['id']!r}")
        seen.add(item["id"])

    seen_src = {s["id"] for s in updated["sources"]}
    srcs = payload.get("sources", [])
    _chk(isinstance(srcs, list), "upsert sources must be a list")
    for item in srcs:
        req_new(item, seen_src, "source")
        updated["sources"].append(copy.deepcopy(item))

    cfgs = updated["configurations"]
    seen_cfg = {c["id"] for c in cfgs}
    new_cfgs = payload.get("configurations", [])
    _chk(isinstance(new_cfgs, list), "upsert configurations must be a list")
    for raw_cfg in new_cfgs:
        req_new(raw_cfg, seen_cfg, "configuration")
        cfg = copy.deepcopy(raw_cfg)
        anchor = cfg.pop("insert_after", None)
        if anchor is not None:
            _chk(isinstance(anchor, str), f"insert_after must be str: {anchor!r}")
            pos = next((i for i, c in enumerate(cfgs) if c["id"] == anchor), None)
            _chk(pos is not None, f"unknown insert_after {anchor!r}")
            cfgs.insert(pos + 1, cfg)  # type: ignore[operator]
        elif cfg.get("kind") == "rpent":
            ext = next(
                (i for i, c in enumerate(cfgs) if c.get("kind") != "rpent"), len(cfgs)
            )
            cfgs.insert(ext, cfg)
        else:
            cfgs.append(cfg)

    bmap = {b["id"]: b for b in updated["benchmarks"]}
    seen_n = {n["id"] for b in updated["benchmarks"] for n in b.get("notes", [])}
    seen_n |= {n["id"] for n in updated.get("footer_notes", [])}
    bnotes = payload.get("benchmark_notes", {})
    _chk(isinstance(bnotes, dict), "benchmark_notes must be an object")
    for bid, nlist in bnotes.items():
        ok_b = isinstance(bid, str) and bid in bmap and isinstance(nlist, list)
        _chk(ok_b, f"bad benchmark_notes[{bid!r}]")
        target = bmap[bid].setdefault("notes", [])
        for note in nlist:
            req_new(note, seen_n, "note")
            target.append(copy.deepcopy(note))

    for field, target, gkey in (
        ("results", updated["results"], "view_id"),
        ("cost_results", updated["cost_results"]["rows"], "benchmark_id"),
    ):
        items = payload.get(field, [])
        _chk(isinstance(items, list), f"upsert {field} must be a list")
        seen_ids = {r["id"] for r in target}
        seen_pairs = {(r[gkey], r["configuration_id"]) for r in target}
        for item in items:
            req_new(item, seen_ids, field)
            gv, cv = item.get(gkey), item.get("configuration_id")
            _chk(isinstance(gv, str) and isinstance(cv, str), f"bad keys in {field}")
            _chk((gv, cv) not in seen_pairs, f"cannot overwrite {field} {(gv, cv)}")
            seen_pairs.add((gv, cv))
            target.append(copy.deepcopy(item))

    validate_data(updated)
    return updated


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p_v = sub.add_parser("validate", help="Validate results.json.")
    p_v.add_argument("--data", type=Path, default=DATA_PATH)
    p_a = sub.add_parser("aggregate", help="Aggregate API planner transcripts.")
    p_a.add_argument("--benchmark", choices=("robotwin", "robocasa"), required=True)
    p_a.add_argument("--input-dir", type=Path, required=True)
    p_a.add_argument("--planner", default="api")
    for fl in ("--manifest", "--target50-summary", "--output"):
        p_a.add_argument(fl, type=Path, default=None)
    for fl in ("--configuration-id", "--protocol-id"):
        p_a.add_argument(fl, default=None)
    p_a.add_argument("--source-id", action="append", dest="source_ids", default=None)
    p_u = sub.add_parser("upsert", help="Append validated entries to results.json.")
    p_u.add_argument("--data", type=Path, default=DATA_PATH)
    p_u.add_argument("--payload", type=Path, required=True)
    mode = p_u.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--in-place", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the leaderboard validation, aggregation, and upsert CLI."""
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "validate":
            d = load_data(args.data)
            print(f"OK: {args.data} (v{d['schema_version']}, {len(d['results'])} rows)")
            return 0
        if args.command == "aggregate":
            keys = ("planner", "configuration_id", "source_ids", "protocol_id")
            kw = {k: getattr(args, k) for k in keys}
            if args.benchmark == "robotwin":
                _must(args.target50_summary is None, "--target50-summary robocasa-only")
                mf = args.manifest or ROBOTWIN_MANIFEST_PATH
                out = aggregate_robotwin(args.input_dir, manifest_path=mf, **kw)
            else:
                _must(args.target50_summary is not None, "--target50-summary required")
                mf = args.manifest or ROBOCASA_MANIFEST_PATH
                out = aggregate_robocasa(
                    args.input_dir, args.target50_summary, manifest_path=mf, **kw
                )
            txt = json.dumps(out, indent=2, ensure_ascii=False) + "\n"
            _atomic_write(args.output, txt) if args.output else print(txt, end="")
            return 0
        payload = json.loads(args.payload.read_text(encoding="utf-8"))
        updated = upsert_data(load_data(args.data), payload)
        if args.dry_run:
            old = args.data.read_text(encoding="utf-8").splitlines(keepends=True)
            new = dump_data(updated).splitlines(keepends=True)
            diff = difflib.unified_diff(old, new, str(args.data), "updated")
            print("".join(diff), end="")
        else:
            write_data(args.data, updated)
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
