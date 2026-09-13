# Benchmark chart authoring

The documentation serves the SVG and PNG assets in `docs/_static/benchmarks/`
directly. Sphinx and RPent do not require Matplotlib at runtime.

`docs/_static/benchmarks/results.json` is the editable source for model
configurations, reported and unreported results, evaluation scopes, protocols,
sources, table order, and chart selections. Rates are decimal strings such as
`"52.0"`, preserving the source precision. Success counts and episode counts are
independent nullable fields: never infer success counts from rounded percentages,
and never apply RPent denominators to external reports. Unreported results use
`status: "not_reported"` with null rate and counts.

Each view represents exactly one evaluation scope. The six non-Long Cap-X/RATS
results have their own view, separate from eight-item PRO Overall. Zero-shot Goal
has its own benchmark group. Each view's `record_ids` retains all its records;
`chart_record_ids` selects every reported RPent configuration plus representative
external methods. The default six `overview_views` use the same selected bars in
`overview_record_ids`, currently 26 in total. Axis limits cover all reported
records in the view and stay fixed when a user filters methods.

To add a model, append a configuration and an explicit reported or unreported
record for every main table row. Model columns follow configuration order, so
existing columns remain stable. To add a score, update its result, sources, and
protocol; include the reported RPent record in the view's chart selection, and
update `axis_max` to the scope's maximum rounded up to 10 percentage points.
Complete Standard/PRO Overall requires all of its component rows and the full
denominator; Long-only Astra results cannot define PRO Overall. New evaluation
rows add a view and reference it from the corresponding table's `view_ids`.

Regenerate and check the English/Chinese RST tables with the standard library:

```bash
python docs/tools/benchmark_data.py --write
python docs/tools/benchmark_data.py --check
```

The generated `benchmark-data-begin` / `benchmark-data-end` blocks preserve the
existing RST `list-table` style. Do not edit them by hand. The checker validates
IDs, references, scopes, rate precision, counts, complete aggregate coverage,
model-column completeness, and exact bilingual table synchronization; `--check`
does not write files. The surrounding explanatory text and source notes remain
authored RST and should be updated when evaluation protocols change.

`render_leaderboard.py` reads the same JSON and checks generated tables before
rendering the selected records. The current 22 views and six-panel overview
produce 184 SVG/PNG assets. The CI data check verifies all 92 SVGs: embedded scores,
axis limits and tick labels, bar colors, zero baselines, actual bar heights
relative to each axis's rendered bounds, and embedded font glyphs. Stale assets
are rejected when scores, labels, sources, or axis scales change. Missing results
cannot be rendered as zero. The figures compare only each panel's stated coverage;
Long-only Astra scores do not enter the complete LIBERO-PRO Overall panel.

Run the data check with the Python standard library:

```bash
python docs/tools/render_leaderboard.py --check-data
```

To regenerate the English/Chinese, light/dark SVG and PNG variants, install the
optional pinned authoring dependency in a separate environment and supply genuine
Times New Roman regular and bold fonts, plus KaiTi (楷体) for Chinese text.
The committed assets use Times New Roman version 2.82 (`Times.TTF` and
`Timesbd.TTF`) and KaiTi version 5.02 (`simkai.ttf`). Supply your local font
paths; font files are not included in the repository. The renderer checks the
font family and PostScript names and rejects substitutes. English text, numerals,
percentages, and effort labels use Times New Roman in both language variants.
Chinese annotations use larger KaiTi text.

```bash
python -m venv .leaderboard-venv
.leaderboard-venv/bin/pip install -r docs/tools/requirements-leaderboard.txt
.leaderboard-venv/bin/python docs/tools/render_leaderboard.py \
  --times-font /path/to/Times.TTF \
  --times-bold-font /path/to/Timesbd.TTF \
  --cjk-font /path/to/simkai.ttf \
  --audit-report /tmp/leaderboard-text-layout.json
```

`--output-dir` can render a temporary copy for comparison. Each panel is saved as
`<panel>-<en|zh>-<light|dark>.<svg|png>`; `leaderboard-...` contains the complete
three-column, two-row layout, with the legend in a separate top header. The page
uses individual SVGs to adapt the columns to narrower screens. SVG text is converted
to paths so browsers do not need the authoring fonts. Each reported record has a
stable SVG group containing its bar, value and model labels. The browser filters
and translates these groups without changing the glyphs or vertical scale; shared
glyph definitions remain outside the groups. Rendering measures every visible text box and rejects
overlapping or clipped labels; `--audit-report` saves these measurements for
review. SVG IDs use a fixed hash salt and metadata excludes dates;
the same dependency/font versions produce byte-identical assets.

When editing a score, update the JSON and its documented evidence, regenerate
the table blocks, then regenerate the chart assets. Keep RPent purple (`#7842B1`), external
methods gray (`#AEB7C6`) and all planner configurations equally styled. Narrow bars
retain equal physical width in the three-model and five-method panels. Every
vertical axis starts at 0%; its maximum is the scope's highest reported score rounded up
to the next multiple of 10 percentage points (100%, 90%, 60%, 60%, 90%, and 80%
for the current six panels). Do not compare bar heights across different panels.
Retain the external baseline role and evaluation scope;
this display is not a claim to rank all methods on a benchmark.
