# Benchmark chart authoring

The documentation serves the SVG and PNG assets in `docs/_static/benchmarks/`
directly. Sphinx and RPent do not require Matplotlib at runtime.

`render_leaderboard.py` contains the six approved panel datasets. Before rendering,
it reads both benchmark RST pages and checks every plotted value against the
correct evaluation row and model/effort column, or the corresponding external
baseline row, coverage, and source. It also checks the model/backend/reasoning
mappings, descending order, source URLs, and the expected 26 bars. The CI data
check verifies all 28 SVGs, including the complete leaderboards: embedded scores,
axis limits and tick labels, bar colors, zero baselines, actual bar heights
relative to each axis's rendered bounds, and embedded font glyphs. Stale assets are rejected when documented
scores or axis scales change. Missing results cannot
be rendered as zero. The figures compare only the coverage stated in each panel;
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
to paths so browsers do not need
the authoring fonts. Rendering measures every visible text box and rejects
overlapping or clipped labels; `--audit-report` saves these measurements for
review. SVG IDs use a fixed hash salt and metadata excludes dates;
the same dependency/font versions produce byte-identical assets.

When editing a score, update its documented evidence and both RST tables first,
then update the panel entry and regenerate. Keep RPent purple (`#7842B1`), external
methods gray (`#AEB7C6`) and all planner configurations equally styled. Narrow bars
retain equal physical width in the three-model and five-method panels. Every
vertical axis starts at 0%; its maximum is the highest plotted score rounded up
to the next multiple of 10 percentage points (100%, 90%, 60%, 60%, 90%, and 80%
for the current six panels). Do not compare bar heights across different panels.
Retain the external baseline role and evaluation scope;
this display is not a claim to rank all methods on a benchmark.
