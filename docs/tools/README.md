# Benchmark chart authoring

The documentation serves the SVG and PNG assets in `docs/_static/benchmarks/`
directly. Sphinx and RPent do not require Matplotlib at runtime.

`render_leaderboard.py` contains the six approved panel datasets. Before rendering,
it reads both benchmark RST pages and checks every plotted value against the
correct evaluation row and model/effort column, or the corresponding external
baseline row, coverage, and source. It also checks the model/backend/reasoning
mappings, descending order, source URLs, and the expected 26 bars. The CI data
check verifies the 24 panel SVGs' embedded scores against the same data, so stale
assets are rejected when documented scores change. Missing results cannot
be rendered as zero. The figures compare only the coverage stated in each panel;
Long-only Astra scores do not enter the complete LIBERO-PRO Overall panel.

Run the data check with the Python standard library:

```bash
python docs/tools/render_leaderboard.py --check-data
```

To regenerate the English/Chinese, light/dark SVG and PNG variants, install the
optional pinned authoring dependency in a separate environment and supply a CJK
font. The committed assets use WenQuanYi Zen Hei (`fonts-wqy-zenhei` on Debian or
Ubuntu), with Matplotlib's bundled DejaVu Sans for Latin text.

```bash
python -m venv .leaderboard-venv
.leaderboard-venv/bin/pip install -r docs/tools/requirements-leaderboard.txt
.leaderboard-venv/bin/python docs/tools/render_leaderboard.py \
  --cjk-font /usr/share/fonts/truetype/wqy/wqy-zenhei.ttc
```

`--output-dir` can render a temporary copy for comparison. Each panel is saved as
`<panel>-<en|zh>-<light|dark>.<svg|png>`; `leaderboard-...` contains the complete
two-column, six-panel layout. The page uses individual SVGs to switch to one
column on narrow screens. SVG text is converted to paths so browsers do not need
the authoring fonts. SVG IDs use a fixed hash salt and metadata excludes dates;
the same dependency/font versions produce byte-identical assets.

When editing a score, update its documented evidence and both RST tables first,
then update the panel entry and regenerate. Keep RPent purple (`#7842B1`), external
methods gray (`#AEB7C6`), all planner configurations equally styled, and every
vertical axis at 0–100%. Retain the external baseline role and evaluation scope;
this display is not a claim to rank all methods on a benchmark.
