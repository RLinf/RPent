# Docs check reference

Background for the rules [check.py](check.py) applies and for the judgment items in [SKILL.md](SKILL.md#check). The writing rules live in [rules/](rules/).

## Source of truth for robots, planners, and extras

Check the implementation rather than copying lists from another page:

- **Robots**: subdirectories under `robots/` discovered by `enumerate_robots()` in `rpent/robots/base.py`.
- **Planners**: modules under `rpent/planner/` (`api`, `claude_code`, `codex`, `gemini_cli`, `flash`).
- **Install extras and pins**: `[project.optional-dependencies]` and `[tool.uv] conflicts` in `pyproject.toml`.
- **CLI flags and defaults**: `rpent/cli/main.py` and each robot's `add_cli_args`.

## The build gate

Both languages are separate Read the Docs projects, configured by `docs/source-en/.readthedocs.yaml` and `docs/source-zh/.readthedocs.yaml`, and both set `fail_on_warning: true`. Any Sphinx warning fails the corresponding Read the Docs check. Because the two projects build independently, a green English build does not prove the Chinese build passes. `check.py` builds both through [build_docs.py](build_docs.py), which can also run alone (`--lang zh` to iterate on a Chinese-only failure, `--keep-output` to inspect rendered HTML).

## Inline markup next to CJK punctuation

docutils only recognizes inline markup when the *start*-string is preceded by whitespace, an opener, or a delimiter, and the *end*-string is followed by whitespace, a closer, or a delimiter. Full-width punctuation splits across those classes:

| Character | docutils class | Before markup | After markup |
|---|---|---|---|
| `（` `《` `“` | opener | ok | **rejected** |
| `）` `》` `”` | closer | **rejected** | ok |
| `，` `、` `：` `；` `。` `！` `？` `—` | delimiter | ok | ok |
| any CJK ideograph | none | **rejected** | **rejected** |

Two failures follow, and only the first breaks CI:

1. **Build break.** No later end-string candidate satisfies the rule, so Sphinx emits `WARNING: Inline literal start-string without end-string. [docutils]` and fails the CN build:

   ```rst
   导出为 ``MUJOCO_EGL_DEVICE_ID``（MuJoCo）和 ``EGL_DEVICE_ID``（其他）。
   ```

2. **Silent mis-render.** A later `` `` `` in the same paragraph satisfies the rule, so docutils closes the literal there and swallows the intervening Chinese prose into the code span with no warning. Similarly, `**粗体**` glued to Chinese characters renders literal asterisks.

Fix both with an escaped space (`\ `) between the markup and the adjacent character, which suppresses the rendered space:

```rst
导出为 ``MUJOCO_EGL_DEVICE_ID``\ （MuJoCo）和 ``EGL_DEVICE_ID``\ （其他）。
通常\ **并不是**\ 这样。
```

[check_rst_markup.py](check_rst_markup.py) reports both classes, and `check.py` maps build breaks to `critical` and silent mis-renders to `major`.

## Hard-wrapped Chinese prose

reStructuredText joins consecutive source lines inside a paragraph with a space, so bare line wrapping inside Chinese prose inserts a visible gap between characters. Keep Chinese prose paragraphs and list items on one source line, or end wrapped source lines with a trailing backslash (`\`) per [rules/chinese.md](rules/chinese.md).

## Code-font names

[check_doc_symbols.py](check_doc_symbols.py) scans single-word inline literals on `development/` pages and flags identifiers that appear nowhere in the repository. Illustrative tutorial names go in `ALLOWED`; external project names go in `EXTERNAL`.

## Asset and media references

Every image, poster, video, or Hugging Face checkpoint reference in changed docs must pass three checks:

1. **Direct HTTP 200.** Use direct URLs (e.g., `https://raw.githubusercontent.com/<org>/<repo>/<branch>/<path>`) rather than redirecting `github.com/.../raw/...` links when adding new media, and verify HTTP 200 (`curl -sL -o /dev/null -w '%{http_code}' <url>`). Report blocked checks if network access is unavailable.
2. **Hugging Face repos exist.** Verify model/dataset IDs against `https://huggingface.co/api/models/<org>/<repo>` (or `/api/datasets/...`) and check that the returned `id` matches the documented name.
3. **Alt text, credits, and dark mode.** Provide descriptive `:alt:` text, captions, and source credits for upstream figures; check transparency and contrast in both light and dark themes.
