# Mistakes agents make in RPent docs and prose

Check this list before finishing any documentation page, README update, prompt/tool prose change, PR description, or review comment.

## Content and accuracy

| Mistake | Fix |
|---|---|
| Naming internal `rpent/`, `robots/`, or `components/` modules, classes, or helper functions on simulator, toolkit, or user guide pages | Describe CLI options, config keys, saved artifacts, and observable behavior |
| Conflating planner `finish` or a clean CLI exit with benchmark task success | State the environment's native success criterion (`terminated` or verifier reward) explicitly |
| Copying simulator/model commands, extras, defaults, or env vars from another page without checking code | Verify against `pyproject.toml`, `rpent/cli/main.py`, and `robots/<robot>/` |
| Deleting leaderboard entries, reproduction conditions/denominators, or coming-soon placeholders while streamlining | Preserve published numbers, experimental conditions, and coming-soon entries unless explicitly asked to change them |
| Narrating review exchanges or replaced implementations ("the fixed installer no longer...") | Describe current behavior and real operational constraints directly |
| Explaining the same point from negative and positive sides ("this is not A, not B, but C" / "这不是……而是……") | State directly what it is or does once |
| Editing only English or only Chinese, or letting `.. toctree::` entries drift | Update both `docs/source-en/` and `docs/source-zh/` together |
| Moving a page without updating both `conf.py` `redirects` maps | Add old-to-new redirects in both `docs/source-en/conf.py` and `docs/source-zh/conf.py` |

## Structure and RST

| Mistake | Fix |
|---|---|
| Skipping standard simulator sections or changing Overview card titles/order | Follow [simulator.md](simulator.md): `Action Models`, `Planners`, `Tasks`, `Hardware` + H3 `Tasks` and `Observation and Action` |
| Bare link lists or duplicate toctrees on `:orphan:` directory routing pages | Use `sphinx-design` cards or a `.. list-table::` on directory routing pages, and register child pages in the home-page `:hidden:` toctrees in `docs/source-{en,zh}/index.rst` |
| Hardcoded `rpent.readthedocs.io` links inside `docs/` | Use `:doc:` and `:ref:` (READMEs keep rendered Read the Docs URLs) |
| CJK punctuation or characters touching `` ``code`` `` or `**bold**` without `\ ` | Insert an escaped space (`\ `) and run `python3 .agents/skills/docs/check.py` |
| Unescaped hard-wrapped Chinese paragraphs | Keep Chinese prose on one source line or end wrapped lines with `\` |
| Adding standalone `.md` guides outside `docs/` | Put user and developer guides as paired `.rst` pages under `docs/source-en/` and `docs/source-zh/` |

## Language

- **English** ([english.md](english.md)): cut filler (`simply`, `note that`, `in order to`), hype (`seamlessly`, `powerful`, `robust`, `leverage`, `unlock`), trailing `-ing` benefit clauses, hedging (`consider using`), and Chinese calques (`video memory`, `carry out evaluation`).
- **Chinese** ([chinese.md](chinese.md)): write natively from the plan; cut 翻译腔 (`当……时`, `进行 + 名词`, `对于……来说`, `通过……的方式`, `允许你……`, `值得注意的是`, `不是……而是……`), use `你` (not `您`), `“”` (not `「」`), full-width Chinese punctuation, and spaces around English/digits; follow [glossary.md](glossary.md).

## Verification

| Mistake | Fix |
|---|---|
| Building only English docs | Run `python3 .agents/skills/docs/check.py` (builds both EN and ZH with `fail_on_warning` semantics) |
| Treating a clean Sphinx build as proof that commands or URLs work | Verify code contracts, check media HTTP 200 directly, and report untested runtime steps honestly |
| Reviewing your own prose in the same context | Run the readability pass in a fresh agent that sees only the changed pages and `docs/rules/` |
