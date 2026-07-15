# `rpent/cli/`

CLI entrypoints for RPent (currently just `main.py`).

## Rules

- **No `__init__.py`.** This directory is not a Python subpackage of
  `rpent`. Do not add one.
- **Never reference `rpent.cli` as a dotted import path** anywhere in
  the codebase. Without `__init__.py` it isn't one, and treating it as
  one would risk import cycles — `main.py` already pulls in
  `rpent.cerebrum`, `rpent.envs`, `rpent.utils`, `rpent.dashboard`, and
  `rpent.tools`, so exposing it as an importable submodule invites a
  cycle back to the CLI.
- **Setuptools skips it.** `pyproject.toml`'s `packages.find` only picks
  up dirs with `__init__.py`, so the CLI ships as source-tree scripts,
  not as an installed submodule.

## Run

```bash
python rpent/cli/main.py --suite libero_object_task --task 0 --seed 0 [...]
```

Do not use `python -m rpent.cli.main`.
