# Test suite structure

RPent organizes tests by the production module they exercise, so a module's
tests remain easy to find as the suite grows.

## Directory layout

```text
tests/
├── cli/                  # rpent.cli
├── dashboard/            # rpent.dashboard
├── memory/               # rpent.memory
├── planner/              # rpent.planner
├── robots/               # shared and robot-specific extension contracts
│   ├── libero/
│   ├── robocasa/
│   └── robotwin/
├── session/              # rpent.session
├── tools/                # rpent.tools
└── utils/                # rpent.utils
    └── rpc/              # local HTTP/socket transport tests
```

Directories that do not yet have tests do not need empty placeholders. Add
them when the first test for that module lands.

## Placement rules

- Mirror the production module whenever there is a clear owner. For example,
  tests for `rpent.utils.rpc` belong in `tests/utils/rpc/`.
- Keep contracts shared by multiple robots directly under `tests/robots/`.
  Put robot-specific coverage under `tests/robots/<robot>/`.
- Keep one-off fakes in the test module that uses them. Put shared fixtures in
  the nearest `conftest.py`: use `tests/conftest.py` only for suite-wide
  fixtures and a module directory's `conftest.py` for local fixtures.
- Name files after the behavior they verify: `*_contracts.py` for stable API
  contracts, `*_loopback.py` for real local transports, `*_lifecycle.py` for
  resource ownership, and `*_smoke.py` for installation or startup checks.

All tests in this tree must run offline on an ordinary CPU machine. Tests may
cross a real local boundary, such as a loopback TCP connection or child
process, but must not contact external services.

Run the complete suite with:

```bash
pytest tests -v
```
