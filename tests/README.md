# Test suite structure

RPent keeps its fast, offline unit tests under `unit_tests`. Tests mirror the
production module they exercise so they remain easy to find as the suite
grows.

## Directory layout

```text
tests/
├── README.md
└── unit_tests/
    ├── cli/              # rpent.cli
    ├── dashboard/        # rpent.dashboard
    ├── memory/           # rpent.memory
    ├── planner/          # rpent.planner
    ├── robots/           # cross-layer robot extension contracts
    ├── session/          # rpent.session
    ├── tools/            # rpent.tools
    └── utils/            # rpent.utils, including RPC loopback tests
```

Directories that do not yet have tests do not need empty placeholders. Add
them when the first test for that module lands.

## Placement rules

- Mirror the production module under `tests/unit_tests/` whenever there is a
  clear owner. For example, tests for `rpent.utils.rpc` belong in
  `tests/unit_tests/utils/rpc/`.
- Keep contracts that span the `rpent.robots` registry and top-level `robots`
  extensions directly under `tests/unit_tests/robots/`. Put robot-specific
  coverage under that directory's `<robot>/` child.
- Keep one-off fakes in the test module that uses them. Put shared fixtures in
  the nearest `conftest.py`: use `tests/conftest.py` only for suite-wide
  fixtures and a module directory's `conftest.py` for local fixtures.
- Name files after the behavior they verify: `*_contracts.py` for stable API
  contracts, `*_loopback.py` for real local transports, `*_lifecycle.py` for
  resource ownership, and `*_smoke.py` for installation or startup checks.

Unit tests must run offline on an ordinary CPU machine. They may cross a real
local boundary, such as a loopback TCP connection or child process, but must
not contact external services.

Run the complete suite with:

```bash
pytest tests/unit_tests -v
```
