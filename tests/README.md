# Test suite structure

RPent keeps its fast, offline unit tests under `unit_tests`. Tests mirror the
production module they exercise so they remain easy to find as the suite
grows.

## Directory layout

```text
tests/
├── README.md
└── unit_tests/
    ├── rpent/             # tests for the core rpent package
    │   ├── cli/
    │   ├── dashboard/
    │   ├── memory/
    │   ├── planner/
    │   ├── robots/
    │   ├── session/
    │   ├── tools/
    │   └── utils/
    └── robots/           # tests for the top-level robot extensions
        ├── libero/
        ├── robocasa/
        └── robotwin/
```

Directories that do not yet have tests do not need empty placeholders. Add
them when the first test for that module lands.

## Placement rules

- Mirror core modules under `tests/unit_tests/rpent/`. For example, tests for
  `rpent.utils.rpc` belong in `tests/unit_tests/rpent/utils/rpc/`.
- Mirror top-level robot extensions under `tests/unit_tests/robots/`. Put
  robot-specific coverage in that directory's `<robot>/` child.
- Place cross-layer tests with the primary contract owner. Registry and config
  contracts belong to `rpent/robots/`; extension toolkit and schema contracts
  belong to `robots/`.
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

Real simulator checks live under `integration_tests` and are always opt-in;
they are not part of the offline CI suite. For example, after installing the
RoboCasa extra and assets, run its 340-cell Target50 environment contract with:

```bash
RPENT_RUN_ROBOCASA_INTEGRATION=1 \
  pytest tests/integration_tests/robots/robocasa/test_target50_runtime_smoke.py -v
```

This opt-in test constructs and resets every manifest task/seed cell, verifies
the 12D action interface, operation cameras, navigation RGB-D/world map,
success predicate, and clean close. It does not invoke a planner or VLA model.
