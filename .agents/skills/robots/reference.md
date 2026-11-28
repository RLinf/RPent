# Robot integration and packaging reference

Concrete shapes for [SKILL.md](SKILL.md). Always verify against the current source on `main` before copying.

## 1. Lazy package exports (`robots/<robot>/__init__.py`)

Prefer `__getattr__` so `import robots.<robot>` or `enumerate_robots()` does not import heavyweight dependencies:

```python
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from robots.my_robot.robot_spec import get_robot_spec, get_toolkit

__all__ = ["get_robot_spec", "get_toolkit"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from robots.my_robot import robot_spec

        return getattr(robot_spec, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

## 2. Owned daemons vs. borrowed endpoints (`init_runtime`)

Use `rpent/robots/runtime.py` helpers so startup failure tears down earlier owned daemons and never touches externally started endpoints:

```python
daemons: dict[str, ProcessDaemon] = {}
env_daemon, env_rpc = try_spawn_server(daemons, dashboard_events, "env", spawn_env)
try_wait_server(daemons, dashboard_events, "env", env_rpc, env_daemon, timeout_s=120.0)
```

When `components` (`set[str] | None`) is passed by the Dashboard, start only the requested subset of components and return `(list(daemons.values()), runtime_handles)`.

## 3. Component manifests and conflicts (`component.toml`)

- Put integration-only dependencies in `robots/<robot>/component.toml` (or `components/{sim,models,tools,infra}/<name>/component.toml`) using `schema = 2`:
  ```toml
  schema = 2
  name = "robot-my_robot"
  probe = "robots.my_robot"
  planner = true
  requires = ["infra-rlinf"]
  dependencies = [
      "upstream-pkg @ git+https://github.com/RLinf/upstream.git@rpent",
  ]
  ```
- When a component pins incompatible `torch`, `transformers`, `numpy`, or `robosuite` versions, declare mutually exclusive components in `env_conflicts`.
- If a component needs `uv` resolver overrides that must not affect other components, keep them alongside the component (following `components/sim/robodojo/requirements/robodojo-override.txt`) and list them in `overrides`.
- Run `rpent-install check --all` and `bash tests/tooling/run.sh layout` to validate manifests, package discovery (`setup.py` / `MANIFEST.in`), and import boundaries.

## 4. Declarative GPU E2E suites

Add a single file `tests/e2e_tests/suites/<suite>.toml` declaring `component`, optional `env`/`base_args`/`[setup]`/`[tool_args]`, and one `[[cases]]` entry per action-model backend. `tests/e2e_tests/run_gpu_suite.sh`, `tests/tooling/ci_targets.py`, and `.github/workflows/embodied-gpu-e2e.yml` discover all `suites/*.toml` files automatically.
