from __future__ import annotations

import sys
import types

import pytest

from rpent.envs import get_runtime
from rpent.envs.runtime import EnvRuntime


class DummyRuntime(EnvRuntime):
    def __init__(self, marker: str) -> None:
        self.marker = marker
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True
        return object()

    def stop(self) -> None:
        self.stopped = True


def test_get_runtime_uses_lazy_environment_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = types.ModuleType("robots.fake_runtime")
    module.get_runtime = lambda **kwargs: DummyRuntime(kwargs["marker"])
    monkeypatch.setitem(sys.modules, "robots.fake_runtime", module)

    runtime = get_runtime("fake_runtime", marker="expected")

    assert isinstance(runtime, DummyRuntime)
    assert runtime.marker == "expected"


def test_get_runtime_reports_missing_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("robots.no_runtime")
    monkeypatch.setitem(sys.modules, "robots.no_runtime", module)

    with pytest.raises(ValueError, match="does not expose get_runtime"):
        get_runtime("no_runtime")
