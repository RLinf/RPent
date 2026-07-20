from __future__ import annotations

from argparse import Namespace
from typing import cast

import robots.libero.runtime as libero_runtime_module
import robots.rebot_robstride.runtime as rebot_runtime_module


class FakeToolkit:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class FakeLiberoEnv:
    def __init__(self, rpc, *, expected_meta) -> None:
        self.rpc = rpc
        self.expected_meta = expected_meta


def test_libero_runtime_preserves_no_driver_attach(monkeypatch, tmp_path) -> None:
    endpoint_calls: list[tuple] = []
    fake_rpc = object()
    monkeypatch.setattr(
        libero_runtime_module,
        "set_socket_endpoint",
        lambda output, host, port: endpoint_calls.append((output, host, port)),
    )
    monkeypatch.setattr(libero_runtime_module, "create_rpc_client", lambda _: fake_rpc)
    monkeypatch.setattr(libero_runtime_module, "LiberoEnvClient", FakeLiberoEnv)
    monkeypatch.setattr(libero_runtime_module, "VLAClient", lambda url: ("vla", url))
    monkeypatch.setattr(libero_runtime_module, "LiberoToolkit", FakeToolkit)
    args = Namespace(
        suite="libero_object_task",
        task=2,
        seed=3,
        max_episode_steps=99,
        no_driver=True,
        env_endpoint="127.0.0.1",
        env_port=45001,
        vla_endpoint="http://127.0.0.1:45002",
        libero_type=None,
        cuda_device=None,
    )

    runtime = libero_runtime_module.LiberoRuntime(
        args=args, output_dir=tmp_path, dashboard=None
    )
    toolkit = cast(FakeToolkit, runtime.start())

    assert endpoint_calls == [(tmp_path, "127.0.0.1", 45001)]
    assert toolkit.kwargs["primitives_kwargs"]["env"].expected_meta == {
        "suite": "libero_object_task",
        "task": 2,
        "seed": 3,
        "max_episode_steps": 99,
    }
    assert toolkit.kwargs["primitives_kwargs"]["model"] == (
        "vla",
        "http://127.0.0.1:45002",
    )


def test_rebot_runtime_supports_no_driver_attach(monkeypatch, tmp_path) -> None:
    endpoint_calls: list[tuple] = []
    fake_rpc = object()
    monkeypatch.setattr(
        rebot_runtime_module,
        "set_socket_endpoint",
        lambda output, host, port: endpoint_calls.append((output, host, port)),
    )
    monkeypatch.setattr(rebot_runtime_module, "create_rpc_client", lambda _: fake_rpc)
    monkeypatch.setattr(
        rebot_runtime_module, "RebotRobstrideEnvClient", lambda rpc: ("env", rpc)
    )
    monkeypatch.setattr(rebot_runtime_module, "RebotRobstrideToolkit", FakeToolkit)
    args = Namespace(
        no_driver=True,
        env_endpoint="127.0.0.1",
        env_port=46001,
        env_config=None,
    )

    runtime = rebot_runtime_module.RebotRobstrideRuntime(
        args=args, output_dir=tmp_path, dashboard=None
    )
    toolkit = cast(FakeToolkit, runtime.start())

    assert endpoint_calls == [(tmp_path, "127.0.0.1", 46001)]
    assert toolkit.kwargs["env"] == ("env", fake_rpc)
