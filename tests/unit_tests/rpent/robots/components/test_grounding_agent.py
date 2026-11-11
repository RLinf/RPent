# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Point-only agent adapters run offline with fake model output."""

import base64
import io
import json
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
from pydantic_ai.models.test import TestModel

from rpent.robots.components.grounding_agent import GroundingAgent


@pytest.fixture
def image():
    buffer = io.BytesIO()
    Image.new("RGB", (20, 10)).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.mark.parametrize("point", [[5, 3], None])
def test_api_point_selection(monkeypatch, image, point):
    import rpent.planner.base as base

    model = TestModel(custom_output_args={"point_xy": point})
    monkeypatch.setattr(base, "build_api_model", lambda *a: model)
    result = GroundingAgent("test:model").ground(image, "cup rim")
    assert result.found == (point is not None)
    assert result.image_size == (20, 10)
    assert model.last_model_request_parameters.function_tools == []


@pytest.mark.parametrize("point", [[25, 3], [-1, 2]])
def test_outside_image_is_rejected(monkeypatch, image, point):
    import rpent.planner.base as base

    monkeypatch.setattr(
        base,
        "build_api_model",
        lambda *a: TestModel(custom_output_args={"point_xy": point}),
    )
    with pytest.raises(ValueError, match="outside"):
        GroundingAgent("test:model").ground(image, "cup rim")


def test_codex_receives_image_and_no_robot_tools(monkeypatch, image, tmp_path):
    import rpent.planner.codex as codex

    source = tmp_path / "user"
    source.mkdir()
    (source / "config.toml").write_text(
        '[mcp_servers.robot]\ncommand="false"\n'
        '[model_providers.proxy]\nbase_url="https://example.test/v1"\n'
    )
    (source / "auth.json").write_text("{}")

    @dataclass
    class Config:
        cwd: str = "/unused"
        config_overrides: tuple = ()
        env: dict = None

    monkeypatch.setattr(
        codex, "build_probe_config", lambda *a: Config(env={"CODEX_HOME": str(source)})
    )

    def probe(config, **kwargs):
        assert config.cwd != "/unused"
        home = Path(config.env["CODEX_HOME"])
        assert home != source
        assert not (home / "config.toml").exists()
        assert (home / "auth.json").resolve() == source / "auth.json"
        assert "mcp_servers={}" not in config.config_overrides
        assert any("model_providers" in value for value in config.config_overrides)
        assert "features.shell_tool=false" in config.config_overrides
        assert "features.view_image=false" in config.config_overrides
        assert "features.view_image_tool=false" in config.config_overrides
        assert kwargs["image_bytes"] == image
        assert kwargs["output_schema"]["required"] == ["point_xy"]
        return '{"point_xy": [5, 3]}'

    import rpent.robots.components.grounding_agent as grounding

    homes = []
    monkeypatch.setattr(
        grounding,
        "_check_codex_mcp_isolation",
        lambda config: homes.append(Path(config.env["CODEX_HOME"])),
    )
    monkeypatch.setattr(codex, "run_probe_turn", probe)
    assert GroundingAgent("codex:test").ground(image, "cup rim").point_xy == (5, 3)

    assert homes and not homes[0].exists()
    assert (source / "auth.json").is_file()


@pytest.mark.parametrize(
    "servers, rejected",
    [({}, False), ({"robot": {}}, True), ({"robot": {"enabled": False}}, False)],
)
def test_effective_mcp_configuration_is_checked(monkeypatch, servers, rejected):
    import openai_codex.client as sdk

    from rpent.robots.components.grounding_agent import _check_codex_mcp_isolation

    class Client:
        def __init__(self, config):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def initialize(self):
            pass

        def request(self, method, params, **kwargs):
            assert method == "config/read"
            return SimpleNamespace(
                config=SimpleNamespace(model_dump=lambda: {"mcp_servers": servers})
            )

    monkeypatch.setattr(sdk, "CodexClient", Client)
    if rejected:
        with pytest.raises(RuntimeError, match="inherited MCP"):
            _check_codex_mcp_isolation(SimpleNamespace(cwd="/tmp"))
    else:
        _check_codex_mcp_isolation(SimpleNamespace(cwd="/tmp"))


def test_keyring_login_has_actionable_error(tmp_path):
    from rpent.robots.components.grounding_agent import _isolated_codex_environment

    source = tmp_path / "user"
    source.mkdir()
    (source / "config.toml").write_text('cli_auth_credentials_store="keyring"')
    with pytest.raises(RuntimeError, match="CODEX_API_KEY"):
        _isolated_codex_environment({"CODEX_HOME": str(source)}, tmp_path / "isolated")


@pytest.mark.parametrize("store", ["keyring", "auto"])
@pytest.mark.parametrize("override_provider", [False, True])
def test_custom_provider_does_not_require_keyring_login(
    tmp_path, store, override_provider
):
    from rpent.robots.components.grounding_agent import _isolated_codex_environment

    source = tmp_path / "user"
    source.mkdir()
    (source / "config.toml").write_text(
        f'cli_auth_credentials_store="{store}"\n'
        f'model_provider="{"openai" if override_provider else "custom"}"\n'
        "[model_providers.custom]\n"
        'base_url="https://example.test/v1"\nenv_key="CUSTOM_API_KEY"\n'
    )
    kwargs = (
        {"config_overrides": ('model_provider="custom"',)} if override_provider else {}
    )
    env, _ = _isolated_codex_environment(
        {"CODEX_HOME": str(source), "CUSTOM_API_KEY": "offline-test"},
        tmp_path / "isolated",
        **kwargs,
    )
    assert env["CUSTOM_API_KEY"] == "offline-test"


def test_provider_override_can_require_file_login(tmp_path):
    from rpent.robots.components.grounding_agent import _isolated_codex_environment

    source = tmp_path / "user"
    source.mkdir()
    (source / "config.toml").write_text(
        'cli_auth_credentials_store="keyring"\nmodel_provider="custom"\n'
        '[model_providers.custom]\nenv_key="CUSTOM_API_KEY"\n'
    )
    with pytest.raises(RuntimeError, match="file-based Codex login"):
        _isolated_codex_environment(
            {"CODEX_HOME": str(source), "CUSTOM_API_KEY": "offline-test"},
            tmp_path / "isolated",
            ('model_provider="openai"',),
        )


def test_codex_cannot_attach_an_image_outside_grounding_directory(
    monkeypatch, tmp_path, image
):
    """Exercise the real CLI with a local model that requests an unprovided image."""

    outside = tmp_path / "outside.png"
    Image.new("RGB", (31, 17), "red").save(outside)
    outside_data = base64.b64encode(outside.read_bytes()).decode()
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(payload)
            if len(requests) == 1:
                item = {
                    "id": "fc_outside",
                    "type": "function_call",
                    "call_id": "call_outside",
                    "name": "view_image",
                    "arguments": json.dumps({"path": str(outside)}),
                }
            else:
                item = {
                    "id": "msg_point",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": '{"point_xy":[5,3]}',
                            "annotations": [],
                        }
                    ],
                }
            events = [
                {"type": "response.output_item.done", "output_index": 0, "item": item},
                {
                    "type": "response.completed",
                    "response": {
                        "id": f"resp_{len(requests)}",
                        "status": "completed",
                        "output": [item],
                    },
                },
            ]
            body = "".join(
                f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
                for event in events
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    source = tmp_path / "codex-home"
    source.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(source))
    monkeypatch.setenv("CODEX_API_KEY", "offline-test")
    monkeypatch.delenv("CODEX_BIN", raising=False)
    try:
        result = GroundingAgent(
            "codex:gpt-5.4", f"http://127.0.0.1:{server.server_port}/v1"
        ).ground(image, "cup rim")
        assert result.point_xy == (5, 3)
        assert len(requests) == 2
        assert outside_data not in json.dumps(requests)
        assert not requests[0].get("tools")
        tool_output = [
            item
            for item in requests[1]["input"]
            if item.get("type") == "function_call_output"
            and item.get("call_id") == "call_outside"
        ]
        assert tool_output and "view_image" in json.dumps(tool_output)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_inherited_mcp_stops_before_model_request(monkeypatch, image, tmp_path):
    import openai_codex

    import rpent.planner.codex as codex
    import rpent.robots.components.grounding_agent as grounding

    monkeypatch.setattr(
        codex,
        "build_probe_config",
        lambda *a: openai_codex.CodexConfig(env={"CODEX_HOME": str(tmp_path)}),
    )
    homes = []

    def reject(config):
        homes.append(Path(config.env["CODEX_HOME"]))
        raise RuntimeError("inherited MCP")

    monkeypatch.setattr(grounding, "_check_codex_mcp_isolation", reject)

    def unexpected_request(*a, **kw):
        pytest.fail("model request must not run after isolation fails")

    monkeypatch.setattr(codex, "run_probe_turn", unexpected_request)
    with pytest.raises(RuntimeError, match="inherited MCP"):
        GroundingAgent("codex:test").ground(image, "cup rim")
    assert homes and not homes[0].exists()


def test_isolated_provider_preserves_endpoint_and_headers(tmp_path):
    import rpent.robots.components.grounding_agent as grounding

    source = tmp_path / "user"
    source.mkdir()
    (source / "config.toml").write_text(
        'model_provider="proxy"\n[model_providers.proxy]\n'
        'base_url="https://example.test/v1"\nenv_key="TEST_KEY"\n'
        'http_headers={"X-Test"="value"}\n'
    )
    _, overrides = grounding._isolated_codex_environment(
        {"CODEX_HOME": str(source)}, tmp_path / "isolated"
    )
    restored = grounding.tomllib.loads("\n".join(overrides))
    assert restored["model_provider"] == "proxy"
    assert restored["model_providers"]["proxy"] == {
        "base_url": "https://example.test/v1",
        "env_key": "TEST_KEY",
        "http_headers": {"X-Test": "value"},
    }
