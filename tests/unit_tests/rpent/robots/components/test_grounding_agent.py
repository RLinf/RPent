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

import io
from dataclasses import dataclass
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
