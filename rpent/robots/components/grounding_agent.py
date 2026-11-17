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

"""Tool-free visual point selection with the Molmo result contract."""

import io
import json
import tempfile
from dataclasses import replace
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from rpent.robots.components.molmo_client import MolmoResult


class PointSelection(BaseModel):
    """One point in original-image pixel coordinates, or no visible target."""

    model_config = ConfigDict(allow_inf_nan=False)
    point_xy: tuple[float, float] | None = Field(
        description="Pixel (column, row) in the original image; null if not visible."
    )


class GroundingAgent:
    """Locate a named object part without planning or executing robot actions."""

    def __init__(self, model: str, base_url: str | None = None) -> None:
        if not model.strip() or (model.startswith("codex:") and not model[6:].strip()):
            raise ValueError("grounding agent requires a nonempty model name")
        self.model = model
        self.base_url = base_url
        self._agent = None

    def ground(self, image: bytes, query: str) -> MolmoResult:
        """Select one visible material point or report the target as absent."""
        with Image.open(io.BytesIO(image)) as frame:
            size = frame.size
        prompt = (
            f"Locate {query} in this robot camera image. Preserve the named object "
            "part and requested spatial relation. Select a clearly visible material "
            "point away from silhouettes, occlusion and image borders. Do not "
            "substitute another object or part. Return null if it cannot be identified. "
            f"The image has {size[0]} columns and {size[1]} rows. Return point_xy as "
            "[column, row] in ORIGINAL pixel units, not normalized coordinates. "
            "Only select a point; do not use tools or execute robot actions."
        )
        if self.model.startswith("codex:"):
            selected = self._ground_codex(image, prompt)
        else:
            from pydantic_ai import Agent, BinaryContent
            from pydantic_ai.usage import UsageLimits

            from rpent.planner.base import build_api_model

            if self._agent is None:
                self._agent = Agent(
                    build_api_model(self.model, self.base_url),
                    output_type=PointSelection,
                    retries=0,
                    model_settings={"timeout": 90, "max_tokens": 1024},
                )
            selected = self._agent.run_sync(
                [prompt, BinaryContent(data=image, media_type="image/png")],
                usage_limits=UsageLimits(request_limit=1),
            ).output
        if selected.point_xy is not None and not all(
            0 <= value < bound
            for value, bound in zip(selected.point_xy, size, strict=True)
        ):
            raise ValueError("grounding agent pixel outside source image")
        return MolmoResult(
            found=selected.point_xy is not None,
            point_xy=selected.point_xy,
            image_size=size,
            answer=selected.model_dump_json(),
        )

    def _ground_codex(self, image: bytes, prompt: str) -> PointSelection:
        from rpent.planner.codex import build_probe_config, run_probe_turn

        with tempfile.TemporaryDirectory(prefix="rpent-point-agent-") as root:
            cwd = Path(root) / "work"
            home = Path(root) / "codex"
            cwd.mkdir()
            home.mkdir()
            config = build_probe_config(self.base_url)
            env, provider_overrides = _isolated_codex_environment(
                config.env, home, config.config_overrides
            )
            config = replace(
                config,
                cwd=str(cwd),
                env=env,
                config_overrides=provider_overrides
                + config.config_overrides
                + (
                    "features.shell_tool=false",
                    "features.unified_exec=false",
                    "features.view_image=false",
                    "features.view_image_tool=false",
                    "features.js_repl=false",
                    "features.code_mode=false",
                    "features.multi_agent=false",
                    "features.multi_agent_v2=false",
                    "features.image_generation=false",
                    "features.goals=false",
                    "tools.update_plan.enabled=false",
                    "tools.experimental_request_user_input.enabled=false",
                    'web_search="disabled"',
                    "apps._default.enabled=false",
                ),
            )
            _check_codex_mcp_isolation(config)
            answer = run_probe_turn(
                config,
                prompt=prompt,
                model=self.model.split(":", 1)[1],
                timeout_s=90,
                image_bytes=image,
                output_schema={
                    "type": "object",
                    "properties": {
                        "point_xy": {
                            "anyOf": [
                                {
                                    "type": "array",
                                    "items": {"type": "number"},
                                    "minItems": 2,
                                    "maxItems": 2,
                                },
                                {"type": "null"},
                            ]
                        }
                    },
                    "required": ["point_xy"],
                    "additionalProperties": False,
                },
            )
        return PointSelection.model_validate_json(answer)


def _isolated_codex_environment(
    env: dict[str, str], home: Path, config_overrides: tuple[str, ...] = ()
) -> tuple[dict[str, str], tuple[str, ...]]:
    """Reuse only model/provider settings and file credentials in a fresh home."""
    source_home = Path(env.get("CODEX_HOME", str(Path.home() / ".codex")))
    source_config = source_home / "config.toml"
    settings = {}
    if source_config.is_file():
        settings = tomllib.loads(source_config.read_text(encoding="utf-8"))
    # Explicit RPent provider overrides take precedence over these user defaults.
    overrides = []
    for key in (
        "model_provider",
        "model_providers",
        "openai_base_url",
        "chatgpt_base_url",
    ):
        value = settings.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            overrides.append(f"{key}={json.dumps(value)}")
        elif key == "model_providers":
            for name, provider in value.items():
                for field, item in provider.items():
                    # Provider entries are TOML primitives, arrays, or nested tables.
                    overrides.append(
                        f"model_providers.{json.dumps(name)}.{json.dumps(field)}="
                        + _toml_value(item)
                    )
    # Resolve RPent's overrides before deciding whether Codex login is needed.
    provider_name = settings.get("model_provider", "openai")
    providers = dict(settings.get("model_providers", {}))
    for override in config_overrides:
        values = tomllib.loads(override)
        provider_name = values.get("model_provider", provider_name)
        for name, fields in values.get("model_providers", {}).items():
            providers[name] = {**providers.get(name, {}), **fields}
    provider = providers.get(provider_name, {})
    requires_login = provider.get("requires_openai_auth", provider_name == "openai")

    auth = source_home / "auth.json"
    if auth.is_file():
        # Share normal token refresh without copying credentials into artifacts.
        (home / "auth.json").symlink_to(auth.resolve())
    elif requires_login and settings.get("cli_auth_credentials_store") in (
        "keyring",
        "auto",
    ):
        if not env.get("CODEX_API_KEY"):
            raise RuntimeError(
                "isolated grounding requires file-based Codex login or CODEX_API_KEY; "
                "OS keyring credentials cannot be reused under a temporary CODEX_HOME"
            )
    overrides.append('cli_auth_credentials_store="file"')
    return {**env, "CODEX_HOME": str(home)}, tuple(overrides)


def _toml_value(value: object) -> str:
    """Encode provider configuration values for Codex's TOML overrides."""
    if isinstance(value, dict):
        return (
            "{"
            + ", ".join(
                f"{json.dumps(key)}={_toml_value(item)}" for key, item in value.items()
            )
            + "}"
        )
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    return json.dumps(value)


def _check_codex_mcp_isolation(config: object) -> None:
    """Reject inherited enabled MCP servers before starting any model turn."""
    from openai_codex.client import CodexClient
    from openai_codex.generated.v2_all import ConfigReadResponse

    with CodexClient(config=config) as client:
        client.initialize()
        response = client.request(
            "config/read",
            {"cwd": config.cwd, "includeLayers": False},
            response_model=ConfigReadResponse,
        )
        effective = response.config.model_dump()
    servers = effective.get("mcp_servers", {})
    enabled = [name for name, server in servers.items() if server.get("enabled", True)]
    if enabled:
        raise RuntimeError(
            "grounding agent refuses inherited MCP servers: " + ", ".join(enabled)
        )
