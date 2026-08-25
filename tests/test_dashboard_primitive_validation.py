import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from rpent.dashboard.server import DashboardServer
from rpent.dashboard.state import DashboardState, PrimitiveArgumentError

_DASHBOARD_SPEC = {
    "task": {
        "command": "/rpent-task",
        "usage": "/rpent-task <name>",
        "fields": ({"name": "name"},),
        "display": "{name}",
        "output_slug": "{name}",
    },
    "launcher_fields": (),
    "runtime_components": (),
    "frame_channels": (
        {
            "name": "camera",
            "label": "camera",
            "artifact": "frame.png",
            "media_type": "image/png",
        },
    ),
    "primitives": ("move",),
}


class _FakeToolkit:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get_tools_spec(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "move",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "steps": {"type": "integer", "minimum": 1},
                        "xyz": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 3,
                            "maxItems": 3,
                        },
                    },
                    "required": ["steps", "xyz"],
                    "additionalProperties": False,
                },
            }
        ]

    def execute_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        self.calls.append((name, arguments))
        return SimpleNamespace(result={"ok": True})


def _bound_state(tmp_path: Path) -> tuple[DashboardState, _FakeToolkit]:
    state = DashboardState(
        run_id="dashboard-session/test",
        output_dir=tmp_path,
        dashboard_spec=_DASHBOARD_SPEC,
    )
    toolkit = _FakeToolkit()
    state.bind_toolkit(toolkit)
    return state, toolkit


@pytest.mark.parametrize(
    "arguments",
    [
        {"steps": 0, "xyz": [0.0, 0.0, 0.0]},
        {"steps": 1, "xyz": [0.0, 0.0]},
        {"steps": 1, "xyz": [0.0, "bad", 0.0]},
        {"steps": 1, "xyz": [0.0, 0.0, 0.0], "extra": True},
    ],
)
def test_dashboard_rejects_primitive_arguments_outside_schema(
    tmp_path: Path,
    arguments: dict[str, Any],
) -> None:
    state, toolkit = _bound_state(tmp_path)

    with pytest.raises(PrimitiveArgumentError, match="invalid arguments for move"):
        state.execute_primitive("move", arguments)

    assert toolkit.calls == []


def test_dashboard_executes_schema_valid_primitive_arguments(tmp_path: Path) -> None:
    state, toolkit = _bound_state(tmp_path)
    arguments = {"steps": 2, "xyz": [0.1, 0.2, 0.3]}

    result = state.execute_primitive("move", arguments)

    assert result.result == {"ok": True}
    assert toolkit.calls == [("move", arguments)]


def test_dashboard_primitive_api_returns_422_before_execution(tmp_path: Path) -> None:
    state, toolkit = _bound_state(tmp_path)
    server = DashboardServer(dashboard_spec=_DASHBOARD_SPEC)
    server.register(state)

    async def post_invalid_primitive() -> httpx.Response:
        transport = httpx.ASGITransport(app=server._app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            return await client.post(
                "/api/run/primitive",
                json={
                    "run": state.run_id,
                    "name": "move",
                    "arguments": {"steps": 0, "xyz": [0.0, 0.0, 0.0]},
                },
            )

    response = asyncio.run(post_invalid_primitive())

    assert response.status_code == 422
    assert response.json()["error"].startswith("invalid arguments for move")
    assert toolkit.calls == []
