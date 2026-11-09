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

"""Offline map queries through cloud RPC, Env RPC and registered agent tools."""

import io
import math
import threading
from contextlib import contextmanager
from unittest.mock import Mock

import numpy as np
import pytest
import yaml
from PIL import Image

from robots.lekiwi.env_client import LeKiwiEnvClient
from robots.lekiwi.env_server import LeKiwiEnvFacade
from robots.lekiwi.localization_server import LocalizationFacade, Localizer
from robots.lekiwi.map_view import render_map
from robots.lekiwi.navigation.runtime import NavigationRuntime
from robots.lekiwi.tools import LeKiwiPrimitives
from rpent.tools import iter_tools
from rpent.utils.rpc.http_rpc import HttpRpcClient, HttpRpcServer
from tests.unit_tests.robots.lekiwi.test_env_server import FakeBackend

NODES = [
    {"node_id": 2, "label": "door", "pose": [2.5, 2.5, 0]},
    {"node_id": 5, "label": "", "pose": [5.5, 2.5, 0]},
    {"node_id": 9, "label": "table", "pose": [8.5, 8.5, 1.0]},
]


@pytest.fixture
def localizer():
    class Native:
        def list_nodes(self):
            return list(reversed(NODES))

        def inspect_node(self, node_id):
            node = next((n for n in NODES if n["node_id"] == node_id), None)
            if node is None:
                raise ValueError("Unknown RTAB-Map node")
            image = None
            if node_id != 5:
                buffer = io.BytesIO()
                Image.new("RGB", (1600, 1200), "red").save(buffer, format="JPEG")
                image = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
            return dict(node, image=image, observed_at_s=123.5)

    result = Localizer.__new__(Localizer)
    result._native = Native()
    result.timeout = 2
    return result


@pytest.fixture
def map_yaml(tmp_path):
    pixels = np.full((20, 20), 255, dtype=np.uint8)
    pixels[0, 0], pixels[0, 1] = 0, 127
    Image.fromarray(pixels).save(tmp_path / "map.pgm")
    path = tmp_path / "map.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "image": "map.pgm",
                "resolution": 1.0,
                "origin": [0, 0, 0],
                "negate": 0,
                "occupied_thresh": 0.65,
                "free_thresh": 0.2,
            }
        )
    )
    return path


@contextmanager
def rpc_server(dispatch):
    server = HttpRpcServer(("127.0.0.1", 0), dispatch)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        worker.join(2)


def test_cloud_listing_map_limit_exact_labels_and_missing_images(localizer):
    full = localizer.list_map_nodes()
    assert full["nodes"] == NODES
    assert not full["nodes_truncated"]
    limited = localizer.list_map_nodes(limit=2)
    assert limited["nodes"] == NODES[:2]
    assert limited["nodes_truncated"]
    assert localizer.list_map_nodes(label="door")["nodes"] == [NODES[0]]
    assert localizer.list_map_nodes(label="doo")["nodes"] == []
    assert localizer.list_map_nodes(center_node_id=2, radius_m=1)["nodes"] == [NODES[0]]
    missing = localizer.inspect_map_node(5)
    assert not missing["image_available"]
    assert "image_png" not in missing
    assert missing["image_kind"] == "historical"
    with pytest.raises(ValueError, match="Unknown"):
        localizer.inspect_map_node(99)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"limit": 0},
        {"limit": True},
        {"label": ""},
        {"center_node_id": 0},
        {"radius_m": 1},
        {"center_node_id": 2, "radius_m": float("nan")},
    ],
)
def test_bad_queries_are_rejected_before_native_access(localizer, kwargs):
    localizer._native = Mock()
    with pytest.raises(ValueError):
        localizer.list_map_nodes(**kwargs)
    localizer._native.list_nodes.assert_not_called()


def test_map_orientation_crop_and_stale_robot_pose(map_yaml, localizer):
    page = localizer.list_map_nodes()
    result = render_map(
        map_yaml,
        page,
        {"ready": False, "pose": [3, 4, 0], "reason": "navigation_status_stale"},
    )
    assert result["nodes"][0]["image_pixel"] == [2, 17]
    assert not result["robot"]["valid"]
    assert "pose" not in result["robot"]
    with Image.open(io.BytesIO(result["image_png"].tobytes())) as image:
        assert image.getpixel((0, 0)) == (0, 0, 0)
        assert image.getpixel((1, 0)) == (205, 205, 205)
    page = localizer.list_map_nodes(center_node_id=2, radius_m=1)
    crop = render_map(
        map_yaml,
        page,
        {"ready": True, "pose": [2.5, 2.5, 0], "source_age_s": 0.1},
        radius_m=1,
    )
    assert crop["crop_pixels"] == [1, 16, 4, 19]
    assert crop["image_size_pixels"] == [3, 3]
    assert crop["robot"]["valid"]
    assert crop["robot"]["visible"]
    config = yaml.safe_load(map_yaml.read_text())
    config["origin"] = [10, 20, math.pi / 2]
    rotated_yaml = map_yaml.with_name("rotated.yaml")
    rotated_yaml.write_text(yaml.safe_dump(config))
    rotated = {
        "nodes": [{"node_id": 2, "pose": [7.5, 22.5, 0]}],
        "nodes_truncated": False,
    }
    view = render_map(rotated_yaml, rotated, {"ready": False})
    assert view["nodes"][0]["image_pixel"] == pytest.approx([2, 17])


def test_tools_round_trip_through_both_rpc_services(localizer, map_yaml, tmp_path):
    cloud = LocalizationFacade(localizer)
    with rpc_server(cloud._dispatch) as cloud_endpoint:
        config = tmp_path / "robot.yaml"
        connection = {"endpoint": cloud_endpoint}
        config.write_text(yaml.safe_dump({"localization": connection}))
        from robots.lekiwi.navigation.localization import wait_for_localization

        probe = HttpRpcClient(cloud_endpoint)
        try:
            wait_for_localization(probe, lambda: False, timeout_s=1)
        finally:
            probe.close()
        control = Mock()
        runtime = NavigationRuntime(
            {"robot_config": str(config), "map_yaml": str(map_yaml)}, control
        )
        backend = FakeBackend()
        for name in ("list_map_nodes", "inspect_map_node", "view_map"):
            setattr(backend, name, getattr(runtime, name))
        for name in (
            "get_stereo_frame",
            "get_navigation_state",
            "navigate_to",
            "recording_start",
            "recording_state",
            "recording_stop",
        ):
            setattr(
                backend,
                name,
                Mock(
                    side_effect=AssertionError(
                        "No device or navigation operation expected"
                    )
                ),
            )
        facade = LeKiwiEnvFacade(backend)
        with rpc_server(facade._dispatch) as endpoint:
            client = LeKiwiEnvClient(HttpRpcClient(endpoint))
            try:
                primitives = LeKiwiPrimitives(
                    env=client,
                    check_cancelled=lambda: None,
                )
                tools = {t.name: t for t in iter_tools(primitives)}
                listed = tools["list_map_nodes"]()
                assert listed.data == {
                    "nodes": [
                        {"node_id": n["node_id"], "label": n["label"]} for n in NODES
                    ]
                }
                assert tools["list_map_nodes"](label="door").data == {
                    "nodes": [{"node_id": 2, "label": "door"}]
                }
                inspected = tools["inspect_map_node"](node_id=2)
                assert inspected.data["image_kind"] == "historical"
                assert "image_png" not in inspected.data
                with Image.open(io.BytesIO(inspected.images[0])) as image:
                    assert image.size == (1024, 768)
                    assert image.getpixel((0, 0)) == pytest.approx((255, 0, 0), abs=2)
                without_image = tools["inspect_map_node"](node_id=5)
                assert without_image.images == []
                assert without_image.data["image_available"] is False
                view = tools["view_map"](center_node_id=2, radius_m=3)
                assert len(view.images) == 1
                assert view.data["center_node_id"] == 2
                assert not view.data["robot"]["valid"]
                assert runtime.daemon is None
                assert control.mock_calls == []
            finally:
                client.close()


def test_map_tool_schemas_and_rendered_prompt():
    from robots.lekiwi.prompt_bundle import system_prompt, user_prompt
    from rpent.prompt.utils import format_prompt

    definitions = {tool.name: tool for tool in iter_tools(LeKiwiPrimitives)}
    for name in ("list_map_nodes", "inspect_map_node", "view_map"):
        assert definitions[name].readonly
    listing = definitions["list_map_nodes"].input_schema["properties"]
    assert set(listing) == {"label"}
    assert "Exact existing label" in listing["label"]["description"]
    assert definitions["inspect_map_node"].input_schema["required"] == ["node_id"]
    prompt = format_prompt(system_prompt())
    assert "MAP NAVIGATION" in prompt
    assert "There is no semantic object search" in prompt
    assert "historical" in prompt
    assert "[x_m, y_m, yaw_rad]" in prompt
    rendered = format_prompt(
        user_prompt(),
        variables={
            "task_name": "visit_door",
            "instruction": "Go to door",
            "success_criteria": "Confirm the door is visible",
            "constraints": "Use map navigation only",
        },
    )
    assert "Go to door" in rendered
    assert "Use map navigation only" in rendered
    assert "{{" not in rendered


def test_map_crop_outside_grid_and_unsupported_mode_are_explicit(map_yaml, localizer):
    page = {
        "nodes": [],
        "nodes_truncated": False,
        "center": {"node_id": 100, "pose": [1000, 1000, 0]},
    }
    with pytest.raises(ValueError, match="does not overlap"):
        render_map(map_yaml, page, {}, radius_m=1)
    config = yaml.safe_load(map_yaml.read_text())
    config["mode"] = "raw"
    raw_yaml = map_yaml.with_name("raw.yaml")
    raw_yaml.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="trinary"):
        render_map(raw_yaml, localizer.list_map_nodes(), {})


def test_agent_listing_reads_graph_once_and_only_exposes_ids_and_labels(localizer):
    nodes = [
        {
            "node_id": index,
            "label": "door" if index == 205 else "",
            "pose": [index, 0, 0],
        }
        for index in range(1, 206)
    ]
    localizer._native.list_nodes = Mock(return_value=nodes)
    primitives = LeKiwiPrimitives(
        env=localizer,
        check_cancelled=lambda: None,
    )
    result = primitives.list_map_nodes()
    localizer._native.list_nodes.assert_called_once_with()
    assert result.data == {
        "nodes": [{"node_id": n["node_id"], "label": n["label"]} for n in nodes]
    }
    assert primitives.list_map_nodes(label="door").data == {
        "nodes": [{"node_id": 205, "label": "door"}]
    }
    assert primitives.list_map_nodes(label="missing").data == {"nodes": []}
