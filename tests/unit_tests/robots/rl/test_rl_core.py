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

"""Unit tests for RPent refactored robots.rl modules."""

from __future__ import annotations

import numpy as np
import pytest

from robots.rl.conversation import (
    CompactionConfig,
    EpisodeConversation,
    count_image_tokens,
    file_id_for,
)
from robots.rl.geometry import (
    encode_png,
    map_box,
    map_pixel,
    resize_image,
    surface_geometry,
    world_from_depth,
)
from robots.rl.toolkit import ActionBatch, DualArmBatch, MobileBatch, render_prompts
from robots.rl.units import Centimeters, validate_relations


def test_centimeters_roundtrip():
    cm = Centimeters()
    args_m = cm.payload(
        {"xyz": [12.34, -56.78, 90.12], "orientation_rpy": [10.0, -20.0, 30.0]},
        to_model=False,
    )
    assert args_m["xyz"] == pytest.approx([0.1234, -0.5678, 0.9012])
    assert args_m["orientation_rpy"] == pytest.approx([10.0, -20.0, 30.0])
    out = cm.payload({"xyz": [0.1234, -0.5678, 0.9012], "step": 1}, to_model=True)
    assert out["xyz"] == pytest.approx([12.34, -56.78, 90.12])
    assert out["step"] == 1
    with pytest.raises(ValueError):
        validate_relations("back_project", {"box": [50, 10, 20, 40]})


def test_geometry_backproject_and_png():
    rgb = np.zeros((128, 128, 3), dtype=np.uint8)
    rgb[40:80, 50:90] = (200, 100, 50)
    png = encode_png(rgb)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(file_id_for(png)) == 6
    assert count_image_tokens(png) > 0
    resized = resize_image(png, 64)
    assert resized[:8] == b"\x89PNG\r\n\x1a\n"
    assert map_pixel([32, 32], (64, 64), (128, 128)) == [65, 65]
    assert map_box([16, 16, 48, 48], (64, 64), (128, 128)) == [32, 32, 96, 96]

    depth = np.full((64, 64), 1.5, dtype=np.float32)
    meta = {
        "intrinsic_K": [[100.0, 0.0, 32.0], [0.0, 100.0, 32.0], [0.0, 0.0, 1.0]],
        "extrinsic_cam2world": np.eye(4).tolist(),
    }
    pts = world_from_depth(depth, meta)
    assert pts.shape == (64, 64, 3)
    assert pts[32, 32] == pytest.approx([0.0, 0.0, 1.5], abs=1e-5)
    geom = surface_geometry(pts.reshape(-1, 3))
    assert geom is not None
    assert geom["world_xyz"][2] == pytest.approx(1.5, abs=1e-3)


@pytest.mark.parametrize("env", ["libero", "robotwin", "robocasa", "robodojo"])
def test_prompts_all_four_envs(env: str):
    sys_p, usr_m = render_prompts(env, reasoning=True, max_actions_per_call=3)
    assert sys_p.strip()
    assert usr_m.strip()


def test_action_batch_declarations():
    for cls in (ActionBatch, DualArmBatch, MobileBatch):
        assert hasattr(cls(None), "execute_actions")


def test_conversation_validation_retry_and_compaction():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "view_env",
                "description": "Render views",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cameras": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["cameras"],
                },
            },
        }
    ]
    conv = EpisodeConversation(
        system_prompt="SYS",
        user_message="TASK",
        initial_observation_text='{"step": 0}',
        tools=tools,
        environment="libero",
        compaction=CompactionConfig(
            max_messages=6, keep_messages=2, context_window=262144
        ),
    )
    assert conv.wire_messages()[0]["role"] == "system"
    assert conv.wire_messages()[1]["role"] == "user"

    conv.append_assistant(
        content=None,
        reasoning_content="thinking",
        tool_calls=[
            {
                "id": "c0",
                "type": "function",
                "function": {"name": "view_env", "arguments": "{}"},
            }
        ],
    )
    err_kind, err_msg, args, should_retry = conv.validate_or_retry(
        [{"id": "c0", "name": "view_env", "arguments": "{}"}]
    )
    assert err_kind is not None and should_retry is True and args is None
    assert "Fix the errors and try again." in conv.wire_messages()[-1]["content"]

    png = encode_png(np.zeros((32, 32, 3), dtype=np.uint8))
    for i in range(4):
        cid = f"call_{i}"
        conv.append_assistant(
            content=None,
            tool_calls=[
                {
                    "id": cid,
                    "type": "function",
                    "function": {
                        "name": "view_env",
                        "arguments": '{"cameras": ["agentview"]}',
                    },
                }
            ],
        )
        conv.append_tool_result(
            tool_name="view_env", tool_call_id=cid, text='{"ok": true}', images=[png]
        )

    assert conv.maybe_compact() is True
    msgs = conv.wire_messages()
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert "TASK" in msgs[1]["content"]
    assert len(conv.messages) < 8


def test_binary_execute_framing_and_zero_base64_conversation():
    from robots.rl.server import pack_execute_response, unpack_execute_response

    png1 = encode_png(np.full((32, 32, 3), 10, dtype=np.uint8))
    png2 = encode_png(np.full((32, 32, 3), 200, dtype=np.uint8))
    meta = {
        "text": '{"ok": true}',
        "done": False,
        "success": False,
        "truncated": False,
        "infrastructure_error": None,
        "sam3_service_errors": [],
    }
    packed = pack_execute_response(meta, [png1, png2])
    out_meta, out_imgs = unpack_execute_response(packed)
    assert out_meta == meta
    assert out_imgs == [png1, png2]

    # Empty image list roundtrip
    out_meta0, out_imgs0 = unpack_execute_response(pack_execute_response(meta, []))
    assert out_meta0 == meta
    assert out_imgs0 == []

    # EpisodeConversation keeps raw bytes without eager Base64, caches OpenAI wire once
    conv = EpisodeConversation(
        system_prompt="SYS",
        user_message="TASK",
        initial_observation_text="{}",
        tools=[],
        environment="libero",
    )
    conv.append_tool_result(
        tool_name="view_env", tool_call_id="c1", text="{}", images=[png1, png2]
    )
    lightweight = conv.wire_messages(inline_base64=False)
    urls = [
        part["image_url"]["url"]
        for msg in lightweight
        if isinstance(msg.get("content"), list)
        for part in msg["content"]
        if part.get("type") == "image_url"
    ]
    assert len(urls) == 2
    assert all(u.startswith("rpent-image://") for u in urls)
    assert conv.images() == [png1, png2]

    openai_msgs_a = conv.wire_messages(inline_base64=True)
    openai_msgs_b = conv.wire_messages(inline_base64=True)
    assert openai_msgs_a[-1] is openai_msgs_b[-1]  # cached across turns
    assert openai_msgs_a[-1]["content"][2]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )


def test_sim_pool_replace_worker_is_idempotent():
    import asyncio
    from unittest.mock import MagicMock

    from robots.rl.server import SimPoolManager, _WorkerHandle

    async def _run():
        mgr = SimPoolManager(
            environment="libero",
            gpus=[0],
            num_workers=1,
            sim_python="python",
            sam3_endpoint=None,
        )
        spawned: list[int | None] = []

        async def _fake_spawn(gpu: int | None):
            spawned.append(gpu)
            return MagicMock()

        mgr._spawn_worker = _fake_spawn  # type: ignore[method-assign]
        proc = MagicMock()
        proc.poll.return_value = 0
        handle = _WorkerHandle(
            worker_id=0,
            gpu=0,
            proc=proc,
            reader=MagicMock(),
            writer=MagicMock(),
        )
        mgr._workers.add(handle)
        await asyncio.gather(
            mgr._replace_worker(handle),
            mgr._replace_worker(handle),
            mgr._replace_worker(handle),
        )
        assert handle.replaced is True
        assert handle not in mgr._workers
        assert spawned == [0]

    asyncio.run(_run())
