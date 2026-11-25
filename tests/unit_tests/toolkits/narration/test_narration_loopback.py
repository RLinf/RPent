# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import base64
import io
import json
import socket
import threading
import time
import wave
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import uvicorn
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.server.fastmcp import FastMCP
from pydantic_ai import BinaryContent, ToolReturnPart, UserPromptPart
from pydantic_ai.models.function import DeltaToolCall, FunctionModel

from rpent.dashboard.events import NullDashboardEventSink
from rpent.planner.api_loop import ApiAgentLoop
from rpent.session import EnvState
from rpent.tools.common import CommonTools
from rpent.tools.toolkit import Toolkit
from rpent.utils.mcp_servers import load_mcp_config
from toolkits.narration.server import build_server
from toolkits.narration.tts_client import OvosTtsClient


@contextmanager
def http_server(mcp):
    """Serve a real MCP transport on an ephemeral loopback port."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        server = uvicorn.Server(
            uvicorn.Config(mcp.streamable_http_app(), log_level="error")
        )
        thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]})
        thread.start()
        try:
            deadline = time.monotonic() + 5
            while (
                not server.started and thread.is_alive() and time.monotonic() < deadline
            ):
                time.sleep(0.01)
            assert server.started
            yield f"http://127.0.0.1:{port}/mcp"
        finally:
            server.should_exit = True
            thread.join(timeout=5)
            assert not thread.is_alive()


@pytest.mark.parametrize("caller", ["mcp_client", "planner"])
def test_two_http_mcp_hops_save_audio(tmp_path, caller, monkeypatch):
    pytest.importorskip("PIL.Image")
    state = EnvState(tmp_path / "state")
    with state.record_step(state={}):
        state.save("scene.png", np.zeros((4, 4, 3), dtype=np.uint8))
    picture = state.artifact_path("scene.png", step=0)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 160)
    audio = buffer.getvalue()
    calls = []
    ovos = FastMCP("fake-ovos", stateless_http=True, json_response=True)

    @ovos.tool()
    def synthesize(
        text: str, voice: str | None = None, lang: str | None = None
    ) -> dict:
        calls.append((text, voice, lang))
        return {
            "mime_type": "audio/wav",
            "data": base64.b64encode(audio).decode(),
            "path": "/remote/unused.wav",
        }

    async def run(url):
        async with streamable_http_client(url) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()
                result = await session.call_tool("synthesize", {"text": "黑色画面。"})
                assert not result.isError, result
                return result.structuredContent

    def run_planner(url):
        class NarrationToolkit(Toolkit):
            def _register_common_tools(self):
                self.add_tool(CommonTools(memory=self._memory).finish)

        monkeypatch.setattr("rpent.planner.api_loop.substitute", lambda value: value)
        monkeypatch.setattr("rpent.tools.toolkit.substitute", lambda value: value)
        monkeypatch.setenv("PYDANTIC_AI_NO_BANNER", "1")
        events = NullDashboardEventSink()
        toolkit = NarrationToolkit(
            dashboard_events=events, memory=SimpleNamespace(), state=state
        )
        captured = {}

        async def model(messages, info):
            returned = [
                p for m in messages for p in m.parts if isinstance(p, ToolReturnPart)
            ]
            if not returned:
                name, args = "read_image", {"name": "scene.png", "step": 0}
            elif returned[-1].tool_name == "read_image":
                images = [
                    item
                    for message in messages
                    for part in message.parts
                    if isinstance(part, (ToolReturnPart, UserPromptPart))
                    and isinstance(part.content, list)
                    for item in part.content
                    if isinstance(item, BinaryContent)
                ]
                assert images[0].data == picture.read_bytes()
                # The planner model supplies the description; the TTS tool only gets text.
                name, args = "tts__synthesize", {"text": "黑色画面。"}
            else:
                name, args = "finish", {"status": "success", "summary": "Audio saved"}
                content = returned[-1].content
                assert returned[-1].tool_name == "tts__synthesize"
                captured.update(json.loads(content[0])["structured_content"])
            yield {0: DeltaToolCall(name=name, json_args=json.dumps(args))}

        config = tmp_path / "mcp.yaml"
        config.write_text(
            f"mcp_servers:\n  tts:\n    url: {url}\n    tools: [synthesize]\n"
        )
        toolkit.add_mcp_servers(load_mcp_config(config))
        planner = ApiAgentLoop(
            model=FunctionModel(stream_function=model),
            dashboard_events=events,
        )
        try:
            result = planner.solve(
                system_prompt="Describe the image aloud and finish.",
                user_message="Narrate scene.png at step 0.",
                toolkit=toolkit,
                max_turns=3,
            )
            assert result.error is None, result.error
            assert result.finish_result["status"] == "success"
            assert base64.b64encode(audio).decode() not in json.dumps(result.messages)
            return captured
        finally:
            toolkit.close()

    with http_server(ovos) as tts_url:
        narration = build_server(
            output_dir=tmp_path / "audio",
            tts=OvosTtsClient(tts_url, voice="test-voice"),
        )
        with http_server(narration) as narration_url:
            result = (
                run_planner(narration_url)
                if caller == "planner"
                else asyncio.run(run(narration_url))
            )
    assert calls == [("黑色画面。", "test-voice", "zh-CN")]
    assert result["text"] == "黑色画面。"
    assert Path(result["audio_path"]).read_bytes() == audio
    assert "data" not in result


def test_api_narrator_reads_and_speaks_in_the_background(tmp_path, monkeypatch):
    pytest.importorskip("PIL.Image")
    state = EnvState(tmp_path / "state")
    with state.record_step(state={}):
        state.save("scene.png", np.zeros((4, 4, 3), dtype=np.uint8))
    picture = state.artifact_path("scene.png", step=0)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 160)
    audio = buffer.getvalue()
    calls = []
    spoken = threading.Event()
    ovos = FastMCP("fake-ovos", stateless_http=True, json_response=True)

    @ovos.tool()
    def synthesize(
        text: str, voice: str | None = None, lang: str | None = None
    ) -> dict:
        calls.append((text, voice, lang))
        return {
            "mime_type": "audio/wav",
            "data": base64.b64encode(audio).decode(),
            "path": "/remote/unused.wav",
        }

    def run_planner(url):
        class NarrationToolkit(Toolkit):
            def _register_common_tools(self):
                self.add_tool(CommonTools(memory=self._memory).finish)

        monkeypatch.setattr("rpent.planner.api_loop.substitute", lambda value: value)
        monkeypatch.setattr("rpent.tools.toolkit.substitute", lambda value: value)
        monkeypatch.setenv("PYDANTIC_AI_NO_BANNER", "1")
        events = NullDashboardEventSink()
        toolkit = NarrationToolkit(
            dashboard_events=events, memory=SimpleNamespace(), state=state
        )

        async def main_model(messages, info):
            returned = [
                p for m in messages for p in m.parts if isinstance(p, ToolReturnPart)
            ]
            if not returned:
                yield {
                    0: DeltaToolCall(
                        name="start_scene_narrator", json_args=json.dumps({})
                    )
                }
                return
            assert await asyncio.to_thread(spoken.wait, 5)
            yield {
                0: DeltaToolCall(
                    name="finish",
                    json_args=json.dumps(
                        {"status": "success", "summary": "Audio saved"}
                    ),
                )
            }

        async def subagent_model(messages, info):
            returned = [
                p for m in messages for p in m.parts if isinstance(p, ToolReturnPart)
            ]
            if not returned:
                name, args = "read_image", {"name": "scene.png", "step": 0}
            elif returned[-1].tool_name == "read_image":
                images = [
                    item
                    for message in messages
                    for part in message.parts
                    if isinstance(part, (ToolReturnPart, UserPromptPart))
                    and isinstance(part.content, list)
                    for item in part.content
                    if isinstance(item, BinaryContent)
                ]
                assert images[0].data == picture.read_bytes()
                name, args = "tts__synthesize", {"text": "黑色画面。"}
            else:
                assert returned[-1].tool_name == "tts__synthesize"
                assert (
                    Path(
                        json.loads(returned[-1].content[0])["structured_content"][
                            "audio_path"
                        ]
                    ).read_bytes()
                    == audio
                )
                spoken.set()
                yield "Narration finished."
                return
            yield {0: DeltaToolCall(name=name, json_args=json.dumps(args))}

        config = tmp_path / "mcp.yaml"
        config.write_text(
            f"mcp_servers:\n  tts:\n    url: {url}\n    tools: [synthesize]\n"
        )
        toolkit.add_mcp_servers(load_mcp_config(config))
        planner = ApiAgentLoop(
            model=FunctionModel(stream_function=main_model),
            subagent_name="narrator",
            subagent_model=FunctionModel(stream_function=subagent_model),
            dashboard_events=events,
        )
        try:
            result = planner.solve(
                system_prompt="Narrate the scene and finish.",
                user_message="Narrate scene.png at step 0.",
                toolkit=toolkit,
                max_turns=3,
            )
            assert result.error is None, result.error
            assert result.finish_result["status"] == "success"
            assert base64.b64encode(audio).decode() not in json.dumps(result.messages)
        finally:
            toolkit.close()

    with http_server(ovos) as tts_url:
        narration = build_server(
            output_dir=tmp_path / "audio",
            tts=OvosTtsClient(tts_url, voice="test-voice"),
        )
        with http_server(narration) as narration_url:
            run_planner(narration_url)

    assert calls == [("黑色画面。", "test-voice", "zh-CN")]
