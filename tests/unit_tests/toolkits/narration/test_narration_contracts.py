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
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest
from mcp import types
from mcp.shared.memory import create_connected_server_and_client_session

from toolkits.narration.server import build_server, main
from toolkits.narration.tts_client import EdgeTtsClient, decode_audio


def wav_bytes() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\x00\x00" * 160)
    return buffer.getvalue()


@pytest.mark.parametrize("structured", [True, False])
def test_ovos_result_decodes_wav_without_using_remote_path(structured):
    audio = wav_bytes()
    payload = {
        "mime_type": "audio/wav",
        "data": base64.b64encode(audio).decode(),
        "path": "/nonexistent/remote/file.wav",
    }
    result = types.CallToolResult(
        content=[types.TextContent(type="text", text=json.dumps(payload))],
        structuredContent=payload if structured else None,
    )
    assert decode_audio(result) == audio


@pytest.mark.parametrize(
    "payload",
    [
        {"mime_type": "audio/mp3", "data": "abc"},
        {"mime_type": "audio/wav", "data": "not base64"},
        {"mime_type": "audio/wav", "data": base64.b64encode(b"invalid").decode()},
        {
            "mime_type": "audio/wav",
            "data": base64.b64encode(wav_bytes()[:-2]).decode(),
        },
    ],
)
def test_invalid_audio_is_rejected(payload):
    with pytest.raises((ValueError, wave.Error, EOFError)):
        decode_audio(types.CallToolResult(content=[], structuredContent=payload))


def test_ovos_error_preserves_useful_detail():
    with pytest.raises(RuntimeError, match="voice unavailable"):
        decode_audio(
            types.CallToolResult(
                isError=True,
                content=[types.TextContent(type="text", text="voice unavailable")],
            )
        )


@pytest.mark.parametrize("failure", [None, "tts", "timeout", "blank", "oversized"])
def test_mcp_synthesis_result_or_error(tmp_path, failure):
    calls = []

    async def synthesize(text):
        calls.append(text)
        if failure == "tts":
            raise RuntimeError("voice unavailable")
        if failure == "timeout":
            await asyncio.sleep(10)
        return wav_bytes()

    output = tmp_path / "output"
    server = build_server(
        output_dir=output,
        tts=SimpleNamespace(synthesize=synthesize),
        timeout=0.05 if failure == "timeout" else 5,
    )
    text = (
        " "
        if failure == "blank"
        else "x" * 2001
        if failure == "oversized"
        else "桌上有杯子。"
    )

    async def run():
        async with create_connected_server_and_client_session(
            server._mcp_server
        ) as session:
            listing = await session.list_tools()
            assert [t.name for t in listing.tools] == ["synthesize"]
            results = [await session.call_tool("synthesize", {"text": text})]
            if failure is None:
                results.append(await session.call_tool("synthesize", {"text": text}))
            return results

    results = asyncio.run(run())
    if failure:
        assert results[0].isError
        assert not list(output.iterdir())
        assert calls == ([] if failure in {"blank", "oversized"} else [text])
        if failure in {"tts", "timeout"}:
            message = "voice unavailable" if failure == "tts" else "timeout"
            assert message in results[0].content[0].text
    else:
        saved = [r.structuredContent for r in results]
        assert calls == [text, text]
        assert all(r["text"] == text for r in saved)
        assert len({r["audio_path"] for r in saved}) == 2
        for result in saved:
            assert Path(result["audio_path"]).read_bytes() == wav_bytes()
            assert "data" not in result


def test_server_saves_mp3_for_a_backend_that_reports_it(tmp_path):
    """The backend, not the server, decides the file suffix and MIME type."""

    async def synthesize(text):
        return b"ID3fake-mp3"

    output = tmp_path / "output"
    server = build_server(
        output_dir=output,
        tts=SimpleNamespace(
            synthesize=synthesize, suffix=".mp3", mime_type="audio/mpeg"
        ),
        timeout=5,
    )

    async def run():
        async with create_connected_server_and_client_session(
            server._mcp_server
        ) as session:
            return await session.call_tool("synthesize", {"text": "你好"})

    result = asyncio.run(run())

    saved = result.structuredContent
    assert saved["mime_type"] == "audio/mpeg"
    assert saved["audio_path"].endswith(".mp3")
    assert Path(saved["audio_path"]).read_bytes() == b"ID3fake-mp3"


def test_edge_client_streams_mp3(monkeypatch):
    seen = {}

    class FakeCommunicate:
        def __init__(self, text, voice, **options):
            seen.update(text=text, voice=voice, **options)

        async def stream(self):
            yield {"type": "audio", "data": b"ID3"}
            yield {"type": "WordBoundary", "data": b"ignored"}
            yield {"type": "audio", "data": b"-rest"}

    monkeypatch.setitem(
        sys.modules, "edge_tts", SimpleNamespace(Communicate=FakeCommunicate)
    )
    client = EdgeTtsClient(voice="zh-CN-XiaoyiNeural")

    audio = asyncio.run(client.synthesize("桌上有一个杯子。"))

    assert audio == b"ID3-rest"
    # Unset options are omitted: edge-tts rejects ``rate=None``.
    assert seen == {"text": "桌上有一个杯子。", "voice": "zh-CN-XiaoyiNeural"}
    assert (client.suffix, client.mime_type) == (".mp3", "audio/mpeg")


@pytest.mark.parametrize(
    ("outcome", "message"),
    [("failure", "no network"), ("empty", "no audio"), ("timeout", "0.05s timeout")],
)
def test_edge_client_reports_failures(monkeypatch, outcome, message):
    class FakeCommunicate:
        def __init__(self, *args, **kwargs):
            pass

        async def stream(self):
            if outcome == "failure":
                raise ConnectionError("no network")
            if outcome == "timeout":
                await asyncio.Event().wait()
            yield {"type": "WordBoundary", "data": b""}

    monkeypatch.setitem(
        sys.modules, "edge_tts", SimpleNamespace(Communicate=FakeCommunicate)
    )
    with pytest.raises(RuntimeError, match=message):
        asyncio.run(EdgeTtsClient(timeout=0.05).synthesize("你好"))


def test_missing_edge_package_fails_before_mcp_server_starts(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "edge_tts", None)
    monkeypatch.setattr(sys, "argv", ["server"])
    monkeypatch.setattr(
        "toolkits.narration.server.build_server",
        lambda **kwargs: pytest.fail("must check edge dependency before serving"),
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "pip install edge-tts" in capsys.readouterr().err
