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

"""Text-to-speech backends that keep audio out of the planner's context.

Both clients expose ``async synthesize(text) -> bytes`` plus the ``suffix`` and
``mime_type`` describing what they return, so the MCP server stays backend
agnostic: OpenVoiceOS returns PCM WAV and edge-tts returns MP3.
"""

import asyncio
import base64
import io
import json
import wave
from datetime import timedelta

import httpx
from mcp import ClientSession, types
from mcp.client.streamable_http import streamable_http_client

MAX_AUDIO_BYTES = 32 * 1024 * 1024


def decode_audio(result: types.CallToolResult) -> bytes:
    """Decode the documented OVOS JSON result and require nonempty PCM WAV."""
    if result.isError:
        detail = " ".join(
            block.text
            for block in result.content
            if isinstance(block, types.TextContent)
        )
        raise RuntimeError(f"OpenVoiceOS synthesis failed: {detail[:500]}")
    payload = result.structuredContent
    if payload is None:
        blocks = [b for b in result.content if isinstance(b, types.TextContent)]
        if len(blocks) != 1:
            raise ValueError("Expected one OpenVoiceOS JSON audio result")
        payload = json.loads(blocks[0].text)
    if not isinstance(payload, dict) or payload.get("mime_type") != "audio/wav":
        raise ValueError("OpenVoiceOS must return audio/wav")
    encoded = payload.get("data")
    if not isinstance(encoded, str) or len(encoded) > 4 * ((MAX_AUDIO_BYTES + 2) // 3):
        raise ValueError("Missing or oversized OpenVoiceOS audio data")
    audio = base64.b64decode(encoded, validate=True)
    with wave.open(io.BytesIO(audio), "rb") as wav:
        frames = wav.getnframes()
        expected = frames * wav.getnchannels() * wav.getsampwidth()
        if not frames or len(wav.readframes(frames)) != expected:
            raise ValueError("OpenVoiceOS returned empty or truncated WAV audio")
    return audio


class OvosTtsClient:
    """Open and close one MCP session per synthesis request."""

    suffix = ".wav"
    mime_type = "audio/wav"

    def __init__(
        self,
        url: str,
        *,
        voice: str | None = None,
        lang: str = "zh-CN",
        timeout: float = 120,
    ) -> None:
        self.url = url
        self.voice = voice
        self.lang = lang
        self.timeout = timeout

    async def synthesize(self, text: str) -> bytes:
        """Return WAV bytes; the remote filesystem path is not used."""
        arguments = {"text": text, "lang": self.lang}
        if self.voice:
            arguments["voice"] = self.voice
        async with httpx.AsyncClient(
            timeout=self.timeout, follow_redirects=True
        ) as http:
            async with streamable_http_client(
                self.url, http_client=http, terminate_on_close=False
            ) as streams:
                async with ClientSession(
                    streams[0],
                    streams[1],
                    read_timeout_seconds=timedelta(seconds=self.timeout),
                ) as session:
                    await session.initialize()
                    result = await session.call_tool("synthesize", arguments)
        return decode_audio(result)


class EdgeTtsClient:
    """Synthesize through Microsoft Edge's online voices and return MP3 bytes.

    ``edge_tts`` is an optional import: it is only needed when this backend is
    selected. The service is remote, so every request needs network access.
    """

    suffix = ".mp3"
    mime_type = "audio/mpeg"

    def __init__(
        self,
        *,
        voice: str = "zh-CN-XiaoxiaoNeural",
        rate: str | None = None,
        volume: str | None = None,
        timeout: float = 60,
    ) -> None:
        try:
            from edge_tts import Communicate
        except ImportError as exc:
            raise ImportError(
                "the edge backend requires edge-tts; install it with `pip install edge-tts`"
            ) from exc

        self._communicate = Communicate
        self.voice = voice
        self.rate = rate
        self.volume = volume
        self.timeout = timeout

    async def synthesize(self, text: str) -> bytes:
        """Return MP3 bytes for ``text``.

        Raises:
            RuntimeError: If the service fails or no audio is produced.
        """
        options = {
            name: value
            for name, value in (("rate", self.rate), ("volume", self.volume))
            if value is not None
        }
        communicate = self._communicate(text, self.voice, **options)
        chunks: list[bytes] = []
        try:

            async def collect() -> None:
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        chunks.append(chunk["data"])

            await asyncio.wait_for(collect(), timeout=self.timeout)
        except asyncio.TimeoutError as exc:
            raise RuntimeError(
                f"edge-tts exceeded its {self.timeout:g}s timeout"
            ) from exc
        except Exception as exc:
            detail = str(exc) or type(exc).__name__
            raise RuntimeError(f"edge-tts synthesis failed: {detail[:300]}") from exc
        audio = b"".join(chunks)
        if not audio:
            raise RuntimeError("edge-tts returned no audio")
        return audio
