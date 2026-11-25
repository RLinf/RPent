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

"""Run with python -m toolkits.narration.server from the RPent checkout."""

import argparse
import asyncio
import math
import time
from pathlib import Path
from typing import Annotated, Any
from uuid import uuid4

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from rpent.utils.logging import get_logger

from .tts_client import EdgeTtsClient, OvosTtsClient

logger = get_logger("narration")


def build_server(
    *,
    output_dir: Path,
    tts: EdgeTtsClient | OvosTtsClient,
    host: str = "127.0.0.1",
    port: int = 8766,
    timeout: float = 180,
) -> FastMCP:
    """Build a text-to-speech service consumed by RPent's external MCP adapters."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("--timeout must be positive and finite")
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    server = FastMCP(
        "tts", host=host, port=port, stateless_http=True, json_response=True
    )
    lock = asyncio.Lock()

    async def generate(text: str) -> dict:
        async with lock:
            if not text.strip():
                raise ValueError("text must not be blank")
            started = time.perf_counter()
            audio = await tts.synthesize(text)
            target = output_dir / f"{uuid4().hex}{getattr(tts, 'suffix', '.wav')}"
            with target.open("xb") as stream:
                try:
                    stream.write(audio)
                except OSError:
                    target.unlink(missing_ok=True)
                    raise
            logger.info("Saved narration audio: %s", target)
            return {
                "text": text,
                "audio_path": str(target),
                "mime_type": getattr(tts, "mime_type", "audio/wav"),
                "elapsed_s": round(time.perf_counter() - started, 3),
            }

    @server.tool()
    async def synthesize(
        text: Annotated[str, Field(min_length=1, max_length=2000)],
    ) -> dict[str, Any]:
        """Convert your prepared text to a speech file, without playback.

        For image narration, inspect observations with the available image tools,
        compose a short description, then pass that description as text here.
        This tool synthesizes text only; it does not inspect images or call an LLM.
        The returned audio_path is on this service's filesystem.

        Args:
            text: The exact words to speak, without Markdown, in the language
                configured for the TTS voice. Maximum 2000 characters.
        """
        try:
            return await asyncio.wait_for(generate(text), timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise RuntimeError(
                f"Speech synthesis exceeded its {timeout:g}s timeout"
            ) from exc

    return server


def main() -> None:
    """Configure and serve the text-to-speech MCP endpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("logs/narration"))
    parser.add_argument(
        "--tts-backend",
        choices=["ovos", "edge"],
        default="edge",
        help="edge: Microsoft Edge online voices (MP3, needs network; default); "
        "ovos: local OpenVoiceOS MCP server (WAV)",
    )
    parser.add_argument("--tts-url", default="http://127.0.0.1:9666/mcp/")
    parser.add_argument(
        "--voice",
        help="OVOS voice id (e.g. piper/zh_CN-huayan-medium) or edge voice "
        "(e.g. zh-CN-XiaoxiaoNeural); defaults per backend",
    )
    parser.add_argument("--lang", default="zh-CN")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be positive and finite")
    if args.tts_backend == "edge":
        client = EdgeTtsClient(
            **({"voice": args.voice} if args.voice else {}),
            timeout=args.timeout,
        )
    else:
        client = OvosTtsClient(
            args.tts_url, voice=args.voice, lang=args.lang, timeout=args.timeout
        )
    server = build_server(
        output_dir=args.output_dir,
        tts=client,
        host=args.host,
        port=args.port,
        timeout=args.timeout,
    )
    server.run(transport="streamable-http")


if __name__ == "__main__":
    main()
