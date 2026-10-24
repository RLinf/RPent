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

from __future__ import annotations

import asyncio

import httpx
import pytest
from pydantic_ai import ModelRequest, UserPromptPart
from pydantic_ai.models import ModelRequestParameters

from rpent.planner.base import build_api_model


def test_google_complete_generate_content_url_uses_embedded_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "offline-test-key")
    endpoint = "https://www.packyapi.com/v1beta/models/gemini-4-argon:generateContent"

    model = build_api_model("google:Argon", endpoint)

    assert type(model).__name__ == "GoogleModel"
    assert model.model_name == "gemini-4-argon"
    assert model.base_url == "https://www.packyapi.com"

    seen_urls: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {"role": "model", "parts": [{"text": "ok"}]},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 1,
                    "candidatesTokenCount": 1,
                    "totalTokenCount": 2,
                },
            },
        )

    async def request() -> None:
        api_client = model.client._api_client
        original_client = api_client._async_httpx_client
        mock_client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        api_client._async_httpx_client = mock_client
        api_client._http_options.httpx_async_client = mock_client
        try:
            response = await model.request(
                [ModelRequest(parts=[UserPromptPart(content="hello")])],
                None,
                ModelRequestParameters(),
            )
            assert response.parts[0].content == "ok"
        finally:
            await mock_client.aclose()
            await original_client.aclose()

    asyncio.run(request())

    assert seen_urls == [endpoint]
