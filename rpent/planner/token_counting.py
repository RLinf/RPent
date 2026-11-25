# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.
"""Count the current Qwen chat request without running a vision encoder."""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import asdict, dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any


@dataclass
class PromptTokens:
    """Input length, including template/schema tokens and every image occurrence."""

    total_tokens: int
    image_tokens: int = 0
    images: list[dict[str, Any]] = field(default_factory=list)
    method: str = "checkpoint_processor"

    def to_dict(self) -> dict[str, Any]:
        """Return audit metadata without image bytes or model-visible annotations."""
        return {
            **asdict(self),
            "text_and_template_tokens": self.total_tokens - self.image_tokens,
        }


class QwenPromptCounter:
    """Use the served checkpoint's tokenizer, template and image processor.

    Loading is CPU-only and local-only. Per-image metadata is cached by asset
    SHA; occurrences still count separately. Provider usage is never reused as
    an anchor: retention, tool changes and compaction can all rewrite history.
    """

    def __init__(
        self, processor: Any, *, fingerprint: str, sglang: bool = False
    ) -> None:
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.fingerprint = fingerprint
        self.sglang = sglang
        self._images: dict[str, dict[str, Any]] = {}
        self._encoded_images: dict[str, str] = {}

    @classmethod
    def from_checkpoint(
        cls, path: str | Path, *, sglang: bool = False
    ) -> QwenPromptCounter:
        """Load the same local processor/tokenizer files used by the server."""
        from transformers import AutoProcessor

        path = Path(path)
        if not path.is_dir():
            raise ValueError(f"tokenizer checkpoint is not a local directory: {path}")
        processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        if not hasattr(processor, "_get_num_multimodal_tokens"):
            raise ValueError("this processor cannot count images from dimensions")
        digest = hashlib.sha256()
        for name in (
            "preprocessor_config.json",
            "processor_config.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "chat_template.jinja",
        ):
            source = path / name
            if source.is_file():
                digest.update(name.encode())
                digest.update(source.read_bytes())
        return cls(processor, fingerprint=digest.hexdigest(), sglang=sglang)

    def image_metadata(self, url: str) -> dict[str, Any]:
        """Measure one data-URI image using processor resize/patch/merge rules."""
        from PIL import Image

        if not url.startswith("data:") or ";base64," not in url:
            raise ValueError(
                "exact multimodal counting requires image bytes/data URIs, "
                "not a remote ImageUrl"
            )
        encoded = url.split(";base64,", 1)[1]
        encoded_digest = hashlib.sha256(encoded.encode("ascii")).hexdigest()
        if encoded_digest in self._encoded_images:
            return dict(self._images[self._encoded_images[encoded_digest]])
        raw = base64.b64decode(encoded, validate=True)
        digest = hashlib.sha256(raw).hexdigest()
        cached = self._images.get(digest)
        if cached is None:
            with Image.open(BytesIO(raw)) as image:
                width, height = image.size
            geometry = self.processor._get_num_multimodal_tokens(
                image_sizes=[(height, width)]
            )
            cached = {
                "sha256": digest,
                "width": width,
                "height": height,
                "visual_tokens": int(geometry["num_image_tokens"][0]),
                "processor_sha256": self.fingerprint,
            }
            # Store small metadata only, never image payloads.
            self._images[digest] = cached
        self._encoded_images[encoded_digest] = digest
        return dict(cached)

    def count(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        template_kwargs: dict[str, Any],
    ) -> PromptTokens:
        """Count the exact mapped chat, replacing each image placeholder once."""
        if self.sglang:
            # SGLang's Tool model serializes this wrapper default even when
            # absent on the wire. The function's keys retain their wire order.
            tools = [
                {**tool, "defer_loading": tool.get("defer_loading")} for tool in tools
            ]
        rendered_messages, images = [], []
        for message in messages:
            message = dict(message)
            if message.get("tool_calls"):
                normalized = []
                for call in message["tool_calls"]:
                    function = dict(call["function"])
                    arguments = function.get("arguments")
                    if isinstance(arguments, str):
                        try:
                            function["arguments"] = json.loads(arguments)
                        except ValueError:
                            pass
                    normalized.append({**call, "function": function})
                message["tool_calls"] = normalized
            if self.sglang:
                content = message.get("content")
                message["content"] = "" if content is None else content
                message = {
                    key: value for key, value in message.items() if value is not None
                }
            content = message.get("content")
            if isinstance(content, list):
                clean = []
                for item in content:
                    if item.get("type") == "image_url":
                        images.append(self.image_metadata(item["image_url"]["url"]))
                        clean.append({"type": "image"})
                    elif item.get("type") == "text":
                        clean.append(item)
                    else:
                        raise ValueError(
                            f"unsupported content for Qwen token counting: {item.get('type')}"
                        )
                content = clean
                if (
                    self.sglang
                    and message.get("role") == "tool"
                    and all(item.get("type") == "text" for item in content)
                ):
                    content = " ".join(item["text"] for item in content)
            rendered_messages.append({**message, "content": content})
        ids = self.tokenizer.apply_chat_template(
            rendered_messages,
            tools=tools or None,
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
        # The chat template includes one image-pad token per image plus the
        # vision boundary tokens. Replace that one token with the merged grid.
        visual = sum(image["visual_tokens"] for image in images)
        return PromptTokens(
            total_tokens=len(ids) + visual - len(images),
            image_tokens=visual,
            images=images,
            method="checkpoint_processor_sglang"
            if self.sglang
            else "checkpoint_processor",
        )
