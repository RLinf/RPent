# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.
"""Request-view history retention; raw model responses and observations stay on disk."""

from __future__ import annotations

from dataclasses import replace

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelRequest, ModelResponse, ThinkingPart

from rpent.planner.compaction import _key, images_in, retain_images
from rpent.utils.serialization import write_json


def retained_history(messages, reasoning: int, images: int):
    """Keep recent model-reply reasoning groups and image-bearing request groups."""
    replies = [
        i for i, message in enumerate(messages) if isinstance(message, ModelResponse)
    ]
    reasoning_keep = set(
        replies if reasoning == -1 else replies[-reasoning:] if reasoning else []
    )
    result, removed = [], 0
    for i, message in enumerate(messages):
        if isinstance(message, ModelResponse) and i not in reasoning_keep:
            removed += sum(isinstance(part, ThinkingPart) for part in message.parts)
            message = replace(
                message,
                parts=[
                    part for part in message.parts if not isinstance(part, ThinkingPart)
                ],
            )
        result.append(message)
    groups = []
    image_count = 0
    for i, message in enumerate(result):
        if not isinstance(message, ModelRequest):
            continue
        keys = set()
        for position, part in enumerate(message.parts):
            content = images_in(part)
            if content:
                image_count += len(content)
                keys.add(_key(i, position, part))
        if keys:
            groups.append(keys)
    kept = groups if images == -1 else groups[-images:] if images else []
    addresses = set().union(*kept) if kept else set()
    filtered = result if images == -1 else retain_images(result, addresses)
    return filtered, {
        "reasoning_parts_omitted": removed,
        "image_groups_before": len(groups),
        "image_groups_after": len(kept),
        "images_before": image_count,
        "images_after": sum(
            len(images_in(part))
            for m in filtered
            if isinstance(m, ModelRequest)
            for part in m.parts
        ),
    }


class HistoryRetention(AbstractCapability):
    """Run before compaction; never mutate the raw response or observation archive."""

    def __init__(self, toolkit):
        self.toolkit = toolkit

    async def before_model_request(self, ctx, request_context):
        tk = self.toolkit
        request_context.messages, counts = retained_history(
            request_context.messages, tk.reasoning_history, tk.image_history
        )
        counts.update(
            request=ctx.usage.requests + 1,
            reasoning_history=tk.reasoning_history,
            image_history=tk.image_history,
        )
        write_json(
            tk.output_dir / "history_retention" / f"{ctx.usage.requests + 1:04d}.json",
            counts,
        )
        for key in ("reasoning_parts_omitted", "images_before", "images_after"):
            name = key + "_request_total"
            tk.workflow_stats[name] = tk.workflow_stats.get(name, 0) + counts[key]
        return request_context
