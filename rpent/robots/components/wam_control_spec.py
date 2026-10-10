# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Platform-owned action semantics shared by clients and model adapters."""

from typing import TypedDict


class WAMControlSpec(TypedDict):
    """Describe field order and native execution mode, without implementing control.

    The versioned action_space identifies units, frame, scaling, gripper
    convention and control rate. action_type selects a native environment mode.
    """

    embodiment: str
    action_space: str
    action_schema: list[str]
    action_type: str | None
