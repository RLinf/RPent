"""Initialize private state required by native RoboTwin success checkers."""

from __future__ import annotations

from typing import Any


def initialize_native_evaluator_state(task: Any) -> None:
    """Initialize checker baselines without invoking privileged expert helpers."""
    task_name = type(task).__name__
    if task_name == "open_laptop":
        from robotwin.envs.utils import get_face_prod

        face_prod = get_face_prod(task.laptop.get_pose().q, [1, 0, 0], [1, 0, 0])
        task.arm_tag = "left" if face_prod > 0 else "right"
    elif task_name == "place_object_scale":
        task.arm_tag = "right" if task.object.get_pose().p[0] > 0 else "left"
    elif task_name == "put_object_cabinet":
        object_position = task.object.get_pose().p
        task.arm_tag = "right" if object_position[0] > 0 else "left"
        task.origin_z = float(object_position[2])
