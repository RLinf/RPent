"""Static dashboard specification for dual-Franka tasks."""

DUAL_FRANKA_DASHBOARD_SPEC = {
    "task": {
        "command": "/rpent-task",
        "usage": "/rpent-task <task_id>",
        "fields": (
            {
                "name": "task_id",
                "kind": "integer",
                "minimum": 0,
                "suggestions": (0, 1),
            },
        ),
        "display": "Dual Franka task {task_id}",
        "output_slug": "dual_franka_t{task_id}",
    },
    "runtime_components": (
        {"name": "env", "label": "DUAL FRANKA", "scope": "unique"},
        {"name": "vla", "label": "VLA", "scope": "shared"},
    ),
    "frame_channels": (
        {
            "name": "left_wrist",
            "label": "left wrist camera",
            "legacy_path_key": "image_left_wrist_path",
        },
        {
            "name": "base",
            "label": "base camera",
            "legacy_path_key": "image_base_path",
        },
        {
            "name": "right_wrist",
            "label": "right wrist camera",
            "legacy_path_key": "image_right_wrist_path",
        },
    ),
}
