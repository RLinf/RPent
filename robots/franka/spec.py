"""Static dashboard specification for single-Franka tasks."""

FRANKA_DASHBOARD_SPEC = {
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
        "display": "Franka task {task_id}",
        "output_slug": "franka_t{task_id}",
    },
    "runtime_components": (
        {"name": "env", "label": "FRANKA", "scope": "unique"},
        {"name": "vla", "label": "VLA", "scope": "shared"},
    ),
    "frame_channels": (
        {
            "name": "camera",
            "label": "external camera",
            "legacy_path_key": "image_cam_path",
        },
        {
            "name": "wrist",
            "label": "wrist camera",
            "legacy_path_key": "image_wrist_path",
        },
    ),
}
