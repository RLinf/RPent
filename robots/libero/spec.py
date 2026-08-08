"""Static LIBERO extension specs."""

LIBERO_SUITE_NAMES = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_90",
    "libero_object_task",
    "libero_object_swap",
    "libero_object_lan",
    "libero_goal_task",
    "libero_goal_swap",
    "libero_goal_lan",
    "libero_spatial_task",
    "libero_spatial_swap",
    "libero_spatial_lan",
    "libero_10",
    "libero_10_task",
    "libero_10_swap",
    "libero_10_lan",
)

LIBERO_DASHBOARD_SPEC = {
    "task": {
        "command": "/rpent-task",
        "usage": "/rpent-task <suite> <task> <seed>",
        "fields": (
            {"name": "suite", "suggestions": LIBERO_SUITE_NAMES},
            {"name": "task", "kind": "integer", "minimum": 0},
            {"name": "seed", "kind": "integer", "minimum": 0},
        ),
        "display": "{suite} / task {task} / seed {seed}",
        "output_slug": "{suite}_t{task}_s{seed}",
    },
    "runtime_components": (
        {"name": "env", "label": "ENV", "scope": "task"},
        {"name": "vla", "label": "VLA"},
        {"name": "sam3", "label": "SAM3"},
    ),
    "frame_channels": (
        {
            "name": "camera",
            "label": "fixed camera",
            "legacy_path_key": "image_cam_path",
        },
        {
            "name": "wrist",
            "label": "wrist camera",
            "legacy_path_key": "image_wrist_path",
        },
    ),
}
