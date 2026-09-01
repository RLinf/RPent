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
