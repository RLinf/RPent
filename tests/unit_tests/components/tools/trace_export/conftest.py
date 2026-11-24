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

import io
import json

import pytest
from PIL import Image

from components.tools.trace_export.trace import sha256_hex

CODES = (
    'target = [0.1, 0.2, 0.3]\nrobo.move_pose(xyz=target, explanation="Move above the bowl.")\n',
    'robo.finish(status="success")\n',
)
DEFAULT_METHODS = ("move_pose", "move_delta", "set_gripper", "parallel", "finish")


def _png(color):
    image = Image.new("RGB", (8, 6), color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue(), sha256_hex(image.tobytes())


def _write_turns(root, turns):
    text = "".join(json.dumps(turn) + "\n" for turn in turns)
    (root / "turns.jsonl").write_text(text)


@pytest.fixture
def make_episode(tmp_path):
    def build(
        name="ep0",
        *,
        episode_id=None,
        robot="libero",
        accepted=True,
        truth_source="environment",
        codes=CODES,
        colors=None,
        robo_methods=DEFAULT_METHODS,
    ):
        root = tmp_path / "traces" / name
        (root / "images").mkdir(parents=True)
        colors = colors or [(10 * i, 20, 30) for i in range(1, len(codes) + 1)]
        turns = []
        for index, (code, color) in enumerate(zip(codes, colors), 1):
            data, rgb = _png(color)
            png = sha256_hex(data)
            (root / "images" / f"{png}.png").write_bytes(data)
            ref = {
                "path": f"{png}.png",
                "png_sha256": png,
                "rgb_sha256": rgb,
                "width": 8,
                "height": 6,
            }
            turns.append(
                {
                    "turn": index,
                    "system": "You control a robot with Python.",
                    "user": [
                        {"type": "text", "text": "Task: put the bowl on the stove"},
                        {"type": "image", "image": 0},
                    ],
                    "images": [ref],
                    "code": code,
                    "executed_code": code,
                    "accepted": True,
                    "interrupted": False,
                    "timed_out": False,
                    "namespace_reset": False,
                }
            )
        episode = {
            "schema": "rpent.code_policy_trace.v1",
            "episode_id": episode_id or name,
            "robot": robot,
            "model": "model-x",
            "verdict": {
                "accepted": accepted,
                "truth_source": truth_source,
                "stop_reason": "solved",
            },
        }
        if robo_methods is not None:
            episode["robo_methods"] = (
                list(robo_methods)
                if isinstance(robo_methods, (tuple, set, frozenset))
                else robo_methods
            )
        (root / "episode.json").write_text(json.dumps(episode))
        _write_turns(root, turns)
        return root

    return build


@pytest.fixture
def edit_turns():
    def edit(root, change):
        path = root / "turns.jsonl"
        turns = [json.loads(line) for line in path.read_text().splitlines()]
        change(turns)
        _write_turns(root, turns)

    return edit
