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

from unittest.mock import Mock

import numpy as np

from rpent.memory import MemoryManager
from rpent.session import EnvState
from rpent.tools import ToolContext, Toolkit, ToolResult, tool


@tool
def finish(status: str, summary: str, *, ctx: ToolContext) -> ToolResult:
    """Accept the requested outcome for this test toolkit."""
    return ToolResult(data={"_finish": True, "status": status, "summary": summary})


def common_toolkit(tmp_path):
    return Toolkit(
        state=EnvState(tmp_path),
        memory=MemoryManager(tmp_path / "memory"),
        robot=object(),
        output_dir=tmp_path,
        tools=(finish,),
    )


def values(frames):
    return [int(frame[0, 0, 0]) for frame in frames]


def rgb(value):
    return np.full((8, 8, 3), value, dtype=np.uint8)[:, ::-1]


def test_close_uses_toolkit_frames_and_clears_before_save(tmp_path, monkeypatch):
    toolkit = common_toolkit(tmp_path)
    toolkit.record_frame(rgb(1))
    toolkit.record_frame(rgb(2))
    saved = []

    def save(name, frames, **kwargs):
        assert toolkit._frames == []
        assert all(frame.flags.c_contiguous for frame in frames)
        saved.append((name, values(frames), kwargs))

    monkeypatch.setattr(toolkit.state, "save", save)
    toolkit.close()
    assert saved == [("episode.mp4", [1, 2], {"step": None, "fps": 20})]
    assert toolkit.execute_tool("list_dir", {}).error == "Toolkit is closed."


def test_empty_close_and_save_failure_keep_calls_closed(tmp_path, monkeypatch, caplog):
    toolkit = common_toolkit(tmp_path)
    save = Mock(side_effect=OSError("disk unavailable"))
    monkeypatch.setattr(toolkit.state, "save", save)
    toolkit.close()
    save.assert_not_called()
    toolkit = common_toolkit(tmp_path / "second")
    monkeypatch.setattr(toolkit.state, "save", save)
    toolkit.record_frame(rgb(1))
    toolkit.close()
    assert toolkit._frames == []
    assert "failed to save episode video" in caplog.text
    assert toolkit.execute_tool("list_dir", {}).error == "Toolkit is closed."
