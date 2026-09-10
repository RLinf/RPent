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


import numpy as np
import pytest

from rpent.memory import MemoryManager
from rpent.session import EnvState
from rpent.tools import ToolContext, Toolkit, ToolResult, tool


@tool
def finish(status: str, summary: str, *, ctx: ToolContext) -> ToolResult:
    """Accept the requested outcome for this test toolkit."""
    return ToolResult(data={"_finish": True, "status": status, "summary": summary})


@pytest.fixture
def toolkit(tmp_path, monkeypatch):
    monkeypatch.setenv("RPENT_REPO_ROOT", str(tmp_path))
    instance = Toolkit(
        state=EnvState(tmp_path / "sessions" / "session_001"),
        memory=MemoryManager(tmp_path / "memory" / "libero"),
        robot=None,
        output_dir=tmp_path,
        tools=(finish,),
    )
    yield instance
    instance.close()


def test_read_text_file_and_list_dir_use_native_results(toolkit, tmp_path):
    (tmp_path / "note.txt").write_text("tool reads", encoding="utf-8")
    text_result = toolkit.execute_tool("read_text_file", {"path": "note.txt"})
    directory_result = toolkit.execute_tool("list_dir", {})
    assert not text_result.is_error
    assert not directory_result.is_error
    assert text_result.data["content"] == "tool reads"
    assert "note.txt" in directory_result.data["files"]


def test_file_contracts_preserve_truncation_overwrite_and_utf8_counts(
    toolkit, tmp_path
):
    written = toolkit.execute_tool(
        "write_text_file", {"path": "files/note.txt", "content": "hello 世界"}
    )
    assert written.data == {
        "path": str(tmp_path / "files/note.txt"),
        "bytes_written": 12,
    }
    read = toolkit.execute_tool(
        "read_text_file", {"path": "files/note.txt", "max_chars": "7"}
    )
    assert read.data["size"] == 8
    assert (
        read.data["content"]
        == "hello 世\n\n[TRUNCATED — file is 8 chars, showed first 7]"
    )
    toolkit.execute_tool("write_text_file", {"path": "files/note.txt", "content": "x"})
    assert (tmp_path / "files/note.txt").read_text() == "x"
    (tmp_path / "files/a").mkdir()
    listed = toolkit.execute_tool("list_dir", {"path": "files"})
    assert listed.data == {
        "path": str(tmp_path / "files"),
        "count": 2,
        "files": ["a", "note.txt"],
    }
    assert toolkit.state.latest_step is None


def test_default_directory_is_bound_to_each_task_not_observation_or_global_output(
    toolkit, tmp_path, monkeypatch
):
    other_dir = tmp_path / "other"
    monkeypatch.chdir(tmp_path)
    other = Toolkit(
        state=EnvState(other_dir / "state"),
        memory=toolkit.memory,
        robot=None,
        output_dir="other",
        tools=(finish,),
    )
    monkeypatch.setattr(
        "rpent.utils.logging.get_output_dir", lambda: tmp_path / "unrelated"
    )
    try:
        monkeypatch.chdir(other_dir)
        supplied = {"ctx": {"output_dir": str(other_dir)}, "output_dir": str(other_dir)}
        assert toolkit.execute_tool("list_dir", supplied).data["path"] == str(tmp_path)
        assert other.execute_tool("list_dir", {}).data["path"] == str(other_dir)
    finally:
        other.close()


@pytest.mark.parametrize(
    "name,args",
    [
        ("read_text_file", {"path": "missing"}),
        ("write_text_file", {"path": "sessions", "content": "x"}),
        ("list_dir", {"path": "missing"}),
    ],
)
def test_file_failures_are_explicit_without_capture(toolkit, name, args):
    result = toolkit.execute_tool(name, args)
    assert result.is_error
    assert "step" not in result.data
    assert not toolkit.execute_tool("list_dir", {}).is_error


def test_png_bytes_preserve_original_artifact_and_step(toolkit):
    image = np.full((3, 4, 3), 128, dtype=np.uint8)
    name = "frame.png"
    with toolkit.state.record_step(state={}):
        toolkit.state.save(name, image)
    path = toolkit.state.artifact_path(name)
    before = path.read_bytes()
    result = toolkit.execute_tool("read_image", {"name": name})
    assert not result.is_error
    assert result.data == {"artifact": name, "step": 0}
    assert result.images == [before]
    assert path.read_bytes() == before
    assert toolkit.state.latest_step == 0


@pytest.mark.parametrize("suffix", ["jpg", "jpeg"])
def test_read_image_rejects_jpeg_artifacts(toolkit, suffix):
    name = f"frame.{suffix}"
    with toolkit.state.record_step(state={}):
        toolkit.state.save(name, np.zeros((2, 2, 3), dtype=np.uint8))
    result = toolkit.execute_tool("read_image", {"name": name})
    assert result.is_error
    assert "PNG" in result.error
    assert result.images == []


@pytest.mark.parametrize(
    "name,step",
    [
        ("frame.png", -1),
        ("frame.png", 42),
        ("../outside.png", 0),
        ("file.txt", 0),
        ("missing.png", 0),
    ],
)
def test_missing_invalid_or_nonimage_artifacts_report_errors(toolkit, name, step):
    with toolkit.state.record_step(state={}):
        toolkit.state.save("file.txt", "text")
    result = toolkit.execute_tool("read_image", {"name": name, "step": step})
    assert result.is_error
    assert result.images == []
    assert toolkit.finish_result is None


def test_file_and_image_tools_apply_memory_permissions(toolkit, tmp_path):
    published = tmp_path / "memory/libero/global/note.md"
    published.parent.mkdir(parents=True)
    published.write_text("published")
    assert (
        toolkit.execute_tool("read_text_file", {"path": str(published)}).data["content"]
        == "published"
    )
    denied = toolkit.execute_tool(
        "write_text_file", {"path": str(published), "content": "changed"}
    )
    assert denied.is_error
    foreign = tmp_path / "memory/robotwin/global/frame.png"
    foreign.parent.mkdir(parents=True)
    with toolkit.state.record_step(state={}):
        toolkit.state.save("frame.png", np.zeros((1, 1, 3), dtype=np.uint8))
    path = toolkit.state.artifact_path("frame.png")
    foreign.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(foreign)
    assert toolkit.execute_tool("read_image", {"name": "frame.png"}).is_error
