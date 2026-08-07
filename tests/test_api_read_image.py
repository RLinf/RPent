from __future__ import annotations

from unittest.mock import patch

from pydantic_ai import ToolReturn

from rpent.planner.api_loop import read_image


def test_read_image_returns_error_when_file_is_missing(tmp_path):
    path = tmp_path / "missing.png"

    assert read_image(str(path)) == {"error": f"file not found: {path}"}


def test_read_image_returns_error_for_directory(tmp_path):
    assert read_image(str(tmp_path)) == {"error": f"is a directory: {tmp_path}"}


def test_read_image_returns_error_when_read_fails(tmp_path):
    path = tmp_path / "unreadable.png"
    path.write_bytes(b"not important")

    with patch(
        "rpent.planner.api_loop.BinaryContent.from_path",
        side_effect=PermissionError("permission denied"),
    ):
        assert read_image(str(path)) == {"error": "permission denied"}


def test_read_image_resolves_relative_path_and_returns_image(tmp_path):
    path = tmp_path / "image.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n")

    with patch("rpent.planner.api_loop.get_repo_root", return_value=tmp_path):
        result = read_image("image.png")

    assert isinstance(result, ToolReturn)
    assert result.return_value == str(path)
    assert len(result.content) == 1
