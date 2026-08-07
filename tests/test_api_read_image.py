from __future__ import annotations

from unittest.mock import patch

import numpy as np
from pydantic_ai import ToolReturn

from rpent.planner.api_loop import read_image, read_image_text_only
from rpent.tools.state import EnvState


def _state_with_step(tmp_path) -> EnvState:
    state = EnvState(tmp_path)
    with state.record_step(state={}):
        pass
    return state


def test_read_image_returns_error_when_artifact_is_missing(tmp_path):
    state = _state_with_step(tmp_path)

    assert read_image("missing.png", state=state) == {
        "error": "image artifact 'missing.png' is not available at step -1"
    }


def test_read_image_returns_error_when_read_fails(tmp_path):
    state = _state_with_step(tmp_path)
    state.save("image.png", np.zeros((2, 2, 3), dtype=np.uint8))

    with patch.object(state, "load_bytes", side_effect=PermissionError("denied")):
        assert read_image("image.png", state=state) == {"error": "denied"}


def test_read_image_returns_step_scoped_image(tmp_path):
    state = _state_with_step(tmp_path)
    state.save("image.png", np.zeros((2, 2, 3), dtype=np.uint8))

    result = read_image("image.png", state=state)

    assert isinstance(result, ToolReturn)
    assert result.return_value == {"artifact": "image.png", "step": 0}
    assert len(result.content) == 1


def test_read_image_text_only_returns_structured_errors(tmp_path):
    state = _state_with_step(tmp_path)

    assert read_image_text_only("missing.png", state=state) == {
        "error": "image artifact 'missing.png' is not available at step -1"
    }
