import numpy as np

from robots.robocasa.primitives import RoboCasaPrimitives


class _RenderedEnv:
    def __init__(self, frame: np.ndarray) -> None:
        self.frame = frame

    def render_camera(self, **kwargs) -> np.ndarray:
        return self.frame


def test_record_frame_keeps_render_camera_orientation() -> None:
    frame = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    primitives = RoboCasaPrimitives.__new__(RoboCasaPrimitives)
    primitives.env = _RenderedEnv(frame)
    primitives._frames = []

    primitives.record_frame()

    np.testing.assert_array_equal(primitives._frames[0], frame)
