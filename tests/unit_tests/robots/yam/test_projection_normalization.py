# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

import numpy as np

from robots.yam.projection import world_from_depth


def test_projection_accepts_real_sense_none_model_with_extra_spacing():
    camera_meta = {
        "width": 1,
        "height": 1,
        "intrinsic_K": np.eye(3),
        "cam2world_cv": np.eye(4),
        "distortion_model": "  distortion.none  ",
        "distortion_coeffs": [0.0] * 5,
    }
    world = world_from_depth(np.array([[0.5]], dtype=np.float32), camera_meta)
    np.testing.assert_allclose(world[0, 0], [0.0, 0.0, 0.5])
