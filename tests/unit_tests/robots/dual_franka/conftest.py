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

"""Shared fixtures for offline dual-Franka tests."""

from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml

from robots.dual_franka.runtime_config import DEFAULT_CONFIG
from robots.franka.runtime_config import set_robot_config_path


@pytest.fixture
def dual_franka_robot_config(tmp_path: Path) -> Iterator[Path]:
    """Return the packaged config with calibration paths redirected to fixtures."""
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    fixtures = Path(__file__).with_name("fixtures")
    config["perception"]["calibration"] = {
        "base_camera": str(fixtures / "third_to_right_base_calib_eye_on_base.yaml"),
        "d455_camera": str(fixtures / "d455_to_right_base_eye_on_base.yaml"),
    }
    path = tmp_path / "dual_franka.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    try:
        yield path
    finally:
        set_robot_config_path(None)
