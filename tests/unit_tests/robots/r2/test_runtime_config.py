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

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from robots.r2.runtime_config import DEFAULT_CONFIG_PATH, load_site_config


def _write(tmp_path: Path, **changes) -> Path:
    data = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    for dotted, value in changes.items():
        section, key = dotted.split("__")
        data[section][key] = value
    path = tmp_path / "site.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_example_config_keeps_base_motion_off_and_bounds_every_arm():
    site = load_site_config()

    assert site.base.allow_motion is False
    assert set(site.motion.workspace) == {"left", "right"}
    assert site.motion.workspace["left"].contains([0.4, 0.2, 0.9])
    assert not site.motion.workspace["left"].contains([0.4, 0.2, 0.3])
    assert set(site.cameras) == set(site.robot["cameras"])


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"robot__enable_dpeth": True}, "unknown keys in R2 config section 'robot'"),
        ({"motion__max_delta_m": 0.0}, "max_delta_m must be finite and positive"),
        (
            {"motion__workspace": {"left": {"low": [1, 0, 0], "high": [0, 1, 1]}}},
            "needs low < high",
        ),
        ({"base__allow_motion": True}, "requires robot.enable_base"),
    ],
)
def test_invalid_site_configs_are_rejected(tmp_path, changes, message):
    with pytest.raises(ValueError, match=message):
        load_site_config(_write(tmp_path, **changes))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("intrinsics", {"fx": 1.0, "fy": 1.0}, "intrinsics must have exactly"),
        ("distortion", [0.1, 0.0], "distortion must be five finite numbers"),
    ],
)
def test_camera_calibration_needs_complete_parameters(tmp_path, field, value, message):
    data = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    data["perception"]["cameras"]["head"][field] = value
    path = tmp_path / "site.yaml"
    path.write_text(yaml.safe_dump(data))

    with pytest.raises(ValueError, match=message):
        load_site_config(path)
