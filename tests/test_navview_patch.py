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

import shutil
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from setuptools.build_meta import build_wheel

REPO_ROOT = Path(__file__).resolve().parents[1]
PATCH_PATH = REPO_ROOT / "robots/robocasa/patches/robosuite_navview.patch"
TARGET = Path("robosuite/models/assets/bases/omron_mobile_base.xml")
UPSTREAM_FIXTURE = """\
<mujoco model="omron_mobile_base">
    <worldbody>
        <body name="base" pos="0 0 0">
            <body name="fixed_support" pos="-0.05 0 0.50">
                <geom type="cylinder"/>
            </body>
        </body>
    </worldbody>
</mujoco>
"""


def test_navview_patch_applies_cleanly_to_upstream_without_camera(tmp_path):
    target = tmp_path / TARGET
    target.parent.mkdir(parents=True)
    target.write_text(UPSTREAM_FIXTURE, encoding="utf-8")

    subprocess.run(
        ["git", "apply", "--check", str(PATCH_PATH)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "apply", str(PATCH_PATH)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    cameras = ET.parse(target).getroot().findall(".//camera[@name='navview']")
    assert len(cameras) == 1
    assert cameras[0].attrib == {
        "name": "navview",
        "mode": "fixed",
        "pos": "0.2 0 1.6",
        "xyaxes": "0 -1 0 0.643 0 0.766",
        "fovy": "75",
    }


def test_wheel_contains_navview_delivery_assets(tmp_path, monkeypatch):
    source = tmp_path / "source"
    shutil.copytree(
        REPO_ROOT,
        source,
        ignore=shutil.ignore_patterns(".git", "logs", "__pycache__", "*.pyc"),
    )
    wheel_dir = tmp_path / "wheel"
    wheel_dir.mkdir()
    monkeypatch.chdir(source)

    wheel_path = wheel_dir / build_wheel(str(wheel_dir))

    with zipfile.ZipFile(wheel_path) as wheel:
        names = set(wheel.namelist())
    assert "robots/robocasa/patches/robosuite_navview.patch" in names
    assert "robots/robocasa/patches/README.md" in names
    assert "robots/robocasa/checkpoint_manifest.json" in names
