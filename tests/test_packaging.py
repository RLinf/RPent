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
import zipfile
from pathlib import Path

from setuptools import find_namespace_packages
from setuptools.build_meta import build_wheel
from setuptools.config.pyprojecttoml import load_file

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_wheel_package_discovery_includes_robot_backends():
    config = load_file(REPO_ROOT / "pyproject.toml")
    package_find = config["tool"]["setuptools"]["packages"]["find"]

    assert "robots*" in package_find["include"]
    discovered = set(
        find_namespace_packages(
            where=str(REPO_ROOT),
            include=package_find["include"],
            exclude=package_find["exclude"],
        )
    )
    assert {
        "robots",
        "robots.libero",
        "robots.robocasa",
        "robots.robotwin",
    } <= discovered


def test_built_wheel_contains_robot_backend_modules(tmp_path, monkeypatch):
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
    assert "robots/libero/env_server.py" in names
    assert "robots/robocasa/env_server.py" in names
    assert "robots/robotwin/env_server.py" in names
