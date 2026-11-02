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

import sys
import threading

import pytest

from rpent.flywheel.cli import main
from rpent.flywheel.export import load_dataset

pytest.importorskip("lerobot")


def test_cli_exports_images_with_requested_threads(trajectory, tmp_path, monkeypatch):
    from lerobot.datasets import image_writer

    writers = set()
    write_image = image_writer.write_image

    def record_writer(image, path):
        writers.add(threading.current_thread())
        write_image(image, path)

    monkeypatch.setattr(image_writer, "write_image", record_writer)
    output = tmp_path / "export"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rpent-flywheel",
            "--input-root",
            str(trajectory["root"]),
            "--output-dir",
            str(output),
            "--image-writer-threads",
            "4",
        ],
    )
    assert main() == 0
    assert len(load_dataset(output, "train")) == 6
    assert writers
    assert threading.current_thread() not in writers
    assert all(not writer.is_alive() for writer in writers)
