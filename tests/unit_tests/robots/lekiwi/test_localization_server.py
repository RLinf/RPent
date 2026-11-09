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

"""Offline checks for cloud localization configuration."""

from robots.lekiwi import localization_server


def test_cloud_reads_token_from_robot_yaml(tmp_path, monkeypatch):
    config = tmp_path / "robot.yaml"
    config.write_text(
        "localization:\n  endpoint: http://localhost:18125\n  token: test-token\n"
    )
    calls = []
    monkeypatch.setattr(
        localization_server, "serve_localization", lambda **kwargs: calls.append(kwargs)
    )
    localization_server.main(
        [
            "serve",
            "--robot-config",
            str(config),
            "--database",
            str(tmp_path / "map.db"),
            "--bind",
            "127.0.0.1",
            "--port",
            "18125",
            "--peer",
            "127.0.0.1",
        ]
    )
    assert calls[0]["token"] == "test-token"
    assert calls[0]["port"] == 18125
