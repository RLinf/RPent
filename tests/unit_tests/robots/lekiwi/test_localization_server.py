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

"""Offline checks for localization endpoints and standalone service startup."""

from unittest.mock import Mock

import pytest
import yaml

from robots.lekiwi import localization_server
from robots.lekiwi.runtime_config import localization_service_config


@pytest.mark.parametrize("host", ["127.0.0.1", "server.example"])
def test_localization_uses_endpoint_without_deployment_settings(tmp_path, host):
    config = tmp_path / "robot.yaml"
    connection = {"endpoint": f"http://{host}:18125"}
    config.write_text(yaml.safe_dump({"localization": connection}))
    assert localization_service_config(config) == connection


@pytest.mark.parametrize("endpoint", [None, "", "  ", 18125])
def test_missing_or_invalid_endpoint_is_rejected(tmp_path, endpoint):
    config = tmp_path / "robot.yaml"
    config.write_text(yaml.safe_dump({"localization": {"endpoint": endpoint}}))
    with pytest.raises(ValueError, match="localization.endpoint"):
        localization_service_config(config)


@pytest.mark.parametrize("bind", [None, "0.0.0.0"])
def test_server_starts_without_robot_config(tmp_path, monkeypatch, bind):
    serve = Mock()
    monkeypatch.setattr(localization_server, "serve_localization", serve)
    database = tmp_path / "map.db"
    args = ["serve", "--database", str(database)]
    if bind:
        args += ["--bind", bind, "--port", "18126"]
    localization_server.main(args)
    serve.assert_called_once_with(
        database=database,
        bind=bind or "127.0.0.1",
        port=18126 if bind else 18125,
        max_processing_ms=2000.0,
    )


def test_server_releases_database_when_serving_fails(tmp_path, monkeypatch):
    localizer = Mock()
    monkeypatch.setattr(localization_server, "Localizer", Mock(return_value=localizer))
    monkeypatch.setattr(
        localization_server.LocalizationFacade,
        "serve",
        Mock(side_effect=OSError("address in use")),
    )
    with pytest.raises(OSError, match="address in use"):
        localization_server.serve_localization(
            database=tmp_path / "map.db", bind="127.0.0.1", port=18125
        )
    localizer.close.assert_called_once()
