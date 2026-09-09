# Copyright 2026 The RPent Authors.
# SPDX-License-Identifier: Apache-2.0

"""Backbone revision routing must not change other robot or planner settings."""

import argparse
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from robots.robocasa import robot_spec, vla_server

REVISION = "4b9f870d1287e0d38d7eb1445e6d8c60afe66dd7"


@pytest.mark.parametrize("revision", [None, REVISION])
def test_revision_is_only_passed_to_local_vla_worker(monkeypatch, tmp_path, revision):
    captured = {}

    class Daemon:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def start(self):
            pass

    monkeypatch.setattr(robot_spec, "ProcessDaemon", Daemon)
    monkeypatch.setattr(robot_spec, "pick_free_port", lambda: 12345)
    monkeypatch.setattr(robot_spec, "HttpRpcClient", lambda *args, **kwargs: "rpc")
    args = SimpleNamespace(
        vla_endpoint=None,
        vla_model_path="checkpoint",
        vla_backbone_revision=revision,
        cuda_device=0,
    )
    robot_spec._spawn_vla_server(args, tmp_path)
    command = captured["cmd"]
    if revision is None:
        assert "--backbone-revision" not in command
    else:
        assert command[command.index("--backbone-revision") + 1] == revision
    assert "env_overrides" not in captured


def test_cli_default_and_external_endpoint_conflict():
    parser = argparse.ArgumentParser()
    robot_spec._add_cli_args(parser, False)
    args = parser.parse_args(["--task-name", "OpenDrawer"])
    assert args.vla_backbone_revision is None
    args = parser.parse_args(
        [
            "--task-name",
            "OpenDrawer",
            "--vla-backbone-revision",
            REVISION,
            "--vla-endpoint",
            "http://localhost:12345",
        ]
    )
    assert args.vla_backbone_revision == REVISION
    with pytest.raises(ValueError, match="set --backbone-revision on the external"):
        robot_spec._parse_config(args)


@pytest.mark.parametrize("revision", [None, REVISION])
def test_server_cli_forwards_backbone_revision(monkeypatch, revision):
    captured = {}

    class Facade:
        def __init__(self, model_path, **kwargs):
            captured["model_path"] = model_path
            captured.update(kwargs)

        def serve(self, **kwargs):
            captured["serve"] = kwargs

    command = ["vla_server.py", "--model-path", "checkpoint"]
    if revision is not None:
        command.extend(["--backbone-revision", revision])
    monkeypatch.setattr(sys, "argv", command)
    monkeypatch.setitem(sys.modules, "flash_attn", types.ModuleType("flash_attn"))
    monkeypatch.setattr(vla_server, "RoboCasaVLAFacade", Facade)
    vla_server.main()
    assert captured["model_path"] == "checkpoint"
    assert captured["backbone_revision"] == revision
    assert captured["serve"]["transport"] == "http"


@pytest.mark.parametrize("revision", [None, REVISION])
def test_facade_preserves_default_or_passes_exact_backbone_revision(
    monkeypatch, revision
):
    calls = []
    processor = SimpleNamespace(image_max_area=65536, image_resize_m=32)
    policy = SimpleNamespace(
        policy=SimpleNamespace(processor=processor),
        get_modality_config=lambda: {"video": SimpleNamespace(delta_indices=[-2, 0])},
    )

    def factory(*args, **kwargs):
        calls.append((args, kwargs))
        return policy

    for name in ("rldx", "rldx.data", "rldx.eval"):
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    tags = types.ModuleType("rldx.data.embodiment_tags")
    tags.EmbodimentTag = SimpleNamespace(GENERAL_EMBODIMENT="general")
    rollout = types.ModuleType("rldx.eval.rollout_policy")
    rollout.create_rldx_sim_policy = factory
    monkeypatch.setitem(sys.modules, tags.__name__, tags)
    monkeypatch.setitem(sys.modules, rollout.__name__, rollout)
    facade = vla_server.RoboCasaVLAFacade("checkpoint", backbone_revision=revision)
    assert calls == [
        (
            ("checkpoint", "general", "", None),
            {} if revision is None else {"backbone_revision": revision},
        )
    ]
    assert facade.get_modality_config()["video_delta_indices"] == [-2, 0]


def test_manifest_freezes_non_weight_support_resource():
    manifest = json.loads(
        (Path(robot_spec.__file__).parent / "eval/target50.json").read_text()
    )
    support = manifest["dependencies"]["rldx_support"]
    assert support["repository"] == "RLWRLD/RLDX-1-VLM"
    assert support["revision"] == REVISION
    assert support["file_count"] == 15
    assert support["download_weights"] is False
    assert "*.safetensors" not in support["include_patterns"]
