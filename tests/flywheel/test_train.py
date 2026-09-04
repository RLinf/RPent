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

from pathlib import Path
from types import SimpleNamespace

import yaml

from rpent.flywheel.train import train_rlinf


def test_train_launcher_uses_official_rlinf_environment(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta/info.json").write_text("{}")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    rlinf_root = tmp_path / "RLinf"
    (rlinf_root / ".venv/bin").mkdir(parents=True)
    (rlinf_root / ".venv/bin/python").write_text("")
    (rlinf_root / "examples/sft").mkdir(parents=True)
    (rlinf_root / "examples/sft/train_vla_sft.py").write_text("")
    output = tmp_path / "output"

    def fake_run(command, **kwargs):
        assert command[0] == str(rlinf_root / ".venv/bin/python")
        assert command[1] == str(rlinf_root / "examples/sft/train_vla_sft.py")
        assert command[5] == "pi05_sft"
        assert "runner.max_steps=20" in command
        assert kwargs["env"]["RPENT_FLYWHEEL_DATASET"] == str(dataset)
        assert kwargs["env"]["RPENT_FLYWHEEL_CHECKPOINT"] == str(checkpoint)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("rpent.flywheel.train.subprocess.run", fake_run)
    monkeypatch.setattr(
        "rpent.flywheel.train.subprocess.check_output",
        lambda *args, **kwargs: "abc123\n",
    )
    report = train_rlinf(
        dataset=dataset,
        checkpoint=checkpoint,
        rlinf_root=rlinf_root,
        output_dir=output,
        max_steps=20,
        save_interval=5,
        cuda_device=2,
    )
    assert report["passed"] is True
    assert report["cuda_device"] == 2
    assert report["max_steps"] == 20
    assert report["rlinf_commit"] == "abc123"
    assert report["save_interval"] == 5


def test_training_config_uses_pi05_full_sft():
    config_path = Path(__file__).parents[2] / "rpent/flywheel/config/pi05_sft.yaml"
    config = yaml.safe_load(config_path.read_text())
    assert config["actor"]["model"]["is_lora"] is False
    assert config["actor"]["model"]["openpi"]["train_expert_only"] is False
    assert config["actor"]["fsdp_config"]["use_orig_params"] is False
