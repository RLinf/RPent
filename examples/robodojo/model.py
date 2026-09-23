# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS.

"""XPolicyLab policy server entry point installed beside RoboDawn's controller."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from XPolicyLab.utils.model_template import ModelTemplate

from rpent.benchmarks.robodojo import RoboDojoEmbodiedClient

from .agent import AgentConfig, VLMAgent
from .vlm_client import VLMError


class Model(ModelTemplate):
    """Keep RoboDawn's physical control loop and use RPent for decisions."""

    def __init__(self, model_cfg: dict):
        self.model_cfg = dict(model_cfg or {})
        config = dict(self.model_cfg.get("vlm_agent") or {})
        config.update(json.loads(os.environ.get("VLM_AGENT_OVERRIDES", "{}")))
        self.cfg = AgentConfig(config)
        model = os.environ["ROBODOJO_LLM_MODEL"]
        base_url = os.environ["ROBODOJO_LLM_BASE_URL"]
        task = self.model_cfg.get("task_name") or "task"
        run_tag = (
            os.environ.get("VLM_AGENT_RUN_TAG")
            or f"{task}_{time.strftime('%Y%m%d_%H%M%S')}"
        )
        if re.fullmatch(r"[A-Za-z0-9._-]+", run_tag) is None or run_tag in (".", ".."):
            raise ValueError("VLM_AGENT_RUN_TAG must be a simple directory name")
        self.cfg.vlm_profile = "rpent-embodied"
        self.client = RoboDojoEmbodiedClient(
            model=model,
            base_url=base_url,
            output_dir=Path(self.cfg.log_dir) / run_tag / "rpent",
            error_type=VLMError,
        )
        self.agent = VLMAgent(self.cfg, self.client, run_tag=run_tag, task_name=task)
        print(
            f"[robodojo-embodied] task={task} model={model} logs={self.agent.run_root}",
            flush=True,
        )

    def update_obs(self, obs):
        self.agent.observe(obs)

    def get_action(self):
        return self.agent.act()

    def reset(self):
        self.agent.reset()

    def report_execution(self, obs):
        self.agent.note_execution(obs)
        return {"ok": True}

    def update_obs_batch(self, obs_list):
        raise NotImplementedError("RoboDojo policy runs one environment per worker")

    def get_action_batch(self, env_idx_list=None):
        raise NotImplementedError("RoboDojo policy runs one environment per worker")
