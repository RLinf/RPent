# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.
"""Task 104: separate no-motion prediction from explicit, single-use execution."""

import time
from pathlib import Path

import numpy as np

from robots.yam.tasks import classify_episode


class VLATest:
    def __init__(self, primitives, output_dir):
        self.primitives = primitives
        self.output_dir = Path(output_dir)
        self.pending = None
        self.uncertain = False

    def infer(self):
        self.pending = None
        if self.uncertain:
            raise RuntimeError(
                "RPC outcome uncertain; exit and reconcile live state before a new test"
            )
        p = self.primitives
        sampled_at = time.monotonic()
        obs, actions, status = p.predict_actions()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"prediction_{time.time_ns()}.npz"
        np.savez_compressed(
            path,
            actions=actions,
            states=obs["states"],
            top=obs["main_images"],
            wrists=obs["extra_view_images"],
            task=np.asarray(obs["task_descriptions"]),
            episode_id=np.asarray(status["episode_id"]),
        )
        self.pending = (
            actions,
            status["episode_id"],
            status["take_action_cnt"],
            np.asarray(obs["states"])[0].copy(),
            sampled_at,
        )
        return {
            "inference_only": True,
            "shape": [1, *actions.shape],
            "evidence": str(path),
        }

    def execute(self, use_length=30):
        if self.uncertain or self.pending is None:
            raise RuntimeError("Fresh infer required; no executable prediction")
        if type(use_length) is not int or not 1 <= use_length <= 30:
            raise ValueError("use_length must be an integer in [1,30]")
        actions, episode, count, qpos, stamp = self.pending
        self.pending = None  # consume before RPC; never replay on timeout
        obs, info = self.primitives.env.read_control_state()
        status = info["episode_status"]
        if (
            not classify_episode(status)["can_continue"]
            or status["episode_id"] != episode
            or status["take_action_cnt"] != count
            or time.monotonic() - stamp > 30
            or np.max(np.abs(np.asarray(obs["state"]["joint_position"]) - qpos)) > 0.02
        ):
            raise RuntimeError(
                "Prediction stale or control not ready; infer again after checking scene"
            )
        try:
            result = self.primitives.env.chunk_step(
                actions[:use_length], expected_episode_id=episode
            )
        except Exception:
            self.uncertain = True
            raise
        return {
            "executed_actions": result[4].get("executed_actions"),
            "episode_status": result[4]["episode_status"],
        }
