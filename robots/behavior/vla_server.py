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

"""Serve the BEHAVIOR main-task VLA through RPent's model-only RPC interface.

The loader reuses RLinf's model and policy transforms. Neither this
service nor its reset method connects to or advances the official simulator.
"""

from __future__ import annotations

import argparse
import json
import signal
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from robots.behavior.env_client import validate_native_actions
from robots.behavior.vla_policy import (
    MODEL_ARGUMENTS,
    RlinfBehaviorPolicy,
    make_model_config,
)
from rpent.robots.components.vla_facade_base import BaseVLAFacade
from rpent.utils.logging import get_logger, init_output_dir

logger = get_logger("behavior_vla_server")


def load_policy(
    checkpoint: str,
    *,
    task_prompt: str,
    device: str = "cuda:0",
    seed: int = 0,
) -> Any:
    """Load the tested RLinf VLA and its existing observation/action adapter.

    Args:
        checkpoint: Local checkpoint directory accepted by the RLinf loader.
        task_prompt: Fixed main-task instruction for this service.
        device: Torch device; defaults to the first visible CUDA device.
        seed: Torch and NumPy inference seed.
    """
    import torch
    from rlinf.models.embodiment.openpi_pytorch import get_model

    torch.manual_seed(seed)
    np.random.seed(seed)
    model_config = make_model_config(checkpoint)
    # The upstream loader converts this precision into torch.bfloat16. Do not
    # cast the policy's native actions or change its normalization here.
    if model_config.precision != "bf16":
        raise ValueError("BEHAVIOR main-task VLA requires the validated bf16 config")
    model = get_model(model_config)
    model.to(device=device)
    model.eval()
    return RlinfBehaviorPolicy(model, task_prompt=task_prompt)


class TracedPolicy:
    """Own a fixed-main-task policy and retain its actual inference outputs.

    Args:
        policy: Constructed RLinf-compatible policy owned by this wrapper.
        task_prompt: Main-task instruction; requests cannot substitute subtasks.
        output_dir: External run directory for exclusive inference/action files.
    """

    def __init__(self, policy: Any, *, task_prompt: str, output_dir: Path):
        if not isinstance(task_prompt, str) or not task_prompt.strip():
            raise ValueError("task_prompt must be a nonempty string")
        self._policy = policy
        self._task_prompt = task_prompt
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self._output_dir / "inference.jsonl"
        # Reusing an old service directory must never overwrite action traces.
        with self._log_path.open("x", encoding="utf-8"):
            pass
        self._lock = threading.Lock()
        self.calls = 0
        self.resets = 0

    def _record(self, value: dict[str, Any]) -> None:
        with self._log_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps(value, ensure_ascii=False) + "\n")

    def infer_chunk(
        self, observation: dict[str, Any], prompt: str | None = None
    ) -> np.ndarray:
        """Infer one fresh native chunk, preserving prompt, values and timing."""
        effective_prompt = self._task_prompt if prompt is None else prompt
        if effective_prompt != self._task_prompt:
            raise ValueError("VLA prompt differs from the fixed main-task prompt")
        with self._lock:
            if self._policy is None:
                raise RuntimeError("BEHAVIOR VLA policy is closed")
            self.calls += 1
            started = time.monotonic()
            record = {
                "call": self.calls,
                "event": "inference",
                "prompt": effective_prompt,
                "started_at": time.time(),
            }
            try:
                actions = self._policy.infer_chunk(observation, prompt=effective_prompt)
                record["duration_s"] = time.monotonic() - started
                record["finished_at"] = time.time()
                actions = validate_native_actions(actions, chunk=True)
                expected = (
                    MODEL_ARGUMENTS["action_chunk"],
                    MODEL_ARGUMENTS["action_dim"],
                )
                if actions.shape != expected:
                    raise ValueError(
                        f"expected VLA shape {expected}, got {actions.shape}"
                    )
                filename = f"vla_actions_{self.calls:03d}.npy"
                with (self._output_dir / filename).open("xb") as output:
                    np.save(output, actions, allow_pickle=False)
            except Exception as exc:
                record.setdefault("duration_s", time.monotonic() - started)
                record.setdefault("finished_at", time.time())
                self._record({**record, "ok": False, "error": str(exc)})
                raise
            self._record(
                {
                    **record,
                    "ok": True,
                    "shape": list(actions.shape),
                    "dtype": str(actions.dtype),
                    "minimum": float(actions.min()),
                    "maximum": float(actions.max()),
                    "actions_file": filename,
                }
            )
            return actions

    def reset(self) -> None:
        """Clear policy caches without changing the task or resetting simulation."""
        with self._lock:
            if self._policy is None:
                raise RuntimeError("BEHAVIOR VLA policy is closed")
            self._policy.reset()
            self.resets += 1
            with (self._output_dir / "policy-events.jsonl").open(
                "a", encoding="utf-8"
            ) as log:
                log.write(
                    json.dumps(
                        {
                            "event": "reset",
                            "reset": self.resets,
                            "time": time.time(),
                        }
                    )
                    + "\n"
                )

    def close(self) -> None:
        """Release the owned policy after in-flight inference has completed."""
        with self._lock:
            policy, self._policy = self._policy, None
            close = getattr(policy, "close", None)
            if callable(close):
                close()


class BehaviorVLAFacade(BaseVLAFacade):
    """Expose a borrowed policy's ``infer_chunk`` using ``vla.predict``.

    Args:
        backend: Existing policy with ``infer_chunk(observation, prompt=None)``
            returning a native [T, 23] NumPy action chunk.
        owns_backend: Close the policy when the service exits. Borrowed backends
            retain their existing caller-owned lifetime by default.
    """

    def __init__(self, backend: Any = None, *, owns_backend: bool = False):
        if backend is None:
            raise RuntimeError("BEHAVIOR requires an injected VLA policy backend")
        self._backend = backend
        self._owns_backend = owns_backend
        super().__init__()

    def _register_rpc(self) -> None:
        super()._register_rpc()
        self._rpc["vla.reset"] = self.reset

    def reset(self) -> dict[str, bool]:
        """Reset the current policy only; this service does not isolate sessions."""
        reset = getattr(self._backend, "reset", None)
        if not callable(reset):
            raise RuntimeError("BEHAVIOR VLA backend does not support policy reset")
        reset()
        return {"ok": True}

    def close(self) -> None:
        """Release an owned backend, leaving borrowed backends untouched."""
        if self._owns_backend:
            self._owns_backend = False
            close = getattr(self._backend, "close", None)
            if callable(close):
                close()

    def predict(
        self, obs: dict[str, Any], options: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Forward only the supported prompt option and validate native output."""
        if not isinstance(obs, dict):
            raise TypeError("BEHAVIOR VLA observation must be a dictionary")
        if options is None:
            options = {}
        if not isinstance(options, dict):
            raise TypeError("BEHAVIOR VLA options must be a dictionary")
        unsupported = options.keys() - {"prompt"}
        if unsupported:
            raise ValueError(f"unsupported BEHAVIOR VLA options: {sorted(unsupported)}")
        prompt = options.get("prompt")
        if prompt is not None and not isinstance(prompt, str):
            raise TypeError("BEHAVIOR VLA prompt must be a string or None")
        actions = self._backend.infer_chunk(obs, prompt=prompt)
        return {"actions": validate_native_actions(actions, chunk=True)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse model-service settings without importing optional GPU packages."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task-prompt", required=True, help="Fixed main-task text.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--transport", choices=("socket", "http"), default="socket")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--parent-watch", action="store_true")
    args = parser.parse_args(argv)
    if not args.task_prompt.strip():
        parser.error("--task-prompt must not be empty")
    if not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535")
    if not 0 <= args.seed <= 2**32 - 1:
        parser.error("--seed must be between 0 and 4294967295")
    if not args.host.strip() or not args.device.strip():
        parser.error("--host and --device must not be empty")
    return args


def main(argv: list[str] | None = None) -> None:
    """Load one main-task policy and serve predictions until clean shutdown."""
    args = parse_args(argv)
    checkpoint = args.checkpoint.expanduser().resolve()
    if not (checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError(f"RLinf checkpoint not found: {checkpoint}")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"VLA service output directory must be empty: {output_dir}"
        )
    init_output_dir(output_dir)
    manifest = {
        "checkpoint": str(checkpoint),
        "task_prompt": args.task_prompt,
        "prompt_mode": "main",
        "precision": "bf16",
        "model_arguments": MODEL_ARGUMENTS,
        "device": args.device,
        "seed": args.seed,
        "transport": args.transport,
        "host": args.host,
        "requested_port": args.port,
        "environment_execution": False,
    }
    with (output_dir / "vla-service.json").open("x", encoding="utf-8") as output:
        json.dump(manifest, output, ensure_ascii=False, indent=2)
        output.write("\n")
    logger.info("loading BEHAVIOR main-task VLA: %s", checkpoint)
    policy = load_policy(
        str(checkpoint),
        task_prompt=args.task_prompt,
        device=args.device,
        seed=args.seed,
    )
    traced = TracedPolicy(policy, task_prompt=args.task_prompt, output_dir=output_dir)
    facade = BehaviorVLAFacade(traced, owns_backend=True)

    def stop_service(signum: int, frame: Any) -> None:
        del signum, frame
        facade._shutdown_event.set()

    previous_handlers = {
        signum: signal.signal(signum, stop_service)
        for signum in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        facade.serve(
            transport=args.transport,
            host=args.host,
            port=args.port,
            parent_watch=args.parent_watch,
        )
    finally:
        facade.close()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    main()
