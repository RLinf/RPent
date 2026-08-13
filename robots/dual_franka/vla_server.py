"""RPC server wrapping the dual-Franka TCP-rot6d Pi0.5 policy."""

from __future__ import annotations

import argparse
import base64
import io
import os
import time
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from rpent.utils.logging import get_logger
from rpent.utils.rpc import RpcFacade

logger = get_logger("dual_franka_vla_server")


def build_model_cfg(model_path: str, repo_id: str) -> Any:
    """Build the RLinf OpenPI inference config for dual Franka."""
    return OmegaConf.create(
        {
            "model_type": "openpi",
            "model_path": model_path,
            "precision": None,
            "num_action_chunks": 20,
            "action_dim": 20,
            "is_lora": False,
            "lora_rank": 32,
            "use_proprio": True,
            "num_steps": 5,
            "add_value_head": False,
            "openpi": {
                "config_name": "pi05_dualfranka_tcp_rot6d",
                "num_images_in_input": 3,
                "noise_level": 0.5,
                "action_chunk": 20,
                "num_steps": 5,
                "train_expert_only": False,
                "action_env_dim": 20,
                "noise_method": "flow_sde",
                "add_value_head": False,
                "value_after_vlm": False,
                "value_vlm_mode": "mean_token",
                "detach_critic_input": True,
                "use_dsrl": False,
            },
            "openpi_data": {"repo_id": repo_id},
        }
    )


def _decode_image_block(block: dict[str, Any]) -> np.ndarray:
    import imageio.v2 as imageio

    if (block.get("format") or "png").lower() != "png":
        raise ValueError("only PNG image blocks are supported")
    data = block.get("data")
    if not isinstance(data, str) or not data:
        raise ValueError("image block missing base64 'data'")
    image = np.asarray(imageio.imread(io.BytesIO(base64.b64decode(data))))
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"image must be HxWx3 RGB; got {image.shape}")
    return image.astype(np.uint8, copy=False)


def _build_env_obs(
    instruction: str,
    images: dict[str, Any],
    state: list,
) -> dict[str, Any]:
    extras = images.get("extra")
    if "main" not in images or not isinstance(extras, list) or len(extras) != 2:
        raise ValueError(
            "dual-Franka inference requires images.main and two images.extra views"
        )
    states = np.asarray(state, dtype=np.float32)
    if states.ndim != 2 or states.shape[1] < 20:
        raise ValueError(f"state must be [B, >=20]; got shape {states.shape}")
    main = _decode_image_block(images["main"])
    extra_views = np.stack([_decode_image_block(block) for block in extras])
    return {
        "main_images": main[None],
        "wrist_images": None,
        "extra_view_images": extra_views[None],
        "states": states,
        "task_descriptions": [str(instruction)] * states.shape[0],
    }


class DualFrankaVLAFacade(RpcFacade):
    """Serve one loaded dual-Franka Pi0.5 model over RPent RPC."""

    def __init__(self, model_path: str, repo_id: str):
        super().__init__()
        from rlinf.models.embodiment.openpi import get_model as get_openpi_model

        cfg = build_model_cfg(model_path, repo_id)
        started_at = time.time()
        logger.info(
            "loading dual-Franka Pi0.5 (model_path=%s, repo_id=%s) ...",
            model_path,
            repo_id,
        )
        self._model = get_openpi_model(cfg, torch_dtype=None).cuda().eval()
        logger.info("model ready in %.1fs", time.time() - started_at)

    def _dispatch(self, method: str, args: tuple, kwargs: dict) -> Any:
        if method == "predict":
            return self.predict(*args, **kwargs)
        raise ValueError(f"unknown RPC method: {method!r}")

    def predict(
        self,
        instruction: str,
        images: dict[str, Any],
        state: list,
        mode: str = "eval",
    ) -> dict[str, Any]:
        env_obs = _build_env_obs(instruction, images, state)
        with torch.no_grad():
            actions, _ = self._model.predict_action_batch(env_obs, mode=mode)
        actions_np = (
            actions.detach().cpu().numpy()
            if isinstance(actions, torch.Tensor)
            else np.asarray(actions)
        ).astype(np.float32)
        return {
            "actions": actions_np.tolist(),
            "shape": list(actions_np.shape),
            "dtype": "float32",
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=["socket", "http"], default="http")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument(
        "--model-path",
        default=os.environ.get("PI05_CHECKPOINT_PATH"),
    )
    parser.add_argument(
        "--repo-id",
        default=os.environ.get("DUAL_FRANKA_REPO_ID"),
        help="SFT dataset repo ID used to locate norm_stats.json in the checkpoint",
    )
    parser.add_argument("--parent-watch", action="store_true")
    parser.add_argument("--cuda-device", type=int, default=None)
    args = parser.parse_args()

    if not args.model_path:
        raise RuntimeError(
            "provide --model-path or set PI05_CHECKPOINT_PATH"
        )
    if not args.repo_id:
        raise RuntimeError("provide --repo-id or set DUAL_FRANKA_REPO_ID")
    if args.cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)

    facade = DualFrankaVLAFacade(args.model_path, args.repo_id)
    facade.serve(
        transport=args.transport,
        host=args.host,
        port=args.port,
        parent_watch=args.parent_watch,
    )


if __name__ == "__main__":
    main()
