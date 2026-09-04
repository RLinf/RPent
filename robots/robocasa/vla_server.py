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

"""RoboCasa VLA server — loads RLDX model and exposes inference calls via RPC."""

import argparse
import os

import numpy as np

from rpent.robots.components.vla_facade_base import BaseVLAFacade
from rpent.utils.logging import get_logger
from rpent.utils.rpc.rpc_facade import DEFAULT_SESSION_TIMEOUT_S

logger = get_logger("vla_server")


def _build_processor_image_transforms(processor):
    from rldx.data.augmentations import build_image_transformations_albumentations

    return build_image_transformations_albumentations(
        image_max_area=processor.image_max_area,
        image_resize_m=processor.image_resize_m,
        random_crop_fraction=getattr(processor, "random_crop_fraction", None),
        random_rotation_angle=getattr(processor, "random_rotation_angle", None),
        color_jitter_params=getattr(processor, "color_jitter_params", None),
    )


def _normalize_legacy_processor_geometry(processor):
    """Restore required image geometry omitted by older checkpoint metadata."""
    image_max_area = getattr(processor, "image_max_area", None)
    image_resize_m = getattr(processor, "image_resize_m", None)
    if image_max_area is not None and image_resize_m is not None:
        return False

    processor.image_max_area = 65536 if image_max_area is None else image_max_area
    processor.image_resize_m = 32 if image_resize_m is None else image_resize_m
    (
        processor.train_image_transform,
        processor.eval_image_transform,
    ) = _build_processor_image_transforms(processor)
    return True


class RoboCasaVLAFacade(BaseVLAFacade):
    """Loads RLDX model and exposes inference-only RPC methods."""

    def __init__(self, model_path, *, session_timeout_s=DEFAULT_SESSION_TIMEOUT_S):
        super().__init__(
            enable_sessions=True,
            session_timeout_s=session_timeout_s,
        )
        from rldx.data.embodiment_tags import EmbodimentTag
        from rldx.eval.rollout_policy import create_rldx_sim_policy

        self.policy = create_rldx_sim_policy(
            model_path,
            EmbodimentTag.GENERAL_EMBODIMENT,
            "",
            None,
        )
        if _normalize_legacy_processor_geometry(self.policy.policy.processor):
            logger.warning(
                "RLDX checkpoint omitted required image geometry; using "
                "image_max_area=65536 and image_resize_m=32"
            )
        mod = self.policy.get_modality_config()
        self._vdi = np.asarray(mod["video"].delta_indices)
        self._hist_maxlen = int(self._vdi.max() - self._vdi.min()) + 2
        print(
            f"[vla_server] policy loaded; video_delta_indices={self._vdi.tolist()} "
            f"hist_maxlen={self._hist_maxlen}",
            flush=True,
        )

    def _register_rpc(self):
        super()._register_rpc()
        self._rpc["vla.get_modality_config"] = self.get_modality_config
        self._rpc["vla.predict"] = self.predict
        self._rpc["vla.reset_session"] = self.reset_session
        self._readonly_methods.add("vla.get_modality_config")

    def get_modality_config(self, *, session_id=None):
        return {
            "video_delta_indices": self._vdi.tolist(),
            "hist_maxlen": self._hist_maxlen,
        }

    def predict(self, obs_dict, options, *, session_id):
        # policy.get_action returns dict[str, np.ndarray] because RLDX's
        # PolicyRuntime._decode already .cpu().numpy()s torch internally, and
        # _NumpyEncoder (http_rpc) tags numpy arrays at JSON time. If you ever
        # bypass _decode (e.g. call the model forward directly), you must
        # .cpu().numpy() the result here — _NumpyEncoder raises on torch.Tensor.
        # The caller's session id is injected by the RPC facade and is the
        # single source of truth for RLDX memory/RTC isolation; reject any
        # caller-supplied session_ids so it cannot shadow the server-side one.
        options = dict(options or {})
        if "session_ids" in options:
            raise ValueError(
                "predict options must not contain 'session_ids'; the server "
                "injects the caller's private session id from the RPC facade"
            )
        options["session_ids"] = [session_id]
        actions, info = self.policy.get_action(obs_dict, options=options)
        return actions

    def reset_session(self, *, session_id):
        """Reset RLDX internal state (memory/RTC) for this session.

        Does NOT destroy the session — only resets the policy state. The
        session stays live for subsequent calls. Mirror of predict's
        session_ids injection.
        """
        self.policy.reset({"session_ids": [session_id]})
        return {"ok": True}

    def _on_session_drop(self, session_id):
        self.policy.reset({"session_ids": [session_id]})


def main():
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        os.environ.setdefault("RLDX_ATTN_IMPL", "sdpa")

    p = argparse.ArgumentParser()
    p.add_argument("--transport", choices=["socket", "http"], default="http")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=0)
    p.add_argument(
        "--parent-watch",
        action="store_true",
        help="watch parent process via stdin pipe and exit when it dies",
    )
    p.add_argument(
        "--cuda-device",
        type=int,
        default=None,
        help="GPU device exposed through CUDA_VISIBLE_DEVICES.",
    )
    p.add_argument("--model-path", required=True, help="RLDX checkpoint path")
    p.add_argument(
        "--session-timeout-s",
        type=float,
        default=3600.0,
        help="session idle-timeout in seconds (default: 3600)",
    )
    p.add_argument(
        "--session-sweep-s",
        type=float,
        default=60.0,
        help="period (s) to sweep idle-expired sessions; must be > 0 (default: 60)",
    )
    args = p.parse_args()

    if args.cuda_device is not None:
        # New RLDX create_rldx_sim_policy hardcodes device=0 internally and
        # does not accept a device argument. Map physical GPU to cuda:0
        # via CUDA_VISIBLE_DEVICES instead.
        prev = os.environ.get("CUDA_VISIBLE_DEVICES")
        if prev is not None:
            logger.warning(
                "CUDA_VISIBLE_DEVICES=%s is already set; overriding with --cuda-device=%s",
                prev,
                args.cuda_device,
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)

    facade = RoboCasaVLAFacade(
        args.model_path, session_timeout_s=args.session_timeout_s
    )
    facade.serve(
        transport=args.transport,
        host=args.host,
        port=args.port,
        parent_watch=args.parent_watch,
        session_sweep_s=args.session_sweep_s,
    )


if __name__ == "__main__":
    main()
