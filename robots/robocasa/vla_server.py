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

logger = get_logger("vla_server")


class RoboCasaVLAFacade(BaseVLAFacade):
    """Loads RLDX model and exposes inference-only RPC methods."""

    def __init__(self, model_path):
        super().__init__()
        from rldx.data.embodiment_tags import EmbodimentTag
        from rldx.eval.rollout_policy import create_rldx_sim_policy

        self.policy = create_rldx_sim_policy(
            model_path,
            EmbodimentTag.GENERAL_EMBODIMENT,
            "",
            None,
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

    def get_modality_config(self):
        return {
            "video_delta_indices": self._vdi.tolist(),
            "hist_maxlen": self._hist_maxlen,
        }

    def predict(self, obs_dict, options):
        # policy.get_action returns dict[str, np.ndarray] because RLDX's
        # PolicyRuntime._decode already .cpu().numpy()s torch internally, and
        # _NumpyEncoder (http_rpc) tags numpy arrays at JSON time. If you ever
        # bypass _decode (e.g. call the model forward directly), you must
        # .cpu().numpy() the result here — _NumpyEncoder raises on torch.Tensor.
        actions, info = self.policy.get_action(obs_dict, options=options)
        return actions

    def reset_session(self, session_id):
        self.policy.reset({"session_ids": [session_id]})
        return {"ok": True}


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

    facade = RoboCasaVLAFacade(args.model_path)
    facade.serve(
        transport=args.transport,
        host=args.host,
        port=args.port,
        parent_watch=args.parent_watch,
    )


if __name__ == "__main__":
    main()
