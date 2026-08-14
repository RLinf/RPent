"""统一 VLA 客户端基类.

对比 ``robots/libero/``（Pi0.5）、``robots/robocasa/``（RLDX-1）、
``robots/robotwin/``（LingBot-VLA, PR #84）三个后端的 VLA client 层后提炼
的共有结构. 设计参考见
``docs/source-zh/rst_source/development/add_vla.rst``.
"""
from __future__ import annotations


class BaseVLAClient:
    """统一 VLA 客户端基类. 
    """

    _TIMEOUT_S: dict[str, float] = {"default": 30.0, "predict": 120.0}

    def __init__(self, client):
        self._client = client

    def predict(self, obs, options=None):
        """请求 VLA 推理一个 action chunk.

        Args:
            obs: 观察数据.
            options: 可选 dict.

        Returns:
            原生 actions.
        """
        return self._client.call("vla.predict", args=(obs, options),
                                timeout_s=self._TIMEOUT_S["predict"])
