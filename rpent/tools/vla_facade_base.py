"""统一 VLA 后端基类.
设计参考见 ``docs/source-zh/rst_source/development/add_vla.rst``.
"""
from __future__ import annotations

from typing import Any

from rpent.utils.rpc import RpcFacade


class BaseVLAFacade(RpcFacade):
    """统一 VLA 后端基类. 只固化三者共有的接口 + 框架层.

    子类需要实现的方法:
        ``predict`` — 子类实现实际推理.
        ``__init__`` —  子类自行实现加载模型.

    RPC 路由:
        ``_dispatch`` 用注册字典（``self._rpc``）替代
        ``if method == "predict"`` 链. 子类在 ``_register_rpc`` 中注册自己
        的方法. 所有 handler 接受 ``session_id`` kwarg（VLA 不需要时忽略）.

    session 隔离模型（后端特有, 不放基类）:
        对于 session-aware 的 vla 模型，支持实现 session 隔离模型。可参考 robocasa rldx vla 的实现。
        需要实现 _on_session_drop 和 reset_session 方法；可自定义 session_timeout_s 和 session_sweep_s 参数用于定时清理过期 session。
    """

    def __init__(self, *, device: str = "cuda"):
        super().__init__()
        self.device = device
        self._model = None
        self._rpc: dict[str, Any] = {}
        self._register_rpc()

    # ---- 抽象方法 ----
    def predict(self, obs, options, *, session_id=None):
        raise NotImplementedError

    # ---- 框架层 ----
    def _register_rpc(self):
        self._rpc["vla.predict"] = self.predict

    def _dispatch(self, method: str, args: tuple, kwargs: dict, *,
                  session_id: str | None = None) -> Any:
        handler = self._rpc.get(method)
        if handler is None:
            raise ValueError(f"unknown RPC method: {method!r}")
        with self._lock:
            return handler(*args, **kwargs, session_id=session_id)
