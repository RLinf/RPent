"""统一 env 后端基类.
设计参考见 ``docs/source-zh/rst_source/development/add_env.rst``.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from rpent.utils.rpc import RpcFacade


class EnvFacade(RpcFacade):
    """统一 env 后端基类. 声明共有的接口 + 框架层.

    状态缓存原则:
        server 侧**不**缓存任何观察结果. ``_last_obs`` / ``_terminated`` 等
        缓存变量只存在于 client 侧 (见 ``rpent/tools/env_client_base.py`` 的
        ``EnvClient``). server 是无状态的 (除 env 本身的物理状态外), 每次请求
        都重新读 env. 

        为什么:
        1. 支持多 client 并发，多个环境互不污染.
        2. 职责清晰 server 只负责执行并返回. client 负责缓存的策略.

        TODO: 当前 robocasa ``robots/robocasa/env_server.py`` 已部分违反此原则
        (server 端 ``_last_obs`` 用于 chunk_step 内嵌渲染). 统一设计后,
        chunk_step 内嵌渲染改为基于本次 RPC 调用内的临时局部变量, 不写 self.

    RPC 路由:
        ``_dispatch`` 用注册字典 (``self._rpc``) 替代 ``getattr`` 动态路由.
        子类在 ``_register_rpc`` 中注册自己的方法. 所有 handler 接受
        ``session_id`` kwarg (env 不需要时忽略).

    EGL 单线程:
        需要保证 egl 单线程的子类必须重写 ``serve``, 
        把所有 dispatch 派发到专用渲染线程, 保证 MuJoCo EGL context 留在
        同一线程. 可以参考 robocasa 的重写实现.

    子类特有的方法不放基类, 由子类自行定义, 通过 _register_rpc 注册.
    """

    def __init__(self):
        super().__init__()
        # server 侧不缓存 obs / terminated — 只在 client 侧缓存
        self._rpc: dict[str, Callable] = {}
        self._register_rpc()

    # ---- 生命周期 ----
    def get_env_meta(self) -> dict:
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def _register_rpc(self):
        self._rpc["env.reset"] = self.reset
        self._rpc["env.get_env_meta"] = self.get_env_meta
        self._rpc["env.close"] = self.close
        self._rpc["env.step"] = self.step
        self._rpc["env.chunk_step"] = self.chunk_step

    def _dispatch(self, method: str, args: tuple, kwargs: dict, *,
                  session_id: str | None = None) -> Any:
        handler = self._rpc.get(method)
        if handler is None:
            raise ValueError(f"unknown RPC method: {method!r}")
        with self._lock:
            return handler(*args, **kwargs)

    # ---- 功能实现 (子类必须重写) ----
    def reset(self):
        """重置 env 到初始状态. 返回初始 obs dict.
        """
        raise NotImplementedError

    def step(self, flat_action):
        """执行一步 env 动作. gym
        5-tuple ``(obs, rew, term, trunc, info)``.
        """
        raise NotImplementedError

    def chunk_step(self, flat_actions, *, return_all_frames: bool = False):
        """批量执行 N 步动作. 返回 5 元组
        ``(obs_or_list, reward, done, info, n_applied)``.

        - ``obs_or_list``: ``return_all_frames=True`` 时为 ``list[Obs]`` (每步
          一个, 携带 per-step agentview); ``False`` 时为最终 obs dict.
        - ``done``: 由后端决定语义 (term OR trunc OR success).
        - ``n_applied``: 实际执行的步数 (可能小于 chunk_size, 因 success 早停
          或 episode 结束).
        - ``return_all_frames``: 标准 perf-vs-video 开关. ``True`` 每步渲染
          agentview (高密度视频, 多 RPC); ``False`` 只渲染最终 obs (快).

        **不在 server 侧缓存 obs** — 返回值由 client 决定如何缓存 (见状态
        缓存原则).
        """
        raise NotImplementedError
