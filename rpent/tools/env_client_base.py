"""统一 env 客户端基类.
设计参考见 ``docs/source-zh/rst_source/development/add_env.rst``.
"""
from __future__ import annotations


class EnvClient:
    """统一 env 客户端基类.
    """

    _TIMEOUT_S: dict[str, float] = {"default": 30.0, "env.reset": 120.0,
                                     "env.step": 60.0, "env.chunk_step": 120.0}

    def __init__(self, client, *, expected_meta: dict):
        self._client = client
        server_meta = self._client.call("env.get_env_meta",
                                        timeout_s=self._TIMEOUT_S["default"])
        assert server_meta == expected_meta, (
            f"env_meta mismatch: expected={expected_meta!r} actual={server_meta!r}. "
            "The env_server was launched with different args than this client "
            "expects — kill the stale env_server and relaunch."
        )
        self.last_obs = self.reset()

    def reset(self):
        """重置 env, 返回初始 obs. 同时更新 ``self.last_obs`` 缓存."""
        self.last_obs = self._client.call("env.reset",
                                           timeout_s=self._TIMEOUT_S["env.reset"])
        return self.last_obs

    def step(self, flat_action):
        """执行一步 env 动作. 返回 gym
        5-tuple ``(obs, rew, term, trunc, info)``.

        同时更新 ``self.last_obs`` 缓存为返回值元组的第一个元素 (obs).
        """
        result = self._client.call("env.step", args=(flat_action,),
                                   timeout_s=self._TIMEOUT_S["env.step"])
        self.last_obs = result[0]
        return result

    def chunk_step(self, flat_actions, *, return_all_frames: bool = False):
        """批量执行 N 步动作. 返回 5 元组
        ``(obs, rew, term, trunc, info)``.

        - ``obs_or_list``: ``return_all_frames=True`` 时为 ``list[Obs]`` (每步
          一个, 携带 per-step agentview); ``False`` 时为最终 obs dict.
        - 更新 ``self.last_obs`` 缓存: 若返回 obs 是 list, 取最后一个; 否则
          直接赋值.

        ``return_all_frames`` 是标准 perf-vs-video 开关. ``True`` 每步渲染
        agentview (高密度视频); ``False`` 只渲染最终 obs (快).
        """
        result = self._client.call("env.chunk_step", args=(flat_actions,),
                                   kwargs={"return_all_frames": return_all_frames},
                                   timeout_s=self._TIMEOUT_S["env.chunk_step"])
        obs_field = result[0]
        if isinstance(obs_field, list):
            if obs_field:
                self.last_obs = obs_field[-1]
        elif obs_field is not None:
            self.last_obs = obs_field
        return result
