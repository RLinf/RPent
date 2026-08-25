添加 env 后端
==============

在 RPent 中，*env 后端* 把机器人模拟器（gym / robosuite / 外部物理引擎）
包装成 RPC 服务，对上层 primitives 层暴露统一的 ``reset`` / ``step`` /
``chunk_step`` 接口。本页讲如何基于统一基类
（:mod:`rpent.tools.env_facade_base` / :mod:`rpent.tools.env_client_base`）
添加新的 env 后端。

.. contents::
   :local:
   :depth: 2

重要规则
--------

**1. client / server 侧分工.**

server 侧负责包装 env 的底层执行并返回, 核心执行逻辑和状态存储由 client 侧负责.
server 侧需要保证, 非 readonly 的操作必须串行执行.
``_last_obs`` / ``_terminated`` 等缓存变量**只存在于 client 侧**.
server 是无状态的（除 env 本身的物理状态外），每次请求都重新读 env.
``EnvFacade.__init__`` **不**初始化 ``self.last_obs`` /
``self._terminated``; ``EnvFacade.chunk_step`` 的默认实现也**不**写这些
字段（基类直接抛 ``NotImplementedError``, 子类实现时同样不应写）.

原因:

- **多 client 并发**: 多个 ``RpcClient`` 连到同一 server 时, 每个 client
  维护自己的缓存, 互不污染. server 缓存会被并发 client 互相覆盖.
- **server 重启恢复**: server 崩溃重启后, client 重新 ``reset`` 即可重建
  状态. server 缓存则会在重启后失效, 但 client 不知情, 继续用旧缓存.
- **职责清晰**: server 只负责"执行 env 动作并返回 obs", client 负责
  "决定如何使用 obs 并缓存".

如果 ``chunk_step`` 内嵌渲染, 渲染结果应基于本次 RPC 调用内的临时
局部变量, 不要写 ``self``.

**2. RPC 路由用注册字典.**

``_dispatch`` 用注册字典（``self._rpc``）替代 ``getattr`` 动态路由.
子类在 ``_register_rpc`` 中注册自己的方法. 子类重写
``_register_rpc`` 时应先调 ``super()._register_rpc()`` 再追加自己的
handler, 避免漏注册基类已注册的方法.

**3. expected_meta 握手校验.**

client 构造时传入 ``expected_meta``（调用方对 server 配置的预期）,
``EnvClient.__init__`` 启动后立即调 ``env.get_env_meta`` 拿 server 实际
配置, 逐项校验, 不匹配直接 assert.

原因:

- **fail-fast**: 配置不匹配（如 ``camera_h`` / ``camera_w`` /
  ``action_dim`` 对不上）在握手阶段就暴露, 而不是跑到 mid-rollout
  才崩, 避免浪费 GPU 算力后才发现 client / server 版本错位.
- **职责清晰**: server 只如实报告自己的配置, client 负责判断配置
  是否符合上层调用方预期. 校验在 client 侧, 不污染 server.
- **解耦**: server 配置不上提到 client 基类硬编码——调用方按自己
  需求传 ``expected_meta``, 不同调用方可以有不同预期.

**4. (可选) EGL 单线程派发（仅 GPU 渲染后端）.**

多 GPU/EGL 环境下, MuJoCo EGL context 必须留在同一线程. 如果你的
后端用 EGL 渲染, 重写 ``serve`` 把 ``_dispatch`` 派发到专用渲染线程
（用 ``work_queue`` + ``render_loop`` 模式）. 不需要 EGL 的后端直接
用基类继承的 ``serve``.

基类
----

统一基类已落地到 ``rpent/tools/`` 下两个文件:

- :mod:`rpent.tools.env_facade_base` —
  :class:`~rpent.tools.env_facade_base.EnvFacade`（server 侧）
- :mod:`rpent.tools.env_client_base` —
  :class:`~rpent.tools.env_client_base.EnvClient`（client 侧）

EnvFacade（server 侧）
~~~~~~~~~~~~~~~~~~~~~

继承 :class:`~rpent.utils.rpc.RpcFacade`, 提供框架层:

- ``__init__()``: 初始化 ``self._rpc = {}`` 并调 ``_register_rpc()``
- 抽象方法 ``reset`` / ``step`` / ``chunk_step`` / ``get_env_meta`` /
  ``close`` — 子类必须实现
- ``_register_rpc()``: 默认注册 ``env.reset`` / ``env.step`` /
  ``env.chunk_step`` / ``env.get_env_meta`` / ``env.close``, 子类可重写
  追加（重写时先调 ``super()._register_rpc()``）
- ``_dispatch(self, method, args, kwargs, *, session_id=None)``:
  从 ``self._rpc`` 取 handler, 用 ``self._lock`` 串行化
- ``serve``: 继承自 :class:`~rpent.utils.rpc.RpcFacade`. 需要 EGL
  单线程的子类重写 ``serve``, 把 ``_dispatch`` 派发到专用渲染线程

EnvClient（client 侧）
~~~~~~~~~~~~~~~~~~~~~

薄包装 :class:`~rpent.utils.rpc.RpcClient`:

- ``__init__(self, client, *, expected_meta: dict)``: 接受一个
  ``RpcClient`` 实例, 调 ``env.get_env_meta`` 校验 server 返回的 meta
  与 ``expected_meta`` 一致, 然后调 ``reset()`` 初始化 ``self.last_obs``
- ``_TIMEOUT_S = {"default": 30.0, "env.reset": 120.0,
  "env.step": 60.0, "env.chunk_step": 120.0}`` — 按方法分配超时
- ``last_obs``: 公开属性, client 侧缓存. ``reset`` / ``step`` /
  ``chunk_step`` 都会更新它, 调用方可直接访问 ``client.last_obs``
  避免额外 RPC
- ``reset()`` / ``step(flat_action)`` / ``chunk_step(flat_actions, *,
  return_all_frames=False)`` — 都更新 ``self.last_obs``

接入步骤
--------

1. 继承 :class:`~rpent.tools.env_facade_base.EnvFacade`, 实现抽象方法
   ``reset`` / ``step`` / ``chunk_step`` / ``get_env_meta`` / ``close``.
   ``step`` / ``chunk_step`` 的入参与你的 env 的 action 空间一致, 返回值
   为 ``(obs, reward, term, trunc, info)`` /
   ``(obs_or_list, reward, term, trunc, info)``——具体怎么从你的 env
   拿到这些字段由子类决定（不同后端的 env.step 返回元组长度、done/term
   语义都不同）. ``chunk_step`` 的 ``return_all_frames`` 控制
   ``obs_or_list`` 是最终 obs 还是每步 obs 列表（``False`` 省 GPU,
   ``True`` 用于录像）:

   .. code-block:: python

      from rpent.tools.env_facade_base import EnvFacade

      class MyEnvFacade(EnvFacade):
          def __init__(self, env_cfg):
              super().__init__()
              self._env = build_my_env(env_cfg)

          def get_env_meta(self) -> dict:
              return {
                  "task_name": self._env.task_name,
                  "camera_h": 256,
                  "camera_w": 256,
                  "action_dim": self._env.action_dim,
              }

          def reset(self):
              return self._env.reset()

          def step(self, flat_action):
              # 子类按 env 语义实现, 返回
              # (obs, reward, term, trunc, info)
              ...

          def chunk_step(self, flat_actions, *, return_all_frames=False):
              # 子类按 env 语义实现, 返回
              # (obs_or_list, reward, term, trunc, info)
              ...

          def close(self):
              self._env.close()

2. 注册额外 RPC 方法（可选）. 如果后端需要暴露 ``render_camera`` /
   ``check_success`` / ``get_action_dim`` 等, 在子类 ``_register_rpc``
   中追加:

   .. code-block:: python

      def _register_rpc(self):
          super()._register_rpc()
          self._rpc["env.render_camera"] = self.render_camera
          self._rpc["env.check_success"] = self.check_success

      def render_camera(self, cam, h, w, depth=False):
          ...

      def check_success(self):
          return self._env._check_success()

3. EGL 单线程（仅 GPU 渲染后端）. 如果用 EGL 渲染, 重写 ``serve``,
   把 ``_dispatch`` 派发到专用渲染线程, 保证 MuJoCo EGL context 留在
   同一线程. 核心是用 ``work_queue`` + ``render_loop``:

   .. code-block:: python

      class MyEnvFacade(EnvFacade):
          def serve(self, *, transport, host, port, parent_watch=False):
              work_queue = queue.Queue()

              def render_loop():
                  while (item := work_queue.get()) is not None:
                      event, req = item
                      try:
                          req["result"] = self._dispatch(
                              req["method"], req["args"], req["kwargs"],
                              session_id=req["session_id"],
                          )
                      except Exception:
                          req["error"] = traceback.format_exc()
                      event.set()

              threading.Thread(target=render_loop, name="egl-render",
                                daemon=True).start()

              def dispatch(method, args, kwargs, *, session_id=None):
                  if method == "healthz":
                      return {"status": "ok"}
                  if method == "shutdown":
                      self._shutdown_event.set()
                      return {"ok": True}
                  event = threading.Event()
                  req = {"method": method, "args": args, "kwargs": kwargs,
                         "session_id": session_id,
                         "result": None, "error": None}
                  work_queue.put((event, req))
                  event.wait()
                  if req["error"]:
                      raise RuntimeError(req["error"])
                  return req["result"]

              server_cls = HttpRpcServer if transport == "http" else SocketRpcServer
              server = server_cls((host, port), dispatch)
              # bind, parent_watch, serve_forever 的完整流程参考
              # robots/robocasa/env_server.py 的 serve 重写

4. 继承 :class:`~rpent.tools.env_client_base.EnvClient`（可选）.
   如果客户端需要额外属性或方法（如 ``eef_pos`` / ``render_camera``）,
   继承后追加, 直接读 ``self.last_obs`` 避免额外 RPC:

   .. code-block:: python

      from rpent.tools.env_client_base import EnvClient

      class MyEnvClient(EnvClient):
          @property
          def eef_pos(self):
              return self.last_obs["robot0_eef_pos"]

          def render_camera(self, cam, h=256, w=256):
              return self._client.call(
                  "env.render_camera",
                  args=(cam, h, w),
                  timeout_s=self._TIMEOUT_S["default"],
              )

5. 启动 server. 在 ``main()`` 中构造 facade 并调 ``serve``:

   .. code-block:: python

      def main():
          parser = argparse.ArgumentParser()
          parser.add_argument("--transport", choices=["socket", "http"])
          parser.add_argument("--host", default="127.0.0.1")
          parser.add_argument("--port", type=int, required=True)
          parser.add_argument("--cuda-device", default="cuda")
          parser.add_argument("--parent-watch", action="store_true")
          args = parser.parse_args()

          facade = MyEnvFacade(env_cfg=...)
          facade.serve(
              transport=args.transport,
              host=args.host,
              port=args.port,
              parent_watch=args.parent_watch,
          )

关键约束
--------

- server 侧**不**缓存 ``_last_obs`` / ``_terminated``——所有缓存只在
  client 侧（``EnvClient.last_obs``）
- ``chunk_step`` 返回 5 元组
  ``(obs_or_list, reward, term, trunc, info)``; 额外信息（如
  ``n_applied``）塞进 ``info``
- ``_register_rpc`` 重写时必须先调 ``super()._register_rpc()``, 再追加
  自己的 handler
- 需要 EGL 单线程的后端必须重写 ``serve``, 把 ``_dispatch`` 派发到
  专用渲染线程; 不需要的直接用基类继承的 ``serve``
- ``expected_meta`` 握手是 client 侧职责: ``EnvClient.__init__`` 启动后
  立即调 ``env.get_env_meta`` 校验 server 配置, 不匹配直接 assert
- ``last_obs`` 是 client 侧缓存, ``reset`` / ``step`` / ``chunk_step``
  都会更新它; 调用方读 ``client.last_obs`` 不触发额外 RPC
- 子类需要 ``render_camera`` / ``check_success`` / ``get_action_dim`` 等
  额外 RPC, 在 ``_register_rpc`` 中注册, 不上提到基类
