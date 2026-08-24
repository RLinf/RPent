添加 VLA 后端
==============

在 RPent 中，*VLA 后端*（Vision-Language-Action 模型）把训练好的策略
（Pi0.5 / RLDX-1 / LingBot-VLA 等）包装成 RPC 服务，对上层 primitives 层
（``run_vla`` / ``pi0_pick`` / ``rldx_skill`` / ``lingbot_act``）暴露
统一的 ``predict`` 接口。本页讲如何基于统一基类
（:mod:`rpent.tools.vla_facade_base` /
:mod:`rpent.tools.vla_client_base`）添加新的 VLA 后端。

.. contents::
   :local:
   :depth: 2

重要规则
--------

**1. 模型在子类 ``__init__`` 中加载.**

加载策略是子类的事——加载方式、模型属性名都由子类决定. 子类在
``__init__`` 中自行加载模型, 之后 ``predict`` 直接读它. 不要在
``predict`` 里懒加载, 避免首次调用延迟和并发竞争.

**2. RPC 路由用注册字典.**

``_dispatch`` 用注册字典（``self._rpc``）替代 ``if method == "predict"``
链. 子类在 ``_register_rpc`` 中注册自己的方法, 所有 handler 必须接受
``session_id`` kwarg（VLA 不需要时忽略）. 子类重写 ``_register_rpc``
时应先调 ``super()._register_rpc()`` 再追加自己的 handler, 避免漏注册
基类已注册的 ``predict``.

**3. predict 的 obs / options / 返回值结构后端特有.**

不同模型的 obs 输入和 action 输出结构天然不同（取决于模型原生接口）.
基类只暴露抽象 ``predict``, 签名固定为
``predict(self, obs, options, *, session_id=None)``; obs 和返回值的
具体结构由后端决定, 调用方按后端特有结构处理.

**4. session 隔离: VLA 分为有 / 无 session 两类.**

VLA 后端按是否有 per-client 策略状态分两类:

- **无 session**: 后端无 per-client 策略 memory, ``predict`` 忽略
  ``session_id``. 大多数 VLA 后端属于这类.
- **有 session**: 后端有 per-client 策略 memory（如 RLDX 的 RTC）,
  需要 ``session_id`` 隔离不同 client 的策略状态.

两类后端的 ``session_id`` 都由 RPC facade 从连接派生, 客户端**不**传——
facade 的 ``_dispatch`` 收到 RPC 调用时从连接派生 ``session_id``, 作为
kwarg 传给 ``predict`` handler. 有 session 的后端在 ``predict`` 里用
``session_id`` 隔离策略状态; 无 session 的后端忽略它.

有 session 的后端需要单独处理:

- 构造 facade 时传 ``enable_sessions=True`` 和 ``session_timeout_s``
  给基类, 启用 per-client session 管理
- 在 ``predict`` 里**强制覆写** ``options["session_ids"]`` 为
  ``[session_id]``, 拒绝客户端传入的值, 防止多客户端并发串台
- ``serve`` 传 ``session_sweep_s`` 定期清理过期 session

基类
----

统一基类已落地到 ``rpent/tools/`` 下两个文件:

- :mod:`rpent.tools.vla_facade_base` —
  :class:`~rpent.tools.vla_facade_base.BaseVLAFacade`（server 侧）
- :mod:`rpent.tools.vla_client_base` —
  :class:`~rpent.tools.vla_client_base.BaseVLAClient`（client 侧）

BaseVLAFacade（server 侧）
~~~~~~~~~~~~~~~~~~~~~~~~~~

继承 :class:`~rpent.utils.rpc.RpcFacade`, 提供框架层:

- ``__init__(*, device="cuda", enable_sessions=False, session_timeout_s=None)``:
  初始化 ``self._rpc = {}`` 并调 ``_register_rpc()``; ``enable_sessions``
  / ``session_timeout_s`` 透传给 :class:`~rpent.utils.rpc.RpcFacade`,
  需要 per-client session 隔离的后端（如 robocasa）传 ``enable_sessions=True``
- 抽象方法 ``predict(self, obs, options, *, session_id=None)`` —
  子类必须实现实际推理
- ``_register_rpc()``: 默认注册 ``vla.predict`` -> ``self.predict``,
  子类可重写追加（重写时先调 ``super()._register_rpc()``）
- ``_dispatch(self, method, args, kwargs, *, session_id=None)``:
  从 ``self._rpc`` 取 handler, 用 ``self._lock`` 串行化, 把
  ``session_id`` 作为 kwarg 传给 handler
- 子类自行在 ``__init__`` 中加载模型

BaseVLAClient（client 侧）
~~~~~~~~~~~~~~~~~~~~~~~~~

薄包装 :class:`~rpent.utils.rpc.RpcClient`:

- ``__init__(self, client)``: 接受一个 ``RpcClient`` 实例
- ``_TIMEOUT_S = {"default": 30.0, "predict": 120.0}`` —
  按方法分配超时, ``predict`` 默认 120s（VLA 推理较慢）
- ``predict(self, obs, options=None)``: 调
  ``self._client.call("vla.predict", args=(obs, options),
  timeout_s=self._TIMEOUT_S["predict"])``
- 客户端**不**传 ``session_id``（由 facade 注入）

接入步骤
--------

1. 继承 :class:`~rpent.tools.vla_facade_base.BaseVLAFacade`, 在
   ``__init__`` 中加载模型, 实现 ``predict``:

   .. code-block:: python

      from rpent.tools.vla_facade_base import BaseVLAFacade

      class MyVLAFacade(BaseVLAFacade):
          def __init__(self, *, model_path: str, device: str = "cuda"):
              super().__init__(device=device)
              self.policy = load_my_policy(model_path, device=device)

          def predict(self, obs, options, *, session_id=None):
              return self.policy.run(obs)

2. 注册额外 RPC 方法（可选）. 如果后端需要暴露 ``reset_session`` /
   ``get_model_meta`` 等, 在子类 ``_register_rpc`` 中追加:

   .. code-block:: python

      def _register_rpc(self):
          super()._register_rpc()
          self._rpc["vla.reset_session"] = self.reset_session
          self._rpc["vla.get_model_meta"] = self.get_model_meta

      def reset_session(self, *, session_id=None):
          ...

      def get_model_meta(self, *, session_id=None):
          return {"action_dim": ..., "horizon": ...}

   所有 handler 必须接受 ``session_id`` kwarg.

3. 继承 :class:`~rpent.tools.vla_client_base.BaseVLAClient`（可选）.
   如果客户端需要额外方法（如 ``reset_session``）, 继承后追加:

   .. code-block:: python

      from rpent.tools.vla_client_base import BaseVLAClient

      class MyVLAClient(BaseVLAClient):
          def reset_session(self):
              return self._client.call(
                  "vla.reset_session",
                  timeout_s=self._TIMEOUT_S["default"],
              )

4. 启动 server. 在 ``main()`` 中构造 facade 并调 ``serve``:

   .. code-block:: python

      def main():
          parser = argparse.ArgumentParser()
          parser.add_argument("--transport", choices=["socket", "http"])
          parser.add_argument("--host", default="127.0.0.1")
          parser.add_argument("--port", type=int, required=True)
          parser.add_argument("--cuda-device", default="cuda")
          parser.add_argument("--model-path", required=True)
          parser.add_argument("--parent-watch", action="store_true")
          args = parser.parse_args()

          facade = MyVLAFacade(
              model_path=args.model_path, device=args.cuda_device
          )
          facade.serve(
              transport=args.transport,
              host=args.host,
              port=args.port,
              parent_watch=args.parent_watch,
          )

   需要 per-client session 隔离时, 构造 facade 传
   ``enable_sessions=True`` 和 ``session_timeout_s``, ``serve`` 传
   ``session_sweep_s``（见 :class:`~rpent.utils.rpc.RpcFacade.serve`）.

关键约束
--------

- ``predict`` 签名固定: ``predict(self, obs, options, *, session_id=None)``,
  子类不能改签名
- ``session_id`` 由 facade 注入, ``predict`` 实现里不要从 ``options``
  读 ``session_ids``——如果需要 session 隔离, 直接用入参的
  ``session_id``
- 模型在子类 ``__init__`` 中加载, 不要在
  ``predict`` 里懒加载
- ``_register_rpc`` 重写时必须先调 ``super()._register_rpc()``, 再追加
  自己的 handler
- ``_dispatch`` 所有 handler 必须接受 ``session_id`` kwarg（不需要时
  忽略）
- 客户端 ``predict`` 不传 ``session_id``——由 facade 从连接派生后
  注入到 server 端 handler
- ``_TIMEOUT_S["predict"]`` 默认 120s, 如果模型推理更慢, 子类可重写
  ``_TIMEOUT_S`` 调大
