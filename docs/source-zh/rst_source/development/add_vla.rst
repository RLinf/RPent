添加 VLA 后端
==============

在 RPent 中，*VLA 后端*（Vision-Language-Action 模型）把训练好的策略
（Pi0.5 / RLDX-1 / LingBot-VLA 等）包装成 RPC 服务，对上层
（primitives 层的 ``run_vla`` / ``pi0_pick`` / ``rldx_skill`` /
``lingbot_act``）暴露统一的 ``predict`` 接口。本页介绍如何基于统一基类
（:mod:`rpent.tools.vla_facade_base` / :mod:`rpent.tools.vla_client_base`）
添加新的 VLA 后端，并对比现有三个后端（libero / robocasa / robotwin）
的差异与共有结构。

.. contents::
   :local:
   :depth: 2

总体格局
--------

.. list-table::
   :header-rows: 1
   :widths: 18 27 27 28

   * - 基类
     - ``RpcFacade``
     - **无** — 直接拉官方 WS server
   * - 传输
     - RPent socket/http RPC
     - RPent socket/http RPC
     - 官方 websocket
   * - session
     - 无
     - per-client（强隔离）
     - 共享单 session
   * - predict 入参
     - ``(instruction, images, state, mode)``
     - ``(obs_dict, options, *, session_id)``
     - ``infer(observation)``
   * - obs 组装
     - 服务器侧解码 base64-PNG
     - 客户端预先组装
     - 客户端转 LingBot payload
   * - 返回值
     - ``{actions, shape, dtype}``
     - RLDX 原生 dict[str, ndarray]
     - ``[chunk, 16]`` ndarray
   * - 元信息 RPC
     - 无
     - ``get_video_delta_indices``
     - 无
   * - ``reset_session``
     - 无
     - 有
     - 有（重置共享单 session）

**关键发现**：libero/robocasa 走同一套 ``RpcFacade`` 模板；robotwin 完全
绕开 RPC 框架，直接拉官方 ``WebsocketPolicyServer``。统一层应抽
``BaseVLAFacade(RpcFacade)``，robotwin 在基类内部 delegate 到 WS server，
对外仍暴露统一的 ``vla.predict`` / ``vla.reset_session``。

设计原则
--------

**1. session_id 由 facade 注入, 客户端不传.**

``session_id`` 由 RPC facade 从连接派生, 客户端**不**主动传（客户端只传
业务参数 obs / options）. facade 的 ``_dispatch`` 收到 RPC 调用时从连接
派生 ``session_id``, 传给 predict handler. 这样:

- robocasa 的 per-client session 强隔离对客户端透明（客户端不需要知道
  自己的 session_id）
- libero / robotwin 无 session, 客户端也不需要特殊处理

robocasa 的 ``predict`` 拒绝客户端传入的 ``options["session_ids"]``,
强制覆写为 ``[session_id]``, 保证 RLDX memory / RTC 隔离（多客户端并发
时不串台）.

**2. 模型加载在子类 ``__init__``, 基类不调 ``_load_model``.**

子类在 ``__init__`` 中自行加载模型并赋值给 ``self._model``. 基类**不在**
``__init__`` 里调 ``_load_model`` — 那是子类的事, 名字也由子类决定
（libero / robocasa 在 ``__init__`` 里直接 load, robotwin 在 ``main()``
里构造 policy 后传入）.

**3. RPC 路由用注册字典.**

``_dispatch`` 用注册字典（``self._rpc``）替代 ``if method == "predict"``
链. 子类在 ``_register_rpc`` 中注册自己的方法. 所有 handler 接受
``session_id`` kwarg（VLA 不需要时忽略）.

**4. predict 的 obs / options / 返回值结构后端特有.**

当前三者 predict 的入参和返回值结构都不同（见下文"后端特有部分"）.
基类只暴露抽象 ``predict``, 用 ``**kwargs`` / 后端特有 dict 容纳差异.
统一设计建议在基类加 ``_normalize_return`` normalize 成统一结构, 当前
未实装.

共有 RPC 接口
-------------

框架层（libero + robocasa 共有）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- 继承 ``RpcFacade``, 构造时 ``super().__init__(enable_sessions=...)``
- 模型在 ``__init__`` 一次性 load
- ``_dispatch(self, method, args, kwargs, *, session_id)`` 用 ``self._lock``
  串行化, 按方法名路由到 ``self.<attr>``
- CLI 统一: ``--transport {socket,http}`` / ``--host`` / ``--port`` /
  ``--cuda-device`` / ``--parent-watch`` / ``--model-path``; 启动调用
  ``facade.serve(...)``
- 客户端薄包装 ``RpcClient``, 内置 ``_TIMEOUT_S``

.. note::

   ``sam3_server.py`` 也走同一套路（``Sam3Facade(RpcFacade)`` +
   ``_dispatch`` + ``serve``）, 证明这是仓库内通用的"本地模型 RPC server"
   模板, 不只 VLA 专属.

业务层（libero + robocasa 共有）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - 方法
     - 签名
     - 返回值
   * - ``predict``
     - ``predict(obs, options, *, session_id?)``
     - libero: ``{actions, shape, dtype}``（已 ``.tolist()``）;
       robocasa: RLDX 原生 dict[str, ndarray]
   * - ``reset_session``（仅 robocasa）
     - ``reset_session(*, session_id)``
     - ``{"ok": True}``
   * - ``get_video_delta_indices``（仅 robocasa）
     - ``get_video_delta_indices()``
     - ndarray / JSON list

客户端层共有
~~~~~~~~~~~~

- 薄包装 ``RpcClient``
- ``_TIMEOUT_S = {"default": 30.0, "predict": 120.0}``
- 客户端不主动传 ``session_ids``（由 facade 注入）

功能表格
--------

**后端覆盖列**: L=libero (Pi0.5), R=robocasa (RLDX-1), T=robotwin
(LingBot). "L+R"表示二者共有但不包括 robotwin; "R only"表示仅 robocasa.
**当前真正三者共有的 VLA 接口只有** ``predict`` **一个**（且签名/入参/
返回值结构三者都不同）. 其余都是 R only 或后端特有, 统一设计建议把它们
作为基类抽象方法或可选 hook.

.. list-table::
   :header-rows: 1
   :widths: 14 28 12 46

   * - 大类功能
     - 函数列表
     - 后端覆盖
     - 功能实现及细节
   * - **推理**
     - ``predict`` / ``infer``
     - L+R+T（签名不同）
     - L: ``predict(instruction, images, state, mode)`` 服务器侧解码
       base64-PNG 组装 ``env_obs``, 返回 ``{actions, shape, dtype}``;
       R: ``predict(obs_dict, options, *, session_id)`` 客户端预组装 obs,
       返回 RLDX 原生 dict; T: ``infer(observation)`` 客户端转 LingBot
       payload 走 WS, 返回 ``[chunk, 16]`` ndarray
   * - **会话管理**
     - ``reset_session`` / ``_on_session_drop`` / ``enable_sessions`` /
       ``session_timeout_s`` / ``session_sweep_s``
     - R only
     - R ``enable_sessions=True`` + per-client session 强隔离;
       ``session_id`` 由 facade 注入, ``predict`` 拒绝客户端覆盖;
       ``_on_session_drop`` 兜底 reset. L 完全无 session. T 有 ``reset()``
       但重置的是**共享单 session**, 语义不同
   * - **元信息**
     - ``get_video_delta_indices``
     - R only
     - R 暴露 RLDX video modality 的 delta_indices（如 ``[-6,-4,-2,0]``）,
       供 env_client 对齐历史帧堆栈; L/T 无元信息 RPC（horizon /
       action_dim 硬编码）
   * - **生命周期**
     - ``load`` / ``close`` / ``serve`` / ``_dispatch`` / ``_to_numpy_tree``
     - L+R（T 走 WS）
     - L/R 模型在 ``__init__`` 一次性 load; ``serve`` 启动 RPC server;
       ``_dispatch`` 路由方法名. T 绕开 ``RpcFacade``, ``main()`` 直接
       ``WebsocketPolicyServer(policy, port).serve_forever()``, 无
       ``_dispatch``
   * - **图像处理**
     - ``_decode_image_block``
     - L only
     - L 服务器侧把 base64-PNG 解码成 ndarray 并组装成 ``env_obs``
       （``main_images`` / ``wrist_images`` / ``extra_view_images``）;
       R 不需要（客户端预组装）; T 在客户端完成 payload 转换
   * - **obs schema 适配**
     - ``infer`` 内的 payload 转换
     - T only
     - T 把 RoboTwin 原生 obs（``views.*.rgb`` / ``robot_state.*`` /
       ``task_language``）转成 LingBot payload（``observation.images.cam_*``
       / ``observation.state`` / ``task``）; L/R 无对应物
   * - **parent_watch**
     - ``watch_parent_death``
     - L+R+T（实现不同）
     - L/R 通过 facade 的 ``_shutdown_event``; T 用
       ``rpent.utils.daemon.watch_parent_death`` 直接 ``os._exit``

基类代码
--------

统一基类已落地到 ``rpent/tools/`` 下两个文件, 子类（libero / robocasa /
robotwin 的 VLA facade 和 VLA client）应继承并实现抽象方法:

- :mod:`rpent.tools.vla_facade_base` —
  :class:`~rpent.tools.vla_facade_base.BaseVLAFacade`（server 侧）
- :mod:`rpent.tools.vla_client_base` —
  :class:`~rpent.tools.vla_client_base.BaseVLAClient`（client 侧）

.. note::

   基类只固化三者共有的接口（``predict``）+ 框架层（``_dispatch`` 注册
   字典 / ``_to_numpy_tree`` / ``serve``）. R only / L only / T only 的
   方法**不放基类**, 由子类自行定义并通过 ``_register_rpc`` 注册.

统一设计建议
------------

1. **抽** :class:`~rpent.tools.vla_facade_base.BaseVLAFacade` ``(RpcFacade)``
   **基类**: 固化 ``_dispatch``（**注册字典** ``self._rpc`` +
   ``self._lock``）+ ``session_id`` 注入策略 + ``serve(...)`` 入口;
   libero 和 robocasa 直接继承; robotwin 在 ``BaseVLAFacade`` 内部
   delegate 到官方 WS server（把 ``_dispatch`` 实现成 WS 转发）, 保留
   对外统一的 ``vla.predict`` / ``vla.reset_session`` 接口
2. **统一** ``predict`` **obs schema**: 标准化为
   ``predict(obs_dict, options, *, session_id?)``, 其中 ``obs_dict`` 至少
   包含 ``{instruction, images{main, wrist, extra}, state}``; 把图像解码
   从 libero server 上提到客户端或共享 helper（参考
   ``_decode_image_block``）, 让 server 只收 ndarray / 已解码张量;
   ``options`` 标准化为 ``{"reset_memory": bool, "session_ids": [...]}``
   （后者仍由 server 注入, client 不传）
3. **统一 session 隔离模型**: 所有后端 ``enable_sessions=True``,
   ``session_id`` 一律由 RPC facade 从连接派生, predict / reset 强制注入,
   禁止客户端覆盖; robotwin 的"共享单 session"需要额外包一层: 在
   ``BaseVLAFacade`` 里维护 ``session_id → LingBotClientPolicy`` 映射,
   或退化为"单 session 但 ``reset_session`` 显式重置"
4. **元信息 RPC 标准化**: 把 ``get_video_delta_indices`` 推广为
   ``get_model_meta()`` 返回 ``{action_dim, horizon, n_action_steps,
   video_delta_indices, proprio_dim, image_keys}``, libero / robotwin 也
   实现（即便返回空 video_delta_indices）, 让 env_client 能统一查询
5. **统一返回值结构**: ``predict`` 一律返回
   ``{actions: ndarray, shape, dtype, horizon, n_action_steps}``, 客户端
   按 ``n_action_steps`` 取前 N 步; 当前 libero 已包成
   ``{actions, shape, dtype}`` / robocasa 返回 RLDX 原生 dict / robotwin
   返回裸 ndarray — 三方都需要在基类里 normalize（``_normalize_return``）
6. **``reset_memory`` 语义统一**: ``options["reset_memory"]`` 作为标准
   reset 信号, libero / robotwin 也实现（即便映射到 ``reset_session``
   或 no-op）, 让 primitives 层的 ``force_reset`` 逻辑能跨后端复用
