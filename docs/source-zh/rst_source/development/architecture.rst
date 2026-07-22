系统设计
========

本页从实现层面看 RPent —— 三个进程各自持有什么、如何通信,
以及 ``rpent/`` 与 ``robots/`` 下的代码如何组织。更高层的框架介绍
见 :doc:`../overview`。

.. raw:: html

   <div style="text-align: center;">
     <img src="../../architecture.svg" alt="RPent 三进程架构"
          style="max-width: 95%; height: auto;" />
   </div>

关键特性
--------

*(这些是架构设计围绕的框架级承诺; 下面各节则展示每一项是如何落地的。)*

- **LLM-in-the-loop 控制。** LLM 不做微调 —— 它纯粹通过调工具
  (``pi0_pick``、``move_to``、``rotate_wrist``、``back_project``、
  ``finish``…) 来驱动机器人。每个工具的返回都以多模态上下文
  (文本 + 渲染图) 喂回, 让模型基于 *看到的世界* 推理。
- **三进程架构。** **Agent 进程** (LLM planner + toolkit, 不 import
  ``torch``)、**env_server** (仿真器 + EGL 渲染)、**vla_server**
  (GPU 策略权重) 是三个独立进程, 用轻量 RPC 串起来。任一重量级
  进程都可以独立重启、迁到另一张 GPU、或指向远程主机。
- **可插拔的 reasoning brain (planner)。** 用一个 flag ——
  ``--planner {api, claude_code, codex}`` —— 就能换决策 brain, 不用
  动 tool 或 prompt:

  - ``api`` —— 基于 `pydantic-ai <https://ai.pydantic.dev/>`_ 的
    provider-无关 tool-calling 循环 (Anthropic / OpenAI / OpenAI 兼容),
    带 prompt 缓存和历史图片剪枝。
  - ``claude_code`` —— `Claude Agent SDK
    <https://docs.claude.com/en/api/agent-sdk/overview>`_,
    把 toolkit 暴露为 in-process MCP server。
  - ``codex`` —— OpenAI Codex SDK, 通过 HTTP MCP server 桥接到
    toolkit。
- **两个 environment、两个 VLA、一份契约。** LIBERO (Pi0.5 走 HTTP) 和
  RoboCasa (RLDX-1 走 socket-RPC) 共享 *完全一致* 的 env/vla 进程划分;
  只有传输协议不同, 且是按各自 observation 形状选出来的。
- **实时 dashboard。** 可选的 ``--dashboard`` 会起一个本地 FastAPI
  监控页, 实时展示 agent 的 reasoning、相机 / Pi0 视图、动作时间线、
  剪辑回放 —— 提供 **双语 UI** (``--dashboard-language {en, zh-cn}``)。
- **加一个 environment 只需把包放进硬盘。** 没有中央注册表要改 ——
  见 :doc:`add_robot`。

单轮循环是怎么发生的
--------------------

一次运行就是一段 LLM-in-the-loop 循环:

1. LLM 分析任务、调一个工具 (如 ``pi0_pick``)。
2. 工具的 **primitive driver** 向 ``vla_server`` 请求一个 action
   chunk (``predict`` / ``vla_infer``)。
3. ``env_server`` 执行这段 chunk (LIBERO 是 ``chunk_step``, RoboCasa
   是逐步 ``step``)。
4. Env 渲染出新的 observation 与相机帧。
5. 结果被组装成 text + image content block, 喂回 LLM 进入下一轮。

循环在 LLM 调 ``finish`` (``success`` / ``failure`` / ``stuck``)
或触达 ``--max-turns`` / ``--max-episode-steps`` 时结束。

仓库布局
--------

实现按关注点拆分得比较干净:

.. code-block:: text

   rpent/
     planner/       # Reasoning brains: api_loop, claude_code, codex, base.
     cli/            # main.py 入口 (无 __init__.py, 不是 subpackage)。
     context/        # Prompt bundles、prompt 工具、共享 prompt 分节。
     dashboard/      # FastAPI 监控 + SSE stream (可选)。
     envs/           # EnvSpec、PromptBundle、以及 env 的 lazy 注册表。
     tools/          # Toolkit 基类和共享 tool 辅助函数。
     utils/          # 配置、日志、RPC client/server、VLA HTTP shim。
   robots/
     libero/         # LIBERO 的 env_client / env_server / vla_server /
                     # toolkit / prompt_bundle。参考实现。
     (robocasa/)     # RoboCasa driver (见 scripts/run_robocasa.sh)。
     (franka/)       # Franka driver —— 研发中。
     (so101/)        # SO-101 driver —— 研发中。
   scripts/          # 安装脚本 (LIBERO PRO/PLUS、RoboCasa、codex proxy)。

Runner (``rpent/cli/main.py``)
------------------------------

``rpent/cli/main.py`` 是编排者。每一次调用它会:

1. 解析 CLI flag (:doc:`../quickstart` 说明了日常最常用的那些)。
2. 创建 per-run 的 scratch 目录 (``--output-dir`` 或
   ``runs/`` 下自动生成的目录)。
3. 以子进程启动 **env_server**, 等待它在 stdout 上打印
   ``transport_ready`` JSON 事件。事件里带着 socket RPC 监听的
   host/port; ``rpent/cli/main.py`` 把 endpoint 记录到 ``<output_dir>/``
   下, 供 client 找到。
4. 以同样方式启动 (或复用) **vla_server**, 复用时用
   ``--vla-endpoint``。
5. 通过 env 的 ``get_toolkit(primitives_kwargs=...)`` 工厂为选中的
   env 构造 **toolkit**, 把 env client 和 VLA client 传进去。
6. 通过 ``rpent.planner.base.build_planner`` 构造 **planner**,
   根据 ``--planner`` 选出 ``api_loop.py`` / ``claude_code.py`` /
   ``codex.py`` 之一。
7. 跑 tool-calling 循环; 如果开了 ``--dashboard`` 就 stream 到 dashboard;
   结束时写出 ``<output_dir>/transcript_*.json`` 和
   ``<output_dir>/episode.mp4``。

Runner 有意保持薄: 一切与 env 相关的东西在 ``robots/<env>/`` 下,
一切与 brain 相关的东西在 ``rpent/planner/`` 下。

Env 侧的注册表
--------------

``rpent/envs/base.py`` 维护一个以 env 名为 key 的 **lazy** 注册表。
传入 ``--env myenv`` 时, 它会执行
``importlib.import_module("robots.myenv")``, 然后调用包暴露的两个工厂:

.. code-block:: python

   # robots/myenv/__init__.py
   def get_env_spec() -> EnvSpec: ...
   def get_toolkit(*, primitives_kwargs, video_path=None): ...

env 是 **没有中央列表** 的。把包放到 ``robots/`` 下就行。这也是新增
机器人时用的机制 (见 :doc:`add_robot`)。

Planner 接口
-------------

每个 planner 实现同一个很小的接口 (见 ``rpent.planner.base``):

- 接受渲染好的 ``prompt_bundle`` (system + user 分节)。
- 接受一个 ``toolkit`` (暴露 tool schema 和 ``dispatch`` 方法)。
- 驱动 tool-calling 循环。
- 把每个 tool 返回值以多模态上下文喂回。
- 遇到 ``finish`` 或触达上限时终止。

抽象就这些。三个内置 planner 只在 *如何满足契约* 上不同 —— 用户视角
见 :doc:`../usage/configure_planner`, 源码见
``rpent/planner/api_loop.py`` / ``claude_code.py`` / ``codex.py``。

Toolkit 接口
------------

一个 toolkit (``rpent.tools.toolkit.Toolkit``) 持有:

- 一个 **primitive driver** —— 一个普通 Python 对象, 持有 env
  client、VLA client 和任何 per-run 状态。LLM 能调的每个工具对应
  它的一个方法。
- 一组 **tool schema** (Anthropic 形状: ``name``、``description``、
  ``input_schema``), 通过 ``self.add_tool(name, spec, handler)``
  注册。
- 每步的 **状态 dump** —— 每个 primitive tool 跑完后重新渲染世界,
  这样下一次 ``view_driver_state`` 看到的就是动作后的状态。

基类还处理 video 录制 (``episode.mp4``) 与 dashboard 事件流。
新增 env 的 ``toolkit.py`` 继承此基类并注册该 env 暴露的工具。

传输层
------

内置支持两种编码:

- **Pickle-framed socket RPC** (``rpent.utils.socket_rpc``) —— 所有
  env_server 以及 RoboCasa vla_server 都走它。适合历史堆叠的嵌套
  numpy dict (RLDX obs) 和 env_server 之间宽泛、形状多变的状态载荷。
- **HTTP** —— LIBERO vla_server 用, 走 Pi0.5 的扁平
  ``image+state → action`` 载荷。方便做标准负载均衡, 也方便
  ``--vla-endpoint`` 式复用。

新增一个传输只需要实现两个方法的 ``RpcClient`` 接口
(``call(method, args, kwargs, timeout_s)``); toolkit 和 planner 不用动。

Dashboard (可选)
----------------

``rpent/dashboard/`` 是一个 FastAPI app 加一份静态前端。
开了 ``--dashboard`` 时, ``rpent/cli/main.py`` 会把它绑在
``--dashboard-host:--dashboard-port`` 上 (默认 localhost, 随机端口),
先起 launcher 页面选配置, 然后 stream:

- Agent 的 reasoning token (SSE)。
- 实时相机 / Pi0.5 叠加帧。
- 动作时间线。
- 结束时的剪辑回放。

Dashboard 是 *观察性的* —— 永远不影响循环 —— 所以 dashboard 内部
出错也不会拖垮 run。

下一步
------

- 新增机器人? —— :doc:`add_robot`。
- 新增 VLA / action primitive? —— :doc:`add_primitive`。
- 想了解 memory 的设计与接入点? —— :doc:`memory`。
- 需要完整的扩展 checklist? —— :doc:`add_robot`。
