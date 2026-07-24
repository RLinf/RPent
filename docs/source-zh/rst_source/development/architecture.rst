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
     (robocasa/)     # RoboCasa driver —— 研发中。
     (franka/)       # Franka driver —— 研发中。
     (so101/)        # SO-101 driver —— 研发中。
   scripts/          # 安装脚本 (LIBERO PRO/PLUS、codex proxy)。

Runner (``rpent/cli/main.py``)
------------------------------

``rpent/cli/main.py`` 是编排者。每一次调用它会:

1. 用 ``parse_known_args`` 解析共享 CLI flag (:doc:`../quickstart` 说明了
   日常最常用的那些), 提前拿到 ``--env`` 和 ``--dashboard``。
2. 通过 ``get_env_spec(args.env_name)`` 找到 env, 调用
   ``env_spec.add_cli_args(parser, use_dashboard=args.dashboard)`` —— env
   把自己的 flag 加到共享 parser 上。``use_dashboard=True`` 时把原本必填
   的 flag 变可选, 好让 dashboard 之后填。
3. 再次调 ``parser.parse_args()`` —— 单次 argparse pass 负责全部校验并
   产出最终的 ``args`` (argparse 自带 usage + error 输出)。
4. 如果开了 ``--dashboard``, 用现在已填入 env CLI 值的 ``args`` 起
   launcher, 把用户表单的选择 apply 回去。
5. 调用 ``env_spec.parse_config(args)`` 派生
   :class:`~rpent.envs.RunConfig`
   (``recipe_tag`` / ``output_dir`` / ``prompt_vars`` / ``dashboard_state`` /
   ``task_desc``)。dashboard 场景下, env 在这里强制之前变为可选的字段
   现在必须已经填好了。
6. 调用 ``init_output_dir`` 创建 per-run scratch 目录并挂 ``run.log``。
7. 调用 ``env_spec.init_runtime(args, output_dir)`` —— env 自己 spawn
   ``env_server`` + ``vla_server`` (或通过 ``--env-endpoint`` /
   ``--vla-endpoint`` 连到已在跑的实例), 返回
   ``(daemons, primitives_kwargs)``。
8. 通过 env 的 ``get_toolkit(primitives_kwargs=...)`` 工厂构造 **toolkit**。
9. 通过 ``rpent.planner.base.build_planner`` 构造 **planner**,
   根据 ``--planner`` 选出 ``api_loop.py`` / ``claude_code.py`` /
   ``codex.py`` 之一。
10. 跑 tool-calling 循环; 如果开了 ``--dashboard`` 就 stream 到 dashboard;
    结束时写出 ``<output_dir>/transcript_*.json`` 和
    ``<output_dir>/episode.mp4``。

Runner 有意保持薄: 一切与 env 相关的东西在 ``robots/<env>/`` 下,
一切与 brain 相关的东西在 ``rpent/planner/`` 下。main.py 不 import
任何 env-specific 的类或脚本。

Env 侧的注册表
--------------

``rpent/envs/base.py`` 维护一个以 env 名为 key 的 **lazy** 注册表。
传入 ``--env myenv`` 时, 它会执行
``importlib.import_module("robots.myenv")``, 然后调用包暴露的两个工厂:

.. code-block:: python

   # robots/myenv/__init__.py
   def get_env_spec() -> EnvSpec: ...           # 标识 + prompts + runner 钩子
   def get_toolkit(*, primitives_kwargs, video_path=None): ...

``EnvSpec`` 有五个字段:

- ``name`` / ``prompts`` —— env 标识与 :class:`PromptBundle`。
- ``add_cli_args(parser, use_dashboard) -> None`` —— 把 env 的 flag 注册
  到共享 argparse parser。``use_dashboard`` 控制原本必填的 flag 是否保持
  可选 (dashboard 场景由表单填)。
- ``parse_config(args) -> RunConfig`` —— 校验最终 ``args``
  (dashboard 之后), 返回派生的 per-run 标识。
- ``init_runtime(args, output_dir) -> (daemons, primitives_kwargs)`` ——
  spawn env / VLA 子进程, 返回 toolkit 输入。

env 是 **没有中央列表** 的。把包放到 ``robots/`` 下就行。这也是新增
机器人时用的机制 (见 :doc:`add_robot`)。

Planner 接口
------------

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

内置支持两种编码, 通过 server 端 ``--transport {http,socket}``
(默认 ``http``) 选择, client 端由 ``--env-endpoint`` /
``--vla-endpoint`` 里的 protocol 前缀对应:

- **HTTP** (``rpent.utils.http_rpc``) —— JSON body 走
  ``POST /call``, 方便做标准负载均衡, 也方便跨语言 client。
  Numpy 数组在 wire 上带标签 ``{"__ndarray__": <base64>, "dtype": ..., "shape": [...]}``。
- **Pickle-framed socket RPC** (``rpent.utils.socket_rpc``) ——
  适合历史堆叠的嵌套 numpy dict 和宽泛、形状多变的载荷 (JSON 重编码
  在这种情况下太浪费)。

Server 端继承 :class:`rpent.utils.rpc.RpcFacade` 并实现
``_dispatch(method, args, kwargs)`` 即可; base 负责 shutdown、healthz、
transport 绑定、感知父进程死亡、以及干净收尾。新增一个传输只需要实现
两个方法的 ``RpcClient`` 接口 (``call(method, args, kwargs, timeout_s)``);
toolkit 和 planner 不用动。

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
