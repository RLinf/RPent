系统设计
========

本页从实现层面看 RPent —— 核心控制链路中的三个进程各自持有什么、如何通信,
以及 ``rpent/`` 与 ``robots/`` 下的代码如何组织。更高层的框架介绍
见 :doc:`../overview`。

.. raw:: html

   <div style="text-align: center;">
     <img src="../../architecture.svg" alt="RPent 核心三进程架构"
          style="max-width: 95%; height: auto;" />
   </div>

关键特性
--------

*（下面先概括架构的主要设计目标，后续各节再说明对应的实现方式。）*

- **LLM-in-the-loop 控制。** 无需对 LLM 进行微调；模型只需调用工具
  （``pi0_pick``、``move_to``、``rotate_wrist``、``back_project``、
  ``finish``……）即可控制机器人。每个工具的执行结果都会以多模态上下文
  （文本和渲染图像）的形式返回给模型，使模型能够结合当前环境状态继续推理。
- **核心三进程架构。** **Agent 进程** (LLM planner + toolkit)、
  **env_server** (仿真器 + EGL 渲染)、**vla_server**
  (GPU 策略权重) 是三个独立进程, 构成用轻量 RPC 串联的核心控制链路。
  任一重量级服务进程都可以独立重启、迁到另一张 GPU、或指向远程主机。
- **可插拔的 planner。** 通过
  ``--planner {api, claude_code, codex}`` 参数即可切换 planner，
  无需修改工具或提示词：

  - ``api`` —— 基于 `pydantic-ai <https://ai.pydantic.dev/>`_
    实现不绑定特定模型提供商的工具调用循环，支持 Anthropic、OpenAI
    及 OpenAI 兼容接口，并支持提示词缓存和自动移除较早的历史图像。
  - ``claude_code`` —— 基于 `Claude Agent SDK
    <https://docs.claude.com/en/api/agent-sdk/overview>`_，将 toolkit
    暴露为进程内 MCP 服务。
  - ``codex`` —— 基于 OpenAI Codex SDK，通过 HTTP MCP 服务连接
    toolkit。
- **两种环境、两个 VLA、一套接口约定。** LIBERO（Pi0.5，通过 HTTP 通信）
  与 RoboCasa（RLDX-1，通过 socket-RPC 通信）采用相同的
  ``env_server`` / ``vla_server`` 进程划分；两者仅传输协议不同，具体协议
  根据各环境观测数据的结构和大小选择。
- **实时 Dashboard。** 启用 ``--dashboard`` 后，RPent 会启动本地 FastAPI
  监控页面，实时显示 agent 的推理过程、相机画面、Pi0 视图和动作时间线，
  并支持回放动作片段。Dashboard 提供中英文界面，可通过
  ``--dashboard-language {en, zh-cn}`` 指定语言。
- **通过独立的环境包进行扩展。** 环境实现位于 ``robots/<env>/``，RPent
  会按环境名称动态加载。具体接入步骤见 :doc:`add_robot`。

LLM-in-the-loop 运行流程
------------------------

一次运行就是一段 LLM-in-the-loop 循环：

1. LLM 分析任务、调一个工具 (如 ``pi0_pick``)。
2. 工具的底层驱动向 ``vla_server`` 请求动作 (``predict`` / ``vla_infer``)。
3. ``env_server`` 执行动作。
4. 环境返回更新后的观测数据和相机画面。
5. 执行结果会整理成由文本和图像组成的上下文，返回给 LLM 进行下一轮推理。

循环在 LLM 调 ``finish`` (``success`` / ``failure`` / ``stuck``)
或达到 ``--max-turns`` / ``--max-episode-steps`` 时结束。

仓库布局
--------

代码按职责组织如下：

.. code-block:: text

   rpent/
     planner/       # planner 实现：api_loop、claude_code、codex、base。
     cli/            # main.py 入口和交互式终端。
     context/        # 提示词工具和共享提示词片段。
     dashboard/      # FastAPI 监控页面和 SSE 事件流（可选）。
     envs/           # EnvSpec、PromptBundle 和按需加载环境的逻辑。
     tools/          # Toolkit 基类和共享 tool 辅助函数。
     utils/          # 配置、日志、RPC 客户端/服务端和 VLA 客户端。
   robots/
     libero/         # LIBERO 的 env_client / env_server / vla_server /
                     # toolkit / prompt_bundle。参考实现。
     (robocasa/)     # RoboCasa 驱动——研发中。
     (franka/)       # Franka 驱动——研发中。
     (so101/)        # SO-101 驱动——研发中。
   scripts/          # 安装脚本（LIBERO PRO/PLUS、Codex 代理）。

Runner (``rpent/cli/main.py``)
------------------------------

``rpent/cli/main.py`` 负责串联一次运行所需的配置、服务和模型组件。
启动后，它依次执行以下步骤：

1. 调用 ``parse_known_args`` 初步解析通用 CLI 参数
   （常用参数见 :doc:`../quickstart`），先读取 ``--env`` 和
   ``--dashboard``。
2. 根据 ``args.env_name`` 调用 ``get_env_spec`` 加载环境定义，再通过
   ``env_spec.add_cli_args(parser, use_dashboard=args.dashboard)`` 将该环境
   的专用参数加入共享 parser。启用 Dashboard 时，原本必填的环境参数会暂时
   设为可选，随后由配置页面填写。
3. 再调用 ``parser.parse_args()``，对完整参数集合执行 argparse 层的校验，
   并生成最终的 ``args``；参数错误仍使用 argparse 的标准提示格式。
4. 如果启用了 ``--dashboard``，启动配置页面，以当前参数作为默认值，并将
   用户提交的配置写回 ``args``。
5. 调用 ``env_spec.parse_config(args)`` 校验运行配置，并生成
   :class:`~rpent.envs.RunConfig`，其中包含 ``recipe_tag``、``output_dir``、
   ``prompt_vars``、``dashboard_state`` 和 ``task_desc``。启用 Dashboard
   时，此处还会确认配置页面已经补齐所需的环境参数。
6. 调用 ``init_output_dir`` 创建本次运行的输出目录，并配置 ``run.log``。
7. 根据 ``--planner`` 调用 ``rpent.planner.base.build_planner`` 构造
   **planner**，并使用环境提供的 prompt bundle 生成 system prompt 和
   user prompt。
8. 调用 ``env_spec.init_runtime(args, output_dir)``。环境实现会启动
   ``env_server`` 和 ``vla_server``；如果指定了 ``--env-endpoint`` 或
   ``--vla-endpoint``，则连接已有服务。该方法返回
   ``(daemons, primitives_kwargs)``。
9. 将 ``primitives_kwargs`` 传给环境的 ``get_toolkit`` 工厂，构造
   **toolkit**。
10. 执行工具调用循环；启用 Dashboard 时，同时将运行事件发送到监控页面。
    循环结束后保存 ``<output_dir>/transcript_*.json``，并在清理 toolkit
    时完成回合录像等收尾工作。

``main.py`` 只负责连接上述步骤。环境相关实现集中在 ``robots/<env>/``，
planner 后端集中在 ``rpent/planner/``，因此 ``main.py`` 不直接导入任何
环境专用的类或脚本。

环境加载机制
------------

``rpent/envs/base.py`` 根据环境名称按需加载对应的实现。传入的环境名称为
``myenv`` 时，它会执行 ``importlib.import_module("robots.myenv")``，
再调用该包提供的两个工厂：

.. code-block:: python

   # robots/myenv/__init__.py
   def get_env_spec() -> EnvSpec: ...  # 环境标识、提示词模板与 Runner 钩子
   def get_toolkit(
       *, primitives_kwargs, video_path=None, dashboard=None
   ): ...

``EnvSpec`` 包含五个字段：

- ``name`` 和 ``prompts``：环境名称与 :class:`PromptBundle`。
- ``add_cli_args(parser, use_dashboard) -> None``：将环境专用参数注册到
  共享的 argparse 解析器。``use_dashboard=True`` 时，配置页面负责填写的
  参数可以暂时保持可选。
- ``parse_config(args) -> RunConfig``：校验最终参数，并生成本次运行所需的
  :class:`~rpent.envs.RunConfig`。
- ``init_runtime(args, output_dir) -> (daemons, primitives_kwargs)``：
  启动环境和 VLA 服务进程，或连接已有服务，并返回构造 toolkit 所需的参数。

加载器本身不维护环境名称列表。不过，当前 CLI 仍将 ``--env`` 限定为
``libero``；接入新的环境名称时，还需要同步更新 CLI 的可选值。完整步骤见
:doc:`add_robot`。

Planner 接口
------------

所有 planner 都实现 ``rpent.planner.base.Planner`` 中定义的 ``solve`` 接口：

- 接收 Runner 已经生成的系统提示词和用户消息。
- 接收一个 ``toolkit``，通过 ``get_tools_spec()`` 获取工具定义，并通过
  ``execute_tool()`` 执行工具。
- 驱动多轮工具调用，并将执行结果返回给模型；结果包含图像内容时，以多模态
  上下文传递。
- 在调用 ``finish`` 或达到 ``max_turns`` 后结束。

三个内置 planner 使用同一接口，但接入模型和工具的方式不同。使用方法见
:doc:`../usage/configure_planner`；对应实现位于
``rpent/planner/api_loop.py``、``claude_code.py`` 和 ``codex.py``。

Toolkit 接口
------------

``rpent.tools.toolkit.Toolkit`` 是 planner 面向的工具容器：

- 基类注册文件读写等通用工具，并维护工具名称、输入参数结构与处理函数之间的
  对应关系。子类通过 ``self.add_tool(name, spec, handler)`` 添加环境专用工具。
- 环境专用 toolkit 可以持有 primitive driver。以 LIBERO 为例，
  ``LiberoPrimitives`` 持有环境 client、model client 和本次运行的状态；
  需要推进环境的原语由 ``LiberoToolkit._step`` 调用。
- 每次原语执行后，``LiberoToolkit._step`` 都会调用 ``dump_state``
  保存状态快照，包括状态、图像、深度和执行日志。后续调用
  ``view_driver_state`` 时读取的就是动作执行后的快照。

Toolkit 基类会将工具执行结果转发给 Dashboard；回合录像
（``episode.mp4``）和动作片段则由 ``LiberoToolkit`` 负责生成。新增环境时，
应继承 Toolkit 基类并注册该环境需要暴露的工具。

RPC 传输层
----------

RPent 内置 HTTP 和 pickle-framed socket 两种 RPC 传输方式。服务进程通过
``--transport {http,socket}`` 选择传输方式，默认使用 HTTP；agent 侧根据
``--env-endpoint`` 或 ``--vla-endpoint`` 中的协议前缀创建对应的客户端。

- **HTTP**\ （``rpent.utils.http_rpc``）：通过 ``POST /call`` 发送 JSON
  请求，便于使用标准负载均衡器或接入其他语言编写的客户端。NumPy 数组会编码为
  ``{"__ndarray__": <base64>, "dtype": ..., "shape": [...]}``。
- **Pickle-framed socket RPC**\ （``rpent.utils.socket_rpc``）：使用带长度前缀的
  pickle 数据帧传输请求和响应，适合包含多帧历史或嵌套 NumPy 字典的观测数据，
  可以避免重复转换为 JSON。由于 pickle 不适合不可信输入，这种方式只应连接
  可信端点。

服务端可以继承 :class:`rpent.utils.rpc.RpcFacade`，并实现业务分发方法
``_dispatch(method, args, kwargs)``。基类负责 ``shutdown``、``healthz``、
传输绑定、监测父进程退出和关闭服务。新增传输方式时，需要同时实现客户端和
服务端，并将其接入 ``RpcFacade`` 及端点解析逻辑；只要继续满足
``RpcClient`` 接口，toolkit 和 planner 无需修改。

Dashboard（可选）
-----------------

``rpent/dashboard/`` 由 FastAPI 应用和静态前端组成。启用 ``--dashboard`` 后，
``rpent/cli/main.py`` 会根据 ``--dashboard-host`` 和 ``--dashboard-port``
启动 Dashboard；默认绑定 ``127.0.0.1``，并由操作系统分配可用端口。运行开始前，
用户可以先在配置页面确认或修改参数。

运行期间，Dashboard 页面提供：

- planner 输出以及工具调用事件；
- 实时相机画面和 Pi0.5 视图；
- 动作时间线和单步动作片段；
- 运行结束后的完整回合录像（如果已生成）。

服务端通过 SSE 推送运行状态摘要，前端再按需读取详细事件、时间线和图像。
Dashboard 使用 planner 与 toolkit 产生的状态进行展示，不直接发出机器人动作。

下一步
------

- 接入新的机器人或仿真环境：:doc:`add_robot`。
- 添加 VLA 或原语：:doc:`add_primitive`。
- 了解记忆功能的设计与扩展点：:doc:`memory`。
