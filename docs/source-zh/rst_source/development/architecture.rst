系统设计
========

本页从实现层面看 RPent —— 核心控制链路中的三个进程各自持有什么、
如何通信，以及 ``rpent/`` 与 ``robots/`` 下的代码如何组织。
更高层的框架介绍见 :doc:`../overview`。

.. raw:: html

   <div style="text-align: center;">
     <img src="https://github.com/RLinf/misc/raw/main/pic/rpent_framework.png" alt="RPent 框架"
          style="max-width: 95%; height: auto;" />
   </div>

关键特性
--------

本节介绍让 RPent 区别于其他具身智能体框架的设计选择。

**可重试的 VLA 工具。** RPent 不训练直接输出动作的端到端策略，
而是让一个通用 LLM 作为 planner，把 VLA 当作 ``pi0_pick``、
``pi0_doubled`` 这样的 action primitive 来调用，与 ``move_to``、
``rotate_wrist``、``back_project`` 等脚本化工具同处一套工具 schema。
每次调用的文本和图像都会回传给 LLM，让它根据实际所见决定下一步；
配合按机器人保存的 memory，planner 还能学到 VLA 在什么时候、
什么条件下才可靠。这样既用上了 LLM 的通用推理和即时纠错，
又不必为每个新任务重新训练模型。新增 primitive 的方法见 :doc:`add_primitive`。

**可替换的 planner。** planner 就是驱动工具调用循环的 LLM agent 运行时。
一个 ``--planner`` 参数就能切换它，而工具和提示词保持不变。内置三种：
``api`` 是 RPent 自研的工具调用循环（基于 pydantic-ai，为默认值，
不绑定具体模型提供商）；``claude_code`` 复用 Claude Agent SDK 运行时；
``codex`` 复用 Codex SDK 运行时。由于三者面对完全相同的工具，
可以在同一套物理基准上正面对比。配置方法见 :doc:`../usage/configure_planner`。

**隔离的仿真环境。** 仿真器作为独立的 env_server 运行，
通过轻量 RPC 与 agent 通信；agent 一侧不导入任何仿真器，
也不与具体环境绑定。更换环境只需实现同一套 env-client 接口——
环境可以自行重启、迁到另一台机器，或替换成另一个仿真器，
而无需改动 planner 或工具。新增机器人也不需要注册代码：
把包放到 ``robots/`` 下，框架就会自动发现。接入新机器人的步骤见 :doc:`add_robot`。

LLM-in-the-loop 运行流程
------------------------

一次运行就是一段 LLM-in-the-loop 循环：

1. LLM 分析任务、调一个工具 (如 ``pi0_pick``)。
2. 工具的底层 primitives 向 ``vla_server`` 请求动作 (``predict``)。
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
     planner/        # planner 实现：api_loop、claude_code、codex、base。
     cli/            # main.py 入口和交互式终端。
     context/        # 提示词工具和共享提示词片段。
     dashboard/      # FastAPI 监控页面和 SSE 事件流（可选）。
     robots/         # RobotSpec、PromptBundle 和按需加载机器人的逻辑。
     tools/          # Toolkit 基类和共享 tool 辅助函数。
     utils/          # 配置、日志、RPC 客户端/服务端和 VLA 客户端。
   robots/
     libero/         # LIBERO 的 env_client / env_server / vla_server /
                     # toolkit / prompt_bundle。参考实现。
     robocasa/       # RoboCasa 机器人 (RLDX-1 VLA，厨房任务)。
     (franka/)       # Franka 机器人——研发中。
     (so101/)        # SO-101 机器人——研发中。
   scripts/
     codex_proxy/    # Codex planner 用的 LiteLLM 代理。
     robocasa/       # RoboCasa 运行 / 安装 / 扫描脚本。

Runner (``rpent/cli/main.py``)
------------------------------

``rpent/cli/main.py`` 负责串联一次运行所需的配置、服务和模型组件。
启动后，它依次执行以下步骤：

1. 调用 ``parse_known_args`` 初步解析通用 CLI 参数
   （常用参数见 :doc:`../quickstart`），先读取 ``--robot`` 和
   ``--dashboard``。
2. 根据 ``args.robot_name`` 调用 ``get_robot_spec`` 加载机器人定义，再通过
   ``robot_spec.add_cli_args(parser, use_dashboard=args.dashboard)`` 将该机器人
   的专用参数加入共享 parser。启用 Dashboard 时，原本必填的机器人参数会暂时
   设为可选，因为任务参数随后通过 Dashboard 命令提供。
3. 再调用 ``parser.parse_args()``，对完整参数集合执行 argparse 层的校验，
   并生成最终的 ``args``；参数错误仍使用 argparse 的标准提示格式。
4. 如果启用了 ``--dashboard``，将控制权交给 ``rpent/cli/dashboard.py``，并在
   长生命周期 Session 结束后返回。Dashboard 专用生命周期见下文；后续步骤属于
   普通 CLI 路径。
5. 调用 ``robot_spec.parse_config(args)`` 校验普通 CLI 的运行配置，并生成
   :class:`~rpent.robots.RunConfig`，其中包含 ``recipe_tag``、``output_dir``、
   ``prompt_vars`` 和 ``task_desc``。
6. 调用 ``init_output_dir`` 创建本次运行的输出目录，并配置 ``run.log``。
7. 根据 ``--planner`` 调用 ``rpent.planner.base.build_planner`` 构造
   **planner**，并使用机器人提供的 prompt bundle 生成 system prompt 和
   user prompt。
8. 调用 ``robot_spec.init_runtime(args, output_dir, dashboard_events, None)``。环境实现会
   启动或连接该环境所需的运行时服务，例如 ``env_server``、``vla_server``，
   以及可选的辅助服务（如 LIBERO 用于分割的 ``sam3_server``），并返回
   ``(daemons, primitives_kwargs)``。
9. 将 ``primitives_kwargs`` 和 ``dashboard_events`` 事件接收器传给机器人的
   ``get_toolkit`` 工厂，构造 **toolkit**。一次性运行链路使用不执行任何
   操作的事件接收器。
10. 执行工具调用循环。循环结束后保存
    ``<output_dir>/transcript_*.json``，并在清理 toolkit 时完成回合录像等
    收尾工作。

``main.py`` 只负责连接上述步骤。机器人相关实现集中在 ``robots/<robot>/``，
planner 后端集中在 ``rpent/planner/``，
因此 ``main.py`` 不直接导入任何机器人专用的类或脚本。

机器人加载机制
--------------

``rpent/robots/base.py`` 根据机器人名称按需加载对应的实现。传入的机器人名称为
``myrobot`` 时，它会执行 ``importlib.import_module("robots.myrobot")``，
再调用该包提供的两个工厂：

.. code-block:: python

   # robots/myrobot/__init__.py
   def get_robot_spec() -> RobotSpec: ...  # 机器人标识、提示词模板与 Runner 钩子
   def get_toolkit(
       *, primitives_kwargs, dashboard_events
   ): ...

``RobotSpec`` 汇集了机器人标识、prompt 模板、可选的 Dashboard 描述与三个 Runner
钩子（``add_cli_args`` / ``parse_config`` / ``init_runtime``）。各字段要填什么见
:doc:`interfaces`。

加载器本身不维护机器人名称列表。当前 CLI 将 ``--robot`` 限定为 ``libero``
和 ``robocasa``；接入新的机器人名称时，还需要同步更新 CLI 的可选值。完整步骤见
:doc:`add_robot`。

Planner、Toolkit 与 RPC 传输层
------------------------------

这三层各管一段、层层解耦。planner 只通过 ``get_tools_spec`` 拿到工具清单、
用 ``execute_tool`` 逐个调用，并不关心工具背后是脚本还是 VLA；
toolkit 把每次工具调用翻译成对 primitive 的调用，再由 primitives
经 RPC 向 ``env_server`` / ``vla_server`` 发起 ``reset`` / ``step`` /
``predict`` 请求；RPC 传输层（HTTP 或 socket）只负责把这些调用和 NumPy
观测在进程间搬运，对上层透明。正因如此，换 planner 不影响工具，
换传输协议也不影响 planner。三者的具体接口契约（``Planner.solve``、
``Toolkit.add_tool``、``RpcFacade._dispatch``）集中在 :doc:`interfaces`。

Dashboard（可选）
-----------------

``rpent/dashboard/`` 由 FastAPI 应用和静态前端组成。启用 ``--dashboard`` 后，
``rpent/cli/main.py`` 会将控制权交给 ``rpent/cli/dashboard.py``，由后者根据
``--dashboard-host`` 和 ``--dashboard-port`` 启动 Dashboard，并在启动共享服务前
确认配置，然后用共享 component 名称调用一次 ``robot_spec.init_runtime``。环境必须
提供 ``robot_spec.dashboard``，由它定义
前端使用的任务命令与字段、runtime components 和 frame channels。Session
controller 随后等待该环境定义的命令（LIBERO 使用 ``/rpent-task``）；每次取得一个
TaskRun 后，Dashboard 会调用 ``parse_config``，再用 unique component 名称调用
同一个 ``robot_spec.init_runtime``，合并 shared 与 unique primitive 参数，并新建 toolkit
和 planner conversation。两个子集都来自环境 Dashboard spec 中显式声明的
``shared`` / ``unique`` scope。在 LIBERO 中，VLA 和 SAM3 会在 Dashboard 运行期间
复用，每个 TaskRun 使用独立环境并按顺序执行。

TaskRun 运行期间，Dashboard 页面提供：

- planner 输出以及工具调用事件；
- 实时固定相机和腕部相机画面；
- 动作时间线和单步动作片段；
- 运行结束后的完整回合录像（如果已生成）。

页面可以提交普通 planner 消息、新任务命令和中断请求，但不会直接发出机器人
动作。planner、toolkit 和机器人运行时通过 ``dashboard_events`` 事件接收器发布
展示更新。
服务端通过 SSE 推送运行状态摘要，前端再按需读取详细事件、时间线和图像。

下一步
------

- 接入新的机器人或仿真环境：:doc:`add_robot`。
- 添加 VLA 或原语：:doc:`add_primitive`。
- 了解记忆功能的设计与扩展点：:doc:`memory`。
