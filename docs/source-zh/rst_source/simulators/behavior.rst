BEHAVIOR
========

使用 RPent 运行官方 BEHAVIOR-1K 轨迹。Planner 选择工具；官方 evaluator 执行
机器人原生动作，并负责推进仿真、终止和评分。
`源码使用指南 <https://github.com/RLinf/RPent/blob/main/robots/behavior/guides/README.zh-CN.md>`_
提供完整的安装和启动命令。

.. _behavior-overview:

概览
----

每条轨迹使用独立的 Env 服务和 evaluator 连接。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 动作模型

      RLinf Pi0.5 BEHAVIOR；可选夹爪、手臂和底盘动作原语。

   .. grid-item-card:: 规划器

      ``api``、``claude_code``、``codex``

   .. grid-item-card:: 任务

      100 个挑战赛任务 ID，``0..99``。

   .. grid-item-card:: 硬件

      Linux 与 NVIDIA GPU；兼容的 OmniGibson 图形依赖和模型 CUDA 环境。

任务
~~~~

能识别任务只说明接入范围，不代表已经测得该任务的成功率。

.. list-table::
   :header-rows: 1

   * - 范围
     - 选择方式
     - 运行方式
   * - 100 个官方挑战赛任务
     - 提供匹配的任务 ID、官方名称和完整 main-task 文字。
     - 每个 Env 服务连接一个 R1Pro 环境，运行一条轨迹。
   * - Public-test 启动示例
     - 任务 ``40``，``make_microwave_popcorn``；split 下标 ``0`` 对应 instance ``301``。
     - 保留官方任务步数预算，使用 ``--num-envs 1``。

.. _behavior-observation-action:

观测与动作
~~~~~~~~~~

Planner 调用的工具与仿真器接收的原生动作是两个不同的接口。

.. list-table::
   :header-rows: 1

   * - 项目
     - 说明
   * - 观测
     - 按头部、左腕、右腕顺序提供 RGB 和公开机器人状态。几何工具需要同帧深度与标定。
   * - 动作
     - Planner 调用 VLA 或运动工具，环境接收 23 通道原生动作。RLinf Pi0.5 预设使用 ``abs_joint``、BF16、5 步去噪和 32 动作的 chunk。
   * - 奖励与成功判定
     - 以官方 evaluator JSON 中的 success 和 QScore 为准。工具完成和 Planner 的 ``finish`` 不是官方任务评分。
   * - 任务指令
     - VLA、Env 服务和 Planner 使用一致的完整 main-task 指令。main-task VLA 服务会拒绝不同的指令。

安装与资源准备
--------------

按官方 `BEHAVIOR-1K 说明 <https://github.com/StanfordVL/BEHAVIOR-1K>`_
安装仿真器、evaluator 和挑战赛资源。仿真、模型、Planner 分别使用独立环境。
RPent 支持 Python 3.10–3.12；仿真器和模型环境还需满足各自的 Python、CUDA
和图形依赖要求。

在源码目录内安装轻量的 Planner／Env 依赖：

.. code-block:: bash

   python -m pip install -e '.[behavior]'

VLA 环境需要 ``.[behavior-vla]``，以及单独安装的、支持 BEHAVIOR 的 RLinf。
后者必须提供 ``rlinf.models.embodiment.openpi_pytorch.get_model``、
``pi05_behavior`` 和 ``predict_action_batch(..., mode="eval")``。
准备兼容的 checkpoint、归一化统计和 tokenizer 资源；按 RLinf 的要求设置
``OPENPI_DATA_HOME``。安装 RPent extra 不会自动安装某个可用的 RLinf fork，
也不会下载这些模型资源。

完整 Agent 还需要共享 SAM3 服务、对应权重和 ``.[sam3]`` 环境，以及用于
``move_to`` 的 CuRobo v0.7.8 和兼容的 PyTorch/Warp/CUDA。不需要这些可选工具时，
先使用 ``vla_only.yaml``。按 :doc:`../guides/configure_planner` 配置支持图像输入
的 Planner；该模型与 VLA checkpoint 分别设置。

运行一个任务
------------

按源码指南中的任务 ``40`` 示例，设置安装路径、源码目录外的新输出目录，并在每个
终端中使用一致的任务变量。服务按以下顺序启动：

1. **VLA：** ``python -m robots.behavior.vla_server`` 加载权重，在内部 RPC 端口
   ``28011`` 提供预测；它不推进仿真。
2. **Env 桥接层：** ``python -m robots.behavior.env_server`` 连接已就绪的 VLA，
   在 ``28010`` 提供内部 Env RPC，在 ``28012`` 提供官方策略 WebSocket。
3. **官方 evaluator：** ``python -m omnigibson.eval.eval`` 连接策略端口
   ``28012``，运行一条 public-test 轨迹。使用 ``--num-envs 1`` 和
   ``--write-video --video-fps 120``，保留官方步数预算。
4. **Planner：** 首帧到达后，运行 ``python -m rpent.cli.main --robot behavior``，
   提供 Planner 模型、任务 ID／文字、服务端点，以及
   ``--behavior-config robots/behavior/configs/vla_only.yaml``。

每条命令使用各自准备好的 Python 环境和独立终端。完整参数与变量定义见
`单轨迹启动示例 <https://github.com/RLinf/RPent/blob/main/robots/behavior/guides/README.zh-CN.md#运行一条轨迹>`_。
``vla_only.yaml`` 仍由 VLM 选择观测、VLA 和结束工具，因此不是固定执行循环的
纯 VLA baseline。

使用 ``agent.yaml`` 时，还要启动 SAM3，为 Env 服务启用
``--motion-backend curobo``，并向 Planner 传入 ``--sam3-endpoint``。
``agent.yaml`` 启用 12 个机器人工具，RPent 的共享文件／记忆工具仍然保留。
凭据和服务商 URL 放在 Planner 配置中，不写入机器人模板。

查看结果
--------

示例在源码目录外的运行根目录中，分别保存三类结果：

- ``agent/``：Planner 对话、保存的观测和工具结果。
- ``vla/``：模型配置、推理耗时和预测动作 chunk。
- ``official/``：官方 JSON 评分和视频；任务成功以该 JSON 为准。

示例将视频编码设置为 120 FPS。播放速度不会设置仿真步长或动作频率。
服务正常退出或工具调用成功，都不代表官方任务成功。

任务记忆
--------

与其他仿真环境一样，BEHAVIOR 通过 RPent 共享的文件工具读取本地经验库。
用 ``--memory-dir`` 为 Planner 指定目录；未指定时使用默认的 ``memory/behavior/``。
Planner 在查看初始观测后读取 ``MEMORY.md``，并在动作前最多读取三个相关条目。
经验库目录结构见 :doc:`../development/memory`；BEHAVIOR 评测只读不写，也不同步远程经验库。

运行约定与问题排查
------------------

RPent 借用这些服务：关闭 Planner 会关闭客户端连接，不会停止服务进程。
运行结束后需要显式关闭服务。下一条轨迹需要新的 Env 服务和输出目录；本接入
尚不支持 Dashboard 启动和多个 evaluator 会话。

保存的观测 ``step`` 是记录编号，不是动作数。只读感知工具既不刷新相机，也不
推进仿真。几何坐标属于所查询的保存帧，移动后需重新定位。``move_to`` 支持
由 IK 求解的短距离单臂调整，不提供通用抓取规划或路径级碰撞规避。``move_base``
不提供可靠里程计或到达保证。

已下发、已确认、未确认动作分别计数。未知执行结果保持未知，超时的 chunk 不会
自动重放。连接失败时查看服务日志；官方策略的 ``ws://`` 地址
不能作为 ``--env-endpoint`` 的 RPC 地址。

开发与测试
----------

机器人本体动作工具写在 ``primitives.py``；观测、感知、坐标查询和会话工具
写在 ``tools.py``；``toolkit.py`` 统一注册并绑定运行资源。``motion.py``
提供环境侧 CuRobo 规划和机器人模型，不注册工具。

`源码指南的文件导航 <https://github.com/RLinf/RPent/blob/main/robots/behavior/guides/README.zh-CN.md#文件结构与开发入口>`_
列出配置、提示词、工具、运动与服务适配的位置。通用决策文字在
``prompts/evaluate.py`` 中修改；工具参数和结果语义在对应带类型标注的
``@tool`` 定义中维护。任务目标保持动态输入。

按 `测试指南 <https://github.com/RLinf/RPent/blob/main/tests/README.md>`_
运行离线 `BEHAVIOR 单元测试 <https://github.com/RLinf/RPent/tree/main/tests/unit_tests/robots/behavior>`_
或显式的 `真实服务检查 <https://github.com/RLinf/RPent/tree/main/tests/e2e_tests/behavior>`_。
接入测试验证接口行为；任务效果需要在记录完整且条件可比的官方测评中确认。
