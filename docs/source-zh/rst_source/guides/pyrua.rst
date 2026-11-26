.. _pyrua-mode:

使用 PyRUA 模式
================

**PyRUA 模式** 让规划器通过 Python 代码片段调用 RPent 已注册的机器人工具。代码可以根据工具返回值执行条件判断和有限循环；同一会话中的后续代码片段可复用先前定义的变量、函数和导入的模块。使用 ``--policy-mode code`` 启用此模式。

整个过程可以概括为：

1. 规划器决定 **做什么**，并提交 Python 代码片段。
2. Python 运行时执行代码，机器人工具集处理每次 ``robo.<tool_name>(...)`` 调用。
3. RPent 把执行输出和最近一次工具结果返回给规划器，再由规划器决定下一步。

运行一个任务
------------

运行 PyRUA 需要启用用户命名空间的 Linux 环境，并安装 Bubblewrap（``bwrap``）。Ubuntu 上可用 ``sudo apt install bubblewrap`` 安装。运行下面的示例前，先按 :doc:`../get_started/quickstart` 安装 LIBERO-PRO、配置模型服务，并准备好 Codex 的模型访问。

在普通 LIBERO-PRO 任务命令中加入 ``--policy-mode code``，即可启用 PyRUA 模式。下面的示例使用 Codex 规划器和 GPT-6.1 Sol 模型，推理强度设为 low，在 LIBERO-PRO Object Swap 套件中运行任务 2，seed 0。

.. code-block:: bash

   rpent --robot libero --libero-type pro \
     --suite libero_object_swap --task 2 --seed 0 \
     --planner codex --model gpt-6.1-sol \
     --policy-mode code --reasoning-effort low --output-dir ./runs/pyrua

在 ``./runs/pyrua/code_policy/`` 查看执行过的代码片段，在 ``transcript_*.json`` 中查看 LIBERO 原生结果 ``environment_success``。如果 worker 启动失败，先确认已安装 ``bwrap`` 且系统允许用户命名空间。进入启动流程后，诊断信息会写入 ``code_policy/worker.stderr.log``。

PyRUA 模式支持 ``api``、``claude_code``、``codex`` 和 ``gemini_cli`` 规划器，可通过 CLI 和 Dashboard 启动，不能与 ``--planner flash`` 或 ``--explore`` 一起使用。PyRUA 模式不加载任务记忆，也不生成动作序列（recipe）。

关闭动作模型
------------

用 ``--policy-mode code --no-am`` 关闭 VLA 和 WAM 服务及工具。LIBERO 的此组合适用于标准套件。先配置标准 LIBERO 和 Claude Code，再用 ``SAM3_CHECKPOINT_PATH`` 指定本地 SAM3 权重，或用 ``--sam3-endpoint`` 连接已有服务。

.. code-block:: bash

   rpent --robot libero --libero-type standard \
     --suite libero_object --task 0 --seed 0 \
     --planner claude_code --model deepseek-v4.1-flash-1017 \
     --policy-mode code --no-am

标准 LIBERO 使用 SAM3 与 RGB-D 定位，再调用原生的运动和夹爪工具执行动作。动作回复包含状态。需要查看图像时，调用 ``robo.view_env_state(step=-1)``。

RoboDojo 也支持 ``--policy-mode code --no-am``，动作由原生 ARX-X5 控制接口执行。环境配置见 :doc:`../simulators/robodojo`。

执行原理
------------

规划器通过 ``python(code)`` 提交代码，用 ``robo.<tool_name>(argument=value)`` 调用机器人工具。用 ``print(...)`` 显示结果，变量可在后续代码片段中复用。

Python 可以导入标准库和 NumPy，并在会话的临时目录写入文件。沙箱隔离 worker 与宿主机的工作区、环境变量和网络。规划器的 Shell 权限独立配置。

每段代码默认限时 120 秒（含工具调用），最多调用 100 次机器人工具，可分别用 ``--code-timeout-s`` 和 ``--code-max-calls`` 调整。调用 ``robo.finish(...)`` 并成功返回结束标记后，RPent 停止当前代码片段并结束本次规划。

如果工具集提供 ``request_scene_reset`` 或 ``request_operator_verdict``，请让规划器单独调用这些工具，等待人工输入的时间由工具自身管理。


查看运行记录
------------

下表路径相对于运行目录下的 ``code_policy/``。可通过 ``--output-dir`` 指定运行目录。

.. list-table::
   :header-rows: 1

   * - 路径
     - 内容
   * - ``cell_*.py`` 和 ``cell_*.json``
     - 提交的代码和执行结果，包括输出与异常。
   * - ``calls.jsonl``
     - Python 发起的机器人工具调用及其结果和耗时。
   * - ``worker.stderr.log``
     - 沙箱启动和 worker 运行诊断。
   * - ``scratch/``
     - 代码在本次会话中写入的文件。


LIBERO-PRO 评测结果
-------------------

PyRUA 模式在完整的 800 回合 LIBERO-PRO 评测中成功完成 663 回合，总体成功率为 82.88%。评测覆盖 Spatial、Object、Goal 和 Long 四类任务的 task、swap 套件；每个套件运行任务 0–9，每个任务使用 seed 1–10。八个套件的成绩见 :doc:`评测成绩 <../leaderboard/performance>`。

:doc:`耗时与 Token 开销 <../leaderboard/time-token-costs>` 表中的平均每回合耗时为 781.98 秒，平均耗时和 Token 用量均包含成功和失败回合。

实验使用 API 规划器和 DeepSeek V4.1 Flash（``deepseek-v4.1-flash-1017``），推理强度为 high，由 Pi0.5 执行动作，不加载任务记忆。每回合最多规划 100 轮；每段 Python 代码限时 120 秒，最多调用 100 次机器人工具。

.. list-table:: PyRUA 在 LIBERO-PRO 上的 800 回合评测
   :header-rows: 1

   * - 任务套件
     - 成功率（成功/总数）
     - 平均总 Token（输入＋输出）
     - 平均耗时（秒）
     - 耗时与 Token 样本数
   * - Spatial Task
     - 95.00%（95/100）
     - 1,104,045.70
     - 567.25
     - 99
   * - Spatial Swap
     - 87.00%（87/100）
     - 1,550,259.47
     - 663.28
     - 94
   * - Object Task
     - 100.00%（100/100）
     - 435,436.86
     - 217.44
     - 100
   * - Object Swap
     - 98.00%（98/100）
     - 527,173.55
     - 263.90
     - 100
   * - Goal Task
     - 79.00%（79/100）
     - 1,775,835.65
     - 769.18
     - 93
   * - Goal Swap
     - 90.00%（90/100）
     - 1,951,382.73
     - 861.88
     - 100
   * - Long Task
     - 60.00%（60/100）
     - 3,345,833.40
     - 1,387.30
     - 100
   * - Long Swap
     - 54.00%（54/100）
     - 3,621,965.83
     - 1,515.42
     - 100
   * - 总体
     - 82.88%（663/800）
     - 1,791,802.63
     - 781.98
     - 786
