.. _pyrua-mode:

使用 PyRUA 模式
================

**PyRUA 模式** 让规划器通过 Python 代码片段调用 RPent 已注册的机器人工具。代码可以根据工具返回值执行条件判断和有限循环；同一会话中的后续代码片段可复用先前定义的变量、函数和导入的模块。使用 ``--policy-mode code`` 启用此模式。

整个过程可以概括为：

1. 规划器决定 **做什么**，并提交 Python 代码片段。
2. Python 运行时执行代码，机器人工具集处理每次 ``robo.<tool_name>(...)`` 调用。
3. RPent 把执行输出和最近一次工具结果返回给规划器，再由规划器决定下一步。

先按照 :doc:`../get_started/quickstart` 完成 LIBERO-PRO 的安装和模型配置。以下示例沿用普通 LIBERO-PRO 运行所需的环境与模型服务。


运行一个任务
------------

在普通 LIBERO-PRO 任务命令中加入 ``--policy-mode code``，即可启用 PyRUA 模式。下面的示例使用 Codex 规划器和 GPT-6.1 Sol 模型，推理强度设为 low，在 LIBERO-PRO Object Swap 套件中运行任务 2，seed 0。

.. code-block:: bash

   rpent --robot libero --libero-type pro \
     --suite libero_object_swap --task 2 --seed 0 \
     --planner codex --model gpt-6.1-sol \
     --policy-mode code --reasoning-effort low

PyRUA 模式支持 ``api``、``claude_code`` 和 ``codex`` 规划器，可通过 CLI 和 Dashboard 启动，不能与 ``--planner flash`` 或 ``--explore`` 一起使用。PyRUA 模式不加载任务记忆，也不生成动作序列（recipe）。

在标准 LIBERO 上加 ``--no-vla`` 可使用 SAM3 和几何控制。Python 代码通过 ``segment`` 和 ``back_project`` 定位，再用 ``move_to``、``move_pose``、``set_gripper`` 和 ``release`` 执行动作，根据返回的状态判断下一步。通过 ``SAM3_CHECKPOINT_PATH`` 指定本地权重，或用 ``--sam3-endpoint`` 连接已有服务。此模式无需启动动作模型。需要查看图片时显式调用 ``view_env_state``。

.. code-block:: bash

   rpent --robot libero --libero-type standard \
     --suite libero_object --task 0 --seed 0 \
     --planner claude_code --model deepseek-v4.1-flash-1017 \
     --policy-mode code --no-vla


执行原理
------------

启用 PyRUA 模式后，RPent 根据所选机器人加载专用提示词，引导规划器通过 Python 代码调用机器人工具。规划器通过 ``python(code)`` 提交代码，用 ``robo.<tool_name>(argument=value)`` 调用机器人工具。用 ``print(...)`` 显示结果；变量可在后续代码片段中复用。

每段代码默认限时 120 秒（含工具调用），最多调用 100 次机器人工具，可分别用 ``--code-timeout-s`` 和 ``--code-max-calls`` 调整。调用 ``robo.finish(...)`` 并成功返回结束标记后，RPent 停止当前代码片段并结束本次规划。


查看运行记录
------------

运行记录保存在任务输出目录下的 ``code_policy/``；可通过 ``--output-dir`` 指定输出目录。``cell_*.py`` 是代码片段，``cell_*.json`` 是执行结果，``calls.jsonl`` 记录机器人工具调用。


LIBERO-PRO 评测结果
-------------------

本评测采用 Codex 规划器和 GPT-6 Astra 模型，推理强度设为 high，覆盖 LIBERO-PRO 的 Spatial、Object、Goal 和 Long 四类任务。每类任务包含 task 和 swap 两个套件，各套件均使用 seed 1，运行任务 0–9，共评测 80 个回合，总体成功率为 80.00%（64/80）。下表展示各 suite 的评测结果；Token 用量、耗时和 Planner 轮数均在具有 transcript 记录的回合上取平均，包含成功与失败回合。

.. list-table:: PyRUA 在 LIBERO-PRO 上的单 Seed 评测
   :header-rows: 1

   * - 任务套件
     - 成功率（成功/总数）
     - Token（输入/输出）
     - 耗时（秒）
     - 规划轮数
   * - Object Task
     - 100.00%（10/10）
     - 409,966 / 1,057
     - 152.8
     - 11.6
   * - Object Swap
     - 100.00%（10/10）
     - 396,309 / 1,083
     - 165.3
     - 11.5
   * - Spatial Task
     - 80.00%（8/10）
     - 385,299 / 1,017
     - 147.4
     - 11.3
   * - Spatial Swap
     - 90.00%（9/10）
     - 1,109,798 / 1,764
     - 278.9
     - 17.4
   * - Long Task
     - 50.00%（5/10）
     - 1,655,538 / 2,610
     - 481.6
     - 24.0
   * - Long Swap
     - 60.00%（6/10）
     - 2,150,170 / 3,361
     - 690.4
     - 29.5
   * - Goal Task
     - 80.00%（8/10）
     - 824,506 / 1,580
     - 258.6
     - 16.1
   * - Goal Swap
     - 80.00%（8/10）
     - 962,640 / 1,831
     - 300.7
     - 16.6

耗时取运行记录中的 ``elapsed_s``，从环境运行时准备完成后开始统计；规划轮数取 ``stats.turns_used``。
