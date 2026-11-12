.. _pyrua-mode:

使用 PyRUA 模式
================

**PyRUA 模式**让规划器通过 Python 代码片段调用 RPent 已注册的机器人工具。代码可以根据工具返回值执行条件判断和有限循环；同一会话中的后续代码片段可复用先前定义的变量、函数和导入的模块。使用 ``--policy-mode code`` 启用此模式。

整个过程可以概括为：

1. 规划器决定 **做什么**，并提交 Python 代码片段。
2. Python 运行时执行代码，机器人工具集处理每次 ``robo.<tool_name>(...)`` 调用。
3. RPent 把执行输出和最近一次工具结果返回给规划器，再由规划器决定下一步。

先按照 :doc:`../get_started/quickstart` 完成 LIBERO-PRO 的安装和模型配置。以下示例沿用普通 LIBERO-PRO 运行所需的环境与模型服务。


运行一个任务
------------

在已能正常运行的任务命令中加入 ``--policy-mode code``。下面的示例使用种子 0 运行 LIBERO-PRO Object Swap 套件中的任务 2。

.. code-block:: bash

   rpent --robot libero --libero-type pro \
     --suite libero_object_swap --task 2 --seed 0 \
     --planner claude_code --model claude-opus-4-8 \
     --policy-mode code

此模式支持 ``api``、``claude_code`` 和 ``codex`` 规划器，不能与 ``--planner flash`` 或 ``--explore`` 一起使用。代码模式不加载任务记忆，也不生成动作序列（recipe）。省略 ``--policy-mode code`` 即可使用默认的直接工具调用模式。

CLI 和 Dashboard 的任务运行都支持代码模式。运行时会为所选机器人加载专用的代码模式提示词；首次发送给规划器的用户消息是运行配置中的 ``task_desc`` 直接序列化得到的 JSON，不会用 ``prompt_vars`` 补充内容，也不会渲染普通工具调用模式的用户提示词。例如，上述 LIBERO-PRO 任务的初始消息只包含 suite、task 和 seed。


执行原理
------------

规划器通过 ``python(code)`` 工具提交代码片段。代码使用关键字参数调用已注册的机器人工具，例如 ``robo.<tool_name>(argument=value)``；可用方法取决于所选机器人。使用 ``print(...)`` 将变量值写入返回输出，单独写表达式不会显示结果。规划器查看返回结果后，可以继续提交代码片段并复用先前的变量。

每个代码片段默认最多运行 120 秒（包括机器人工具调用），最多调用 100 次机器人工具。可分别使用 ``--code-timeout-s`` 和 ``--code-max-calls`` 调整限制。Python 异常会保留会话变量；超时或取消会先等待正在进行的机器人工具调用结束，再清空 Python 命名空间。已执行的机器人动作不会回滚，继续运行前应先检查当前场景。

``robo.finish(...)`` 调用成功后，当前代码片段和任务都会结束；``api``、``claude_code`` 和 ``codex`` 规划器均会收到该完成结果。


查看运行记录
------------

运行记录保存在任务输出目录下的 ``code_policy/``；可通过 ``--output-dir`` 指定输出目录。``cell_*.py`` 是代码片段，``cell_*.json`` 是执行结果，``calls.jsonl`` 记录机器人工具调用。LIBERO-PRO 的任务成功与否以运行目录中 ``transcript_*.json`` 的 ``environment_success`` 为准。


LIBERO-PRO 评测结果
-------------------

本评测采用 Codex 规划器和 GPT-6 Astra 模型，推理强度设为 high，覆盖 LIBERO-PRO 的 Spatial、Object、Goal 和 Long 四个任务套件。每个任务套件包含 task 和 swap 两个套件，各套件均使用 seed 0，运行任务 0–9，共评测 80 个回合。下表按任务套件汇总结果，并报告总体表现；token 用量、耗时和轮数均在具有 transcript 记录的回合上取平均值。

.. list-table:: PyRUA 在 LIBERO-PRO 上的单种子评测
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

成功率按成功回合数占全部 80 个计划回合的比例计算。耗时取运行记录中的 ``elapsed_s``，从环境运行时准备完成后开始统计；规划轮数取 ``stats.turns_used``。
