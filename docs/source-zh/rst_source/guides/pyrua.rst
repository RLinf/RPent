.. _pyrua-mode:

使用 PyRUA 模式
================

PyRUA 模式让规划器通过 Python 代码片段调用 RPent 已注册的机器人工具。代码可以根据工具返回值执行条件判断和有限循环；同一会话中的后续代码片段可复用先前定义的变量、函数和导入的模块。使用 ``--policy-mode code`` 启用此模式。

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


代码如何执行
------------

规划器通过 ``python(code)`` 工具提交代码片段。代码使用关键字参数调用已注册的机器人工具，例如 ``robo.<tool_name>(argument=value)``；可用方法取决于所选机器人。使用 ``print(...)`` 将变量值写入返回输出，单独写表达式不会显示结果。规划器查看返回结果后，可以继续提交代码片段并复用先前的变量。

每个代码片段默认最多运行 120 秒（包括机器人工具调用），最多调用 100 次机器人工具。可分别使用 ``--code-timeout-s`` 和 ``--code-max-calls`` 调整限制。Python 异常会保留会话变量；超时或取消会清空 Python 命名空间，但已执行的机器人动作不会回滚。继续运行前应先检查当前场景。

``robo.finish(...)`` 调用成功后，当前代码片段和任务都会结束。``request_scene_reset`` 和 ``request_operator_verdict`` 等操作员请求是独立的规划器工具，因为等待人工响应可能超过单个代码片段的时间限制。


查看运行结果
------------

运行输出目录中的 ``code_policy/cell_000001.py`` 保存第一个代码片段，``code_policy/cell_000001.json`` 保存其输出、异常（如有）、耗时和最近一次工具结果。``code_policy/calls.jsonl`` 记录每次机器人工具调用。``transcript_*.json`` 还记录 ``policy_mode`` 与 ``environment_success``。对于 LIBERO-PRO，应以环境成功信号判断任务结果；工具调用成功或调用 ``finish`` 本身不代表任务成功。


LIBERO-PRO 评测结果
-------------------

一次单种子评测使用 Codex 规划器和 GPT-6 Astra 模型，推理强度为 ``high``。PyRUA 模式在 80 次运行中成功 64 次，成功率为 80.00%。评测覆盖 LIBERO-PRO 的全部八个套件，每个套件均使用种子 1 运行任务 0–9：

.. list-table::
   :header-rows: 1

   * - 套件
     - 成功次数 / 总次数
     - 成功率
   * - Spatial Task
     - 8/10
     - 80.00%
   * - Spatial Swap
     - 9/10
     - 90.00%
   * - Object Task
     - 10/10
     - 100.00%
   * - Object Swap
     - 10/10
     - 100.00%
   * - Goal Task
     - 8/10
     - 80.00%
   * - Goal Swap
     - 8/10
     - 80.00%
   * - Long Task
     - 5/10
     - 50.00%
   * - Long Swap
     - 6/10
     - 60.00%
   * - 总体
     - 64/80
     - 80.00%

以上结果每个任务仅覆盖一个种子。其他已发布配置及其评测范围见 :doc:`评测成绩 <../leaderboard/performance>`。
