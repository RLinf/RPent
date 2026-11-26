.. _pyrua-mode:

使用 PyRUA 模式
================

启用 PyRUA 后，规划器用 Python 调用机器人工具，并根据结果决定下一步。运行记录中保留了代码和工具调用。

运行一个任务
------------

运行 PyRUA 需要启用用户命名空间的 Linux 环境，并安装 Bubblewrap（``bwrap``）。Ubuntu 上可用 ``sudo apt install bubblewrap`` 安装。按 :doc:`../get_started/quickstart` 的前两步安装 LIBERO-PRO、下载仿真资源和模型权重。配置 :ref:`Codex <planner-codex>` 后，用 ``rpent-check-llm --planner codex`` 检查连接。

用下面的命令运行 LIBERO-PRO Object Swap 套件的任务 2，随机种子为 0。``--policy-mode code`` 用于启用 PyRUA 模式。

.. code-block:: bash

   rpent --robot libero --libero-type pro \
     --suite libero_object_swap --task 2 --seed 0 \
     --planner codex \
     --policy-mode code --reasoning-effort low --output-dir ./runs/pyrua

运行结束后，打开 ``./runs/pyrua/``。``transcript_*.json`` 中的 ``environment_success`` 字段记录 LIBERO 的任务判定。代码和工具调用的保存位置见 :ref:`pyrua-run-records`。

命令行和 :doc:`Dashboard <dashboard>` 均支持 ``api``、``claude_code``、``codex`` 和 ``gemini_cli``。启动 Dashboard 时加上 ``--policy-mode code`` 即可启用 PyRUA。使用 ``--planner flash``、``--explore`` 或任务记忆时，请设置 ``--policy-mode tools``。参数列表见 :doc:`cli`。

关闭动作模型
------------

运行 PyRUA 时加上 ``--no-am``，即可关闭动作模型服务和工具，使用机器人原生控制。请在独立于 LIBERO-PRO 的 Python 环境中安装 :doc:`标准 LIBERO <../simulators/libero>`。激活后沿用上面的 Codex 配置，用 ``SAM3_CHECKPOINT_PATH`` 指定本地 SAM3 权重，或用 ``--sam3-endpoint`` 连接已有服务。此示例只需 SAM3 权重。

.. code-block:: bash

   rpent --robot libero --libero-type standard \
     --suite libero_object --task 0 --seed 0 \
     --planner codex \
     --policy-mode code --no-am --output-dir ./runs/pyrua-no-am

在 ``./runs/pyrua-no-am/code_policy/calls.jsonl`` 查看运动和夹爪调用记录。标准 LIBERO 在此模式下的动作回复不附带图像。查看日志中的 ``view_env_state`` 调用可确认规划器何时请求了图像。

下表中的机器人和仿真环境也支持这两个选项。环境、感知和操作员控制的配置见表中的指南。

.. list-table::
   :header-rows: 1

   * - 机器人或仿真环境
     - ``--no-am`` 下可用的控制方式
   * - :doc:`RoboDojo <../simulators/robodojo>`
     - ARX-X5 原生控制。
   * - :doc:`RoboCasa <../simulators/robocasa>` 和 :doc:`RoboTwin <../simulators/robotwin>`
     - 脚本化运动和夹爪工具。
   * - :doc:`RoboLab <../simulators/robolab>`
     - DROID 原生控制。
   * - :doc:`EFE <../simulators/efe>`
     - 由已连接的 EFE 控制器提供工具。EFE 本身无需 RPent 动作模型。
   * - :doc:`Franka <../real_world_robots/franka>`、:doc:`双臂 Franka <../real_world_robots/dual_franka>` 和 :doc:`YAM <../real_world_robots/yam>`
     - 原生运动和夹爪工具，沿用各后端的操作员控制。

单臂和双臂 Franka 的任务 1 均需要动作模型。可用任务 0 验证 ``--no-am`` 下的原生运动和夹爪控制。

.. _pyrua-run-records:

查看运行记录
------------

在 ``--output-dir`` 指定的运行目录中打开 ``code_policy/``。如果 PyRUA 启动失败，确认已安装 ``bwrap``、系统已启用用户命名空间，并查看 ``worker.stderr.log`` 中的诊断信息。

.. list-table::
   :header-rows: 1

   * - 路径
     - 内容
   * - ``cell_*.py`` 和 ``cell_*.json``
     - 提交的代码和执行结果，包括输出与异常。
   * - ``calls.jsonl``
     - Python 发起的机器人工具调用及其结果和耗时。
   * - ``worker.stderr.log``
     - 启动和运行诊断。
   * - ``scratch/``
     - 代码在本次会话中写入的文件。

.. _how-code-execution-works:

调整运行上限
------------

每段代码默认限时 120 秒，包含工具调用时间，最多可调用 100 次机器人工具。可用 ``--code-timeout-s`` 和 ``--code-max-calls`` 调整，超出调用次数会报错。等待操作员复位场景或确认结果的时间不计入代码片段的限时。

代码可使用 Python 标准库和 NumPy，并在 ``scratch/`` 下写入文件。沙箱将这些代码与宿主机工作区和网络隔离。Codex 的终端权限由 ``--codex-sandbox`` 单独设置。

后续代码片段可复用 Python 变量。超时或取消会清空变量，已执行的动作仍然生效。继续运行前，先检查机器人当前状态。

.. warning::

   需要立即停止实体机器人时，请使用硬件急停。

LIBERO-PRO 评测结果
-------------------

PyRUA 模式在 LIBERO-PRO 评测中成功了 663 回合，总计 800 回合，成功率为 82.88%。评测覆盖 Spatial、Object、Goal 和 Long 的 task 与 swap 套件。每个套件运行任务 0–9，每个任务使用 seed 1–10。

成功以 LIBERO 原生判定为准，运行记录中的 ``environment_success`` 为 ``true`` 时计为成功。成功率以全部 800 回合为分母，失败和未完成的回合（含超时）均计为失败。各套件的成绩见 :doc:`评测成绩 <../leaderboard/performance>`。

每回合的平均耗时和平均总 Token 数按 786 条完整的最终记录计算，其中 663 条成功、123 条失败。14 次超时不纳入这两项平均值。耗时从服务就绪后计起，包含运行结束后的清理。重试过的回合只统计最后一次尝试，输入 Token 包含缓存输入。汇总见 :doc:`耗时与 Token 开销 <../leaderboard/time-token-costs>`。

评测使用 API 规划器和 DeepSeek V4.1 Flash（``deepseek-v4.1-flash-1017``），推理强度为 high，由 Pi0.5 执行动作，不加载任务记忆。每回合最多规划 100 轮。每段 Python 代码限时 120 秒，最多调用 100 次机器人工具。

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
