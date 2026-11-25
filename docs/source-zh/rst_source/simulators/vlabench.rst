VLABench
============

在 `VLABench <https://github.com/OpenMOSS/VLABench>`_ 中用 RPent 运行语言条件的桌面操作任务。规划器通过位姿和夹爪工具控制 Franka 机械臂，并用标定过的 RGB-D 相机定位目标；不使用动作模型。

.. _vlabench-overview:

概览
------------

先确认运行环境要求，再按安装和运行步骤操作。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 动作模型

      无。规划器直接调用 ``move_pose``、``move_delta`` 和 ``set_gripper``。

   .. grid-item-card:: 规划器

      见 :doc:`../guides/configure_planner`。

   .. grid-item-card:: 任务

      VLABench 评测 track 中列出的任务。

   .. grid-item-card:: 硬件

      Linux；Python 3.10；EGL 渲染。

任务
~~~~~~~~~~~~

一次运行由任务名、track 和 episode 编号确定一个 episode。VLABench 目录中的 ``configs/evaluation/tracks/<track>.json`` 列出每个任务的 episode。``--track`` 默认为 ``track_1_in_distribution``；加上 ``--eval-unseen`` 使用未见物体划分。

.. _vlabench-observation-action:

观测与动作
~~~~~~~~~~~~~~~~~~~~~~

下表区分规划器工具、观测和环境的成功判据。

.. list-table::
   :header-rows: 1

   * - 项目
     - 说明
   * - 观测
     - ``agentview``、``left``、``right`` 和 ``wrist`` 四个相机的 RGB、度量深度与标定（512 x 512），Franka 关节与手指状态，以及任务指令。
   * - 动作
     - ``move_pose`` 通过有界 Franka IK 把末端移动到世界系绝对位姿，``move_delta`` 相对测得位姿移动；``set_gripper`` 驱动手指；``back_project`` 读取 ``[row, col]`` 像素处的世界坐标。
   * - 奖励与成功判定
     - 以环境原生的任务成功标志判定成功。规划器结束和步数预算耗尽是另外的结果。
   * - 任务指令
     - 指令在重置后由环境提供。

安装
------------

VLABench 需要 Python 3.10。在 RPent 仓库根目录创建虚拟环境并以 editable 模式安装 RPent，然后按 VLABench 上游说明在同一环境中（或单独的 Python 3.10 环境中，运行时通过 ``--vlabench-python`` 指定）安装 VLABench：

.. code-block:: bash

   uv venv --python 3.10 .venv-vlabench
   source .venv-vlabench/bin/activate
   uv pip install -e .

RPent 不内置 VLABench 源码或其场景资源；环境服务会从你指定的 VLABench 目录导入它。

准备资源
--------------

把 VLABench 体积较大的 ``obj/`` 和 ``scenes/`` 资源放在代码目录之外，并链接到其 ``assets/`` 目录：

.. code-block:: bash

   export VLABENCH_ROOT=/path/to/VLABench/VLABench
   export VLABENCH_ASSET_ROOT=/path/to/vlabench-assets
   ln -s "$VLABENCH_ASSET_ROOT/obj" "$VLABENCH_ROOT/assets/obj"
   ln -s "$VLABENCH_ASSET_ROOT/scenes" "$VLABENCH_ROOT/assets/scenes"

``VLABENCH_ROOT`` 是包含 ``configs/`` 的内层包目录。任一链接缺失或指向别处时，服务会拒绝启动。

运行一个任务
------------

在已激活的环境中运行一个 episode：

.. code-block:: bash

   rpent --robot vlabench \
      --task-name select_painting \
      --track track_1_in_distribution \
      --episode-index 0 \
      --seed 42 \
      --planner codex \
      --model <planner-model>

RPent 会启动 VLABench 环境服务、等待其就绪，并在运行结束时停止它。完整选项见 ``rpent --robot vlabench --help``。

查看结果
------------

默认保存在 ``logs/<timestamp>_vlabench_<task-name>_e<episode>_s<seed>/``。检查运行时先看这些文件：

- ``run.log``：RPent 进程日志。
- ``vlabench_env_server.log``：仿真服务的启动错误。
- ``transcript_<recipe_tag>.json``：规划器对话和最终回复。
- ``result.json``：记录任务标识、原生 ``environment_success``、episode 步数预算状态、原生动作步数和规划器统计信息。

最近一次 ``task_status`` 中的原生任务成功标志是成功判据。调用 ``finish`` 只结束规划器循环；声称成功但环境未确认时，会按 ``failure`` 报告。

常用选项
--------------

.. list-table::
   :header-rows: 1

   * - 选项
     - 含义
   * - ``--vlabench-root``、``--vlabench-asset-root``
     - 覆盖 ``VLABENCH_ROOT`` 和 ``VLABENCH_ASSET_ROOT``。
   * - ``--vlabench-python``
     - 运行环境服务的解释器；默认取 ``VLABENCH_PYTHON``，其次是运行 RPent 的解释器。
   * - ``--max-episode-steps``
     - 原生控制步数预算，默认 10000。这是自定义运行预算，任务的官方上限仅作溯源记录。
   * - ``--env-endpoint``
     - 连接已运行的环境服务，而不是自行启动。
   * - ``--env-cuda-device``
     - 本地服务使用的 ``CUDA_VISIBLE_DEVICES``；默认取 ``--cuda-device``。

VLABench 没有公开的记忆语料，RPent 会创建一个空的本地记忆目录，并且只支持评测模式。
