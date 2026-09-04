LIBERO 数据飞轮
===============

可选的 Flywheel 功能会记录 LIBERO evaluation 中实际执行的轨迹，但不会改变
planner 或 action primitive。它首先保存不可变的 raw episode；转换为训练格式是
独立的后续步骤。

采集 episode
------------

在普通 LIBERO evaluation 命令中开启采集，并指定数据根目录：

.. code-block:: bash

   rpent --robot libero \
     --suite libero_goal --task 0 --seed 0 \
     --planner codex \
     --collect-flywheel-data \
     --flywheel-root /path/to/datacollection

每次运行会在
``/path/to/datacollection/raw/libero/<suite>/<task>/<seed>/`` 下写入一个
episode，其中包含 policy observation、实际执行的 action、reward、终止标记、
primitive ID 和 VLA proposal。采集功能默认关闭，首版仅支持 evaluation mode。

校验并导出成功轨迹
------------------

使用 episode 前可以单独校验 raw 数据：

.. code-block:: bash

   rpent-flywheel validate /path/to/raw/episode

将一个 suite/task 下所有已完成的成功 episode 导出为 LeRobot 数据集：

.. code-block:: bash

   rpent-flywheel export-lerobot \
     --data-root /path/to/datacollection \
     --suite libero_goal \
     --task 0 \
     --dataset-id goal-task-00 \
     --output-root /path/to/lerobot

exporter 不会改写 raw episode。失败 episode 会继续保留以便审计，但不会进入这份
监督训练数据。

使用 RLinf 训练
---------------

使用 Flywheel 训练命令，显式指定官方 RLinf checkout、导出的数据集、初始 Pi0.5
checkpoint 和一个全新的输出目录：

.. code-block:: bash

   rpent-flywheel train-rlinf \
     --dataset /path/to/lerobot/goal-task-00 \
     --checkpoint /path/to/pi05-checkpoint \
     --rlinf-root /path/to/RLinf \
     --output-dir /path/to/new-training-output \
     --max-steps 1000 \
     --save-interval 100 \
     --cuda-device 0

该命令会记录 RLinf commit，并使用随 Flywheel 提供的 Pi0.5 配置调用 RLinf 原生
VLA SFT 入口。输出目录必须尚不存在。
