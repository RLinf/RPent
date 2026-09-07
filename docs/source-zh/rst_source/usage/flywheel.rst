LIBERO 数据飞轮
===============

可选的数据飞轮功能会记录 LIBERO 评测中实际执行的轨迹，但不会改变规划器或动作
原语。它首先保存不可变的原始轨迹；转换为训练格式是独立的后续步骤。

采集轨迹
--------

在普通 LIBERO 评测命令中开启采集，并指定数据根目录：

.. code-block:: bash

   rpent --robot libero \
     --suite libero_goal --task 0 --seed 0 \
     --planner codex \
     --collect-flywheel-data \
     --flywheel-root /path/to/datacollection

每次运行会在
``/path/to/datacollection/raw/libero/<suite>/task_<id>/seed_<seed>/`` 下写入一条
轨迹，其中包含策略观测、实际执行的动作、奖励、终止标记、原语调用编号和 VLA
提议的动作序列。采集功能默认关闭，首版仅支持评测模式。

校验并导出成功轨迹
------------------

使用轨迹前可以单独校验原始数据：

.. code-block:: bash

   rpent-flywheel validate /path/to/raw/episode

将同一任务套件中某个任务下所有已完成的成功轨迹导出为 LeRobot 数据集：

.. code-block:: bash

   rpent-flywheel export-lerobot \
     --data-root /path/to/datacollection \
     --suite libero_goal \
     --task 0 \
     --dataset-id goal-task-00 \
     --output-root /path/to/lerobot

导出程序不会改写原始轨迹。失败轨迹会继续保留以便审计，但不会进入这份监督训练
数据。

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
