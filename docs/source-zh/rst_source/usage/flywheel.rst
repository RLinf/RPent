LIBERO 数据飞轮
===============

可选的数据飞轮功能会记录 LIBERO 评测中实际执行的轨迹，但不会改变规划器或动作
原语。它首先保存不可变的原始轨迹；转换为训练格式是独立的后续步骤。

安装
----

环境准备和仿真资源下载请参考 :doc:`../installation`。在 RPent 仓库根目录安装
LIBERO-PRO 及数据飞轮导出依赖：

.. code-block:: bash

   pip install -e ".[libero-pro,flywheel]"

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
预测的动作序列。采集功能默认关闭，首版仅支持评测模式。

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

其中 ``--data-root`` 是原始数据根目录，``--output-root`` 是导出数据集的父目录。
本例导出的数据集位于 ``/path/to/lerobot/goal-task-00``。

对于成功轨迹，导出从开始到首次完成任务的所有实际执行动作，包含使任务成功的
那一步。其中既有脚本原语动作，也有 VLA 动作；成功后的后续动作不导出。

导出程序不会改写原始轨迹。失败轨迹会继续保留以便审计，但不会进入这份监督训练
数据。

使用 RLinf 训练
---------------

将成功轨迹导出为 LeRobot 数据集后，可以在独立的 RLinf 环境中进行 Pi0.5 监督
微调。RPent 负责采集和导出；模型训练、日志和权重保存由 RLinf 管理。

环境安装和训练配置请参考
`RLinf OpenPI_RLinf 监督微调指南 <https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/sft_openpi_rlinf.html>`_。
训练数据路径应指向导出的 LeRobot 数据集目录，即包含 ``meta/info.json`` 的目录，
而不是原始轨迹目录。

先准备支持 LIBERO LeRobot 数据加载的 RLinf 环境，并将 Pi0.5 LIBERO SFT 配置
保存为 ``/path/to/sft-config/libero_pi05_sft.yaml``，再运行：

.. code-block:: bash

   cd /path/to/RLinf
   unset PYTHONPATH
   export PYTHONPATH="$PWD"
   export EMBODIED_PATH="$PWD/examples/sft"

   .venv/bin/python examples/sft/train_vla_sft.py \
     --config-path /path/to/sft-config \
     --config-name libero_pi05_sft \
     data.train_data_paths=/path/to/lerobot/goal-task-00 \
     actor.model.model_path=/path/to/pi05-checkpoint \
     runner.logger.log_path=/path/to/new-training-output

根据可用 GPU 设置配置中的资源分配和 batch size，并使用与所选模型和数据集
相匹配的 checkpoint 及归一化设置。
