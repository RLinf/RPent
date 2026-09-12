.. _id1:

Task Cards (Low Latency Execution)
==================================

**任务卡（Task Card）** 保存一个 LIBERO 任务的动作序列，并标出这些动作依赖的
关键物体或位置。重放时，RPent 从相机画面中找到它们在当前场景里的坐标，更新动作
坐标，然后按顺序执行任务卡中的动作。通过重放已有动作，任务卡无需在每一步调用
LLM 重新规划，从而降低执行延迟。

整个过程可以概括为：

1. 任务卡决定 **做什么**。
2. SAM3 或 Molmo 判断在当前场景中 **在哪里做**。
3. LIBERO toolkit 在更新后的坐标上执行动作。

因此，``--planner task_card`` 不会调用 LLM 重新规划动作。Molmo 在这里只负责视觉
定位：它在相机画面中指出指定的物体或位置，RPent 再将该像素转换成当前场景坐标。

性能与执行时间
--------------

在 LIBERO Object 的 200 次评测（20 个任务，每个任务 10 个 seed）中，Task Card
成功完成 179 次（89.5%），不使用 reasoning 的 Codex 成功完成 186 次（93.0%）。
Task Card 的平均执行时间为每个 episode 40.9 秒，Codex 为 283.6 秒。

.. image:: ../../_static/task_card_object_performance_time.png
   :alt: Task Card 与不使用 reasoning 的 Codex 在 LIBERO Object 上的逐任务性能和执行时间对比
   :width: 100%
   :align: center

时间统计不包含模型及服务启动时间。Codex 时间是每个任务 10 个评测 seed 的 planner
执行时间均值。Task Card 原始 10-seed 耗时日志已经不可用，因此图中的 Task Card
耗时采用每张最终任务卡对应录制 episode 的工具执行时间（每个任务一个耗时样本）。
两种方法的成功率均来自完整的 200-episode 评测。

重放流程
--------

每张任务卡包含一组动作和一组锚点（anchor）。锚点表示与任务有关的物体或位置，
例如需要抓取的物体、放置目标等。依赖锚点的动作记录的是相对锚点的偏移，而不只是
某次场景中的绝对坐标。

运行时，RPent 从任务卡中提取当前计划需要的锚点，并按照卡中记录的方式逐一定位：

* **SAM3** 处理分割类型的锚点，返回物体掩膜及其位置。
* **Molmo** 处理点定位类型的锚点，在相机画面中指出目标物体或位置。

RPent 将实时锚点位置与任务卡保存的偏移组合成新的路点，再执行对应动作。因此，
即使物体在新布局中换了位置，同一张任务卡仍能使用当前场景的坐标执行。

任务卡文件
----------

任务卡不随 Git 仓库提交，而是通过 Hugging Face 上的 `RLinf/RPent-memory 任务卡目录
<https://huggingface.co/datasets/RLinf/RPent-memory/tree/main/libero/task_card>`_
分发。RPent 会随其他 LIBERO memory 自动下载任务卡，并保存到本地
``memory/libero/task_card``。每个受支持的任务对应一张任务卡。

.. code-block:: text

   memory/libero/task_card/
     object_swap_t3_anchors.json   运行时需要定位的物体和位置
     object_swap_t3_plan.json      动作及其相对锚点的坐标

任务决定使用哪张卡；seed 只改变环境布局，不改变该任务使用的任务卡。

如果只想手动下载任务卡，可以运行：

.. code-block:: bash

   hf download RLinf/RPent-memory --repo-type dataset \
     --include "libero/task_card/**" --local-dir memory

运行任务卡
----------

先启动 Molmo 服务，再把服务地址传给 RPent：

.. code-block:: bash

   rpent --robot libero --planner task_card \
     --suite libero_object_swap --task 3 --seed 0 \
     --molmo-endpoint http://127.0.0.1:20703

任务卡重放目前支持 ``libero_object_task`` 和 ``libero_object_swap``。其他
LIBERO suite 暂时还没有对应的任务卡。

VLA 和 SAM3 沿用普通 LIBERO 运行方式。也可以通过 ``--vla-endpoint`` 和
``--sam3-endpoint`` 连接已经启动的服务。

Molmo 配置
----------

Molmo 需要的 ``transformers`` 版本比 LIBERO 策略环境更新，因此应在独立 Python
环境中运行：

.. code-block:: bash

   uv venv --python 3.11 /path/to/molmo-venv
   /path/to/molmo-venv/bin/pip install -e ".[molmo]"

从 `Hugging Face <https://huggingface.co/allenai/Molmo2-8B>`_ 或
`ModelScope <https://modelscope.cn/models/allenai/Molmo2-8B>`_ 下载
``allenai/Molmo2-8B``，然后启动服务：

.. code-block:: bash

   export MOLMO_CHECKPOINT_PATH=/path/to/Molmo2-8B
   PYTHONPATH=/path/to/RPent /path/to/molmo-venv/bin/python \
     rpent/robots/components/molmo_server.py \
     --transport http --host 127.0.0.1 --port 20703

重放多个布局
------------

使用不同 seed 运行同一任务卡，即可在不同布局上执行。复用已经启动的 VLA、SAM3
和 Molmo 服务，可以避免每次运行都重新加载模型：

.. code-block:: bash

   for seed in $(seq 0 9); do
     rpent --robot libero --planner task_card \
       --suite libero_object_swap --task 3 --seed "$seed" \
       --output-dir logs/sweep/swap_t3_s$seed \
       --vla-endpoint http://127.0.0.1:20701 \
       --sam3-endpoint http://127.0.0.1:20702 \
       --molmo-endpoint http://127.0.0.1:20703
   done
