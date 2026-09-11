任务卡模式
==========

**任务卡（Task Card）** 保存一个 LIBERO 任务的动作序列，并标出这些动作依赖的
关键物体或位置。重放时，RPent 从相机画面中找到它们在当前场景里的坐标，更新动作
坐标，然后按顺序执行任务卡中的动作。

整个过程可以概括为：

1. 任务卡决定 **做什么**。
2. SAM3 或 Molmo 判断在当前场景中 **在哪里做**。
3. LIBERO toolkit 在更新后的坐标上执行动作。

因此，``--planner task_card`` 不会调用 LLM 重新规划动作。Molmo 在这里只负责视觉
定位：它在相机画面中指出指定的物体或位置，RPent 再将该像素转换成当前场景坐标。

全系列 LIBERO-PRO 性能与执行时间
-------------------------------------

在完整的 800-case LIBERO-PRO 矩阵（Spatial、Object、Goal 和 Long；
task/swap；每个任务 10 个 seed）上，Task Card 成功 581 次（72.63%）。
不使用 reasoning 的 Codex 成功 500 次（62.50%），high reasoning Codex
成功 628 次（78.50%）。两个没有成功源轨迹、因而没有 Task Card 的任务
按 0/10 保守计入。

.. image:: ../../_static/task_card_libero_pro_performance_time.png
   :alt: Task Card 与 Codex 在 LIBERO-PRO 全系列上的逐任务成功率和执行时间对比
   :width: 100%
   :align: center

时间统计不包含模型及服务启动时间。Codex 时间是每个任务可用 planner 耗时记录的
均值。Task Card 耗时采用每张最终任务卡对应成功 episode 的
工具执行时间（每张卡一个耗时样本）。所有方法的成功率都使用完整 800-case 矩阵。
两组 Codex baseline 均有完整的 800/800 planner 耗时记录。

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

生成任务卡
----------

生成器从一条经模拟器确认成功的 episode 生成一张卡。必需输入只有
episode audit JSON 和与之匹配的 primitive recipe JSONL：

.. code-block:: bash

   python -m robots.libero.task_card.generate \
     --audit results/goal_swap_t3_s7.json \
     --recipe results/recipe_goal_swap_t3_s7.jsonl \
     --destination memory/libero/task_card

audit 必须包含非空的 ``task_language``（或 ``perturbed_task_language``）以及
``libero_terminated: true``。audit 和 recipe 的文件名，以及 audit 中存在的
suite/task/seed 字段，必须指向同一个 episode。

如果 episode 保存了 ``segment_*.json`` 读数，可通过 ``--segments`` 指定目录；
否则生成器会根据任务指令以及 recipe 中有序的抓取/释放或关节交互 transaction，
生成供 Molmo 使用的语义锚点。附近的 ``move_to`` 和 ``move_pose`` 坐标会被保存成
相对锚点的 XY offset，供 task-card replay 直接使用。

关系解析覆盖 LIBERO-PRO 全部 80 个任务：Spatial、Object、Goal、Long（``10``）
各自的 task 和 swap suite。Long 的 transaction 顺序会被保留，例如先打开炉灶再
放置物体，或者先把物体放进设备再关闭设备。

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

任务卡重放支持 LIBERO-PRO Spatial、Object、Goal、Long（``10``）的 task 和
swap suite，共 80 个 task identity。

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
