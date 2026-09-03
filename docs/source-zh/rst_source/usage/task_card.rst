任务卡模式
==========

只记录了绝对路点的方案，无法跟随移动过的物体。而如果同时记录下 *定位了什么*
以及 *每个路点离那次读数有多远*，就可以：偏移是任务逻辑，换个布局依然成立，
坐标不是。

**任务卡**就是后一种形式的方案，从一个已解出的 episode 里录制一次。重放时被
替换的只有感知——动作、动作顺序、给策略的提示词、夹爪指令，全都来自任务卡本身，
定位器负责给出它们执行时的坐标。

语料
----

任务卡放在 ``resources/libero/task_card``，与 ``resources/libero/memory`` 下
整理好的 memory 并列，和这份 payload 的其余部分一起流转：从 HuggingFace 资源
数据集同步，或直接读磁盘上已有的副本。和 ``resources/`` 下的所有东西一样，该
目录不纳入 git。

一个任务只有一张卡，所以运行时不做任何挑选：任务决定了卡，卡加上实时定位就
生成轨迹。

.. code-block:: text

   resources/libero/task_card/
     index.json                 全部卡片：任务、来源 episode、任务指令
     object/swap_t3/
       anchors.json             定位过的短语、各自的读数，以及由此得到的锚点
       plan.json                每个动作，标出其坐标背后的锚点与偏移
       trace.md                 已记录动作的可读执行轨迹

语料里不出现 seed。一张卡无论重放到哪个布局上都服务于它那个任务；它从哪个
episode 录来，作为溯源信息保存在卡内部的 ``source`` 字段里，而不是一个可调项。

Molmo 配置
----------

重放用 **Molmo** 定位点类型锚点，由
``rpent/robots/components/molmo_server.py`` 提供服务。SAM3 回答的是"哪些像素
是这个短语"，Molmo 回答的是"你会把夹爪放在哪里"——一个开放词表的点，面向那些
掩膜候选叫不出名字的短语。

Molmo 装在自己的独立环境里，与 LIBERO 环境分开。新建该环境、装上 ``molmo``
extra，然后从
`Hugging Face: allenai/Molmo2-8B <https://huggingface.co/allenai/Molmo2-8B>`_
或 `ModelScope: allenai/Molmo2-8B
<https://modelscope.cn/models/allenai/Molmo2-8B>`_ 下载权重，并通过
``MOLMO_CHECKPOINT_PATH`` 指向它：

.. code-block:: bash

   uv venv --python 3.11 /path/to/molmo-venv
   /path/to/molmo-venv/bin/pip install -e ".[molmo]"

   # Hugging Face
   hf download allenai/Molmo2-8B --local-dir /path/to/Molmo2-8B

   # ModelScope（用它替代上面的 Hugging Face 命令）
   modelscope download --model allenai/Molmo2-8B --local_dir /path/to/Molmo2-8B

   export MOLMO_CHECKPOINT_PATH=/path/to/Molmo2-8B

用该环境的解释器启动服务：

.. code-block:: bash

   PYTHONPATH=/path/to/RPent /path/to/molmo-venv/bin/python \
     rpent/robots/components/molmo_server.py \
     --transport http --host 127.0.0.1 --port 20703

下面两个入口都是连接该服务的地址，而不是自己去启动它。

重放单个 episode
----------------

``--planner task_card`` 和 ``api``、``codex`` 一样是一个 planner 后端，只是由
任务卡决定动作，不调用任何模型。运行的其余部分完全不变：

.. code-block:: bash

   rpent --robot libero --planner task_card \
     --suite libero_object_swap --task 3 --seed 0 \
     --molmo-endpoint http://127.0.0.1:20703

语料里每个任务只有一张卡，所以 seed 选的是要解决的布局，而不是用来解决它的
方案。

重放整轮扫描
------------

整轮扫描就是同一条命令跑更多 cell。把各个端点都指向已经在提供服务的模型，
这样一整轮只加载一次：

.. code-block:: bash

   for seed in $(seq 0 9); do
     rpent --robot libero --planner task_card \
       --suite libero_object_swap --task 3 --seed "$seed" \
       --output-dir logs/sweep/swap_t3_s$seed \
       --vla-endpoint http://127.0.0.1:20701 \
       --sam3-endpoint http://127.0.0.1:20702 \
       --molmo-endpoint http://127.0.0.1:20703
   done

   # 数一下解出了几个
   grep -l '"status": "success"' logs/sweep/*/transcript_*.json | wc -l

评测是单次尝试且不重置环境：失败的 episode 记为失败，不会从干净状态重来。

读数如何变成路点
----------------

每个锚点都 **用它最初被读取的那个接口** 重新读取。``segment`` 来源的锚点重新
分割，因为它的偏移是相对掩膜质心算的，而斜视角下的宽口容器，其质心离指点模型
所指的位置有相当距离。点定位锚点则交给定位器回答。

定位分两级：先对开局画面粗看全场，然后机械臂停到每个点定位锚点上方，用腕部
相机再问一次——此时物体填满视野。近距读数只在与粗读数相差 5 厘米以内时才采纳，
这样腕部视野里认错的东西不会覆盖掉正确答案。

对 *被握持* 物体的腕部读数会落在其中心之前，物体越高偏得越远。该修正与物体
高度成线性关系。

配置项
------

除了常规的 LIBERO 运行参数之外没有别的东西要配。``--suite`` / ``--task`` /
``--seed`` 指定 cell，卡由任务本身决定。唯一新增的是 ``--molmo-endpoint``：
指向一个已在提供服务的定位器。Molmo 的 ``transformers`` 依赖与策略环境冲突，
因此它需要运行在独立环境中，RPent 不会用当前 Python 解释器代为启动。
