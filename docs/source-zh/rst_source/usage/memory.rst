使用记忆与探索模式
===========================

记忆用于复用任务经验；探索模式通过多次尝试生成本地经验。先完成 :doc:`../quickstart`，再按本页示例使用 LIBERO 记忆。其他环境的任务参数和成功判定见各自页面。

评测模式只读取记忆，不更新记忆；探索可以复位并尝试多次。正式复现应使用实验指定的记忆版本和评测模式。

使用公开记忆
------------------

评测默认使用 ``hf`` 模式，从 Hugging Face 上的 `RLinf/RPent-memory <https://huggingface.co/datasets/RLinf/RPent-memory>`_ 同步当前机器人的公开数据。下面显式写出该选项：

.. code-block:: bash

   rpent --robot libero --suite libero_10_task --task 0 --seed 1 \
     --planner claude_code --model claude-opus-4-8 \
     --memory-profile hf

下载内容保存在 ``memory/libero/``。设置 ``HF_HUB_OFFLINE=1`` 可跳过同步并使用本地副本。网络不可用时，运行器会记录警告；复现实验前应确认需要的数据已完整下载。

探索并生成本地记忆
---------------------------

以下命令使用独立的本地目录，最多运行 3 个规划会话，每个会话最多尝试 5 次。目录可以为空；``--explore`` 会启用本地模式。重新运行时，请使用新的输出目录：

.. code-block:: bash

   rpent --robot libero --suite libero_10_task --task 0 --seed 0 \
     --planner claude_code --model claude-opus-4-8 \
     --explore --explore-sessions 3 --explore-attempts-per-session 5 \
     --memory-dir ./memory/libero-local \
     --output-dir logs/explore_10_task_t0_s0

每个规划会话创建独立的工具集（toolkit），会话内每次尝试前都会复位环境。状态和观测保存在 ``<output-dir>/sessions/session_NNN/``，经验草稿写入 ``<memory-dir>/_internal/inbox/<cell>/``。

正常结束时，运行器校验并合并草稿、更新索引。只有 LIBERO 判定成功后，才将成功任务的运行记录（audit）和动作序列加入可供后续任务读取的记忆。探索产生的数据保存在本地，不会自动上传。添加 ``--dashboard`` 可观看探索过程，操作方法见 :doc:`dashboard`。

使用本地记忆评测
------------------------

探索结束后，先检查本地目录和运行结果，再用该目录评测另一个场景：

.. code-block:: bash

   rpent --robot libero --suite libero_10_task --task 0 --seed 1 \
     --planner claude_code --model claude-opus-4-8 \
     --memory-profile local --memory-dir ./memory/libero-local

``local`` 模式不下载记忆，也不开启探索。LIBERO 会检查目录中是否已有记忆；空目录会报错。``hf`` 模式不能与 ``--memory-dir`` 同用。

在对话记录中查看记忆读取工具的调用，确认它读取了当前任务允许使用的文件。历史坐标、像素和姿态只能作为参考，执行动作前仍需根据当前观测定位。

检查与人工合并
---------------------

检查记忆文件格式，并根据已有经验重建索引：

.. code-block:: bash

   rpent-memory --memory-dir ./memory/libero-local validate
   rpent-memory --memory-dir ./memory/libero-local build-index

如需先审核草稿，在探索命令中加入 ``--no-auto-merge-memory``。审核完成并核实 LIBERO 的成功结果后，可用以下命令合并对应任务；``--solved`` 是人工提供的成功标记，不能仅凭规划器的总结设置：

.. code-block:: bash

   rpent-memory --memory-dir ./memory/libero-local merge \
     --cell 10_task_t0_s0 --output-dir logs/explore_10_task_t0_s0 --solved

格式校验不证明任务真实成功。目录、权限和发布规则见 :doc:`../development/memory`。

其他环境
------------

- :doc:`robocasa` 和 :doc:`robotwin` 支持探索；任务参数、复位行为和记忆范围见各自页面。
- :doc:`dual_franka` 探索需要现场操作员复位场景并确认结果，默认不自动合并记忆。先完成部署与自检，再开始探索。
- :doc:`franka` 当前不支持探索模式。
