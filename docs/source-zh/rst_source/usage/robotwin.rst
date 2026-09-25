RoboTwin
============

.. figure:: https://robotwin-platform.github.io/assets/images/teaser.png
   :alt: RoboTwin 2.0 环境概览
   :width: 90%
   :align: center

   RoboTwin 2.0 概览。图片来源：`RoboTwin 项目 <https://robotwin-platform.github.io/>`_。图中实验结果来自 RoboTwin 项目。

使用 RPent 在 `RoboTwin <https://robotwin-platform.github.io/>`_ 中运行双臂桌面操作任务，并复现随机场景下的 C2R 实验。RPent 通过 RLinf 接入仿真器，使用 LingBot-VLA 生成动作。

.. _robotwin-overview:

概览
------------

先确认所需模型、任务与运行环境，再按后续步骤安装并运行。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 动作模型

      LingBot-VLA

   .. grid-item-card:: 规划器

      ``api``、``claude_code``、``codex``

   .. grid-item-card:: 任务

      50 个 C2R 任务

   .. grid-item-card:: 硬件

      Linux、NVIDIA GPU；Python 3.11；CUDA 与 GL/EGL/Vulkan。

任务
~~~~~~~~~~~~

C2R 使用干净场景中的成功轨迹作为经验，在随机场景中评测。两种任务配置的用途如下。

.. list-table::
   :header-rows: 1

   * - 配置
     - 场景
     - 用途
   * - ``demo_clean``
     - 干净场景。
     - 采集任务经验。
   * - ``demo_randomized``
     - 随机背景、杂物、光照和桌面高度。
     - C2R 评测：50 个任务，每个任务 5 个已验证 seed。

.. _robotwin-observation-action:

观测与动作
~~~~~~~~~~~~

下表区分规划器使用的工具、模型输入及环境的成功判定。

.. list-table::
   :header-rows: 1

   * - 项目
     - 说明
   * - 观测
     - 头部与左右腕部的 RGB 图像、用于定位的深度／世界坐标及机器人状态。LingBot-VLA 使用 RGB 图像、机器人状态和任务文字。
   * - 动作
     - 规划器调用 ``lingbot_act`` 或动作原语。环境支持 16 维末端动作（``ee``）和 14 维关节／夹爪动作（``qpos``）。
   * - 奖励与成功判定
     - 任务成功以原生 ``TASK_ENV.eval_success`` 为准，规划器结束与任务超时需分别判断。
   * - 任务指令
     - 评测表中的任务与 seed 组合在重置后使用对应的 ``task_language``；自定义 seed 使用原生环境生成的指令。

安装
----

RoboTwin 要求 Python 3.11。宿主机需预先具备兼容的 CUDA toolkit/NVCC、编译工具链以及 SAPIEN 依赖的系统级 GL/EGL/Vulkan 库。创建虚拟环境并安装 RoboTwin 所需依赖：

.. code-block:: bash

   git clone https://github.com/RLinf/RPent.git
   cd RPent
   uv venv --python 3.11 .venv-robotwin
   source .venv-robotwin/bin/activate
   uv pip install -e ".[robotwin]"

用户不需要运行 RLinf 安装器，也不需要单独克隆 RoboTwin。

国内网络可使用 PyPI 镜像加速：

.. code-block:: bash

   uv pip install -e ".[robotwin]" \
      --default-index https://mirrors.aliyun.com/pypi/simple \
      --index https://pypi.tuna.tsinghua.edu.cn/simple

.. note::

   ``.[robotwin]`` 使用 SAPIEN 3.0.0b1。其他版本可能改变仿真观测，导致模型效果下降。

.. note::

   ``.[robotwin]`` 的 RoboTwin 与 LingBot 运行时已作为发布包安装到 PyPI； cuRobo 仍从 GitHub 官方 tag 源码构建，因此即使配置了 PyPI 镜像，安装时仍需能访问 GitHub。

下载仿真资源
------------

下载 RPent 支持的 RoboTwin 仿真资源，并设置资源目录：

.. code-block:: bash

   robotwin-download-assets --output ~/.robotwin/assets
   export ROBOTWIN_ASSETS_PATH=~/.robotwin/assets
   # 国内用户可以使用下面的命令
   # HF_ENDPOINT=https://hf-mirror.com robotwin-download-assets --output ~/.robotwin/assets

下载工具会先校验已有文件；如果目标目录中的 RoboTwin 资源已经完整，则不会重复下载。

下载模型
--------

下载 LingBot 模型并设置模型目录：

.. code-block:: bash

   # 国内用户可设置 HF_ENDPOINT=https://hf-mirror.com
   hf download RLinf/LingBot-VLA-RoboTwin-EEF-ckpt1500 \
      --revision e727b46cd220b66981ea4d2fd9ba84adc189e2cc \
      --local-dir /path/to/LingBot-VLA-RoboTwin-EEF-ckpt1500
   export LINGBOT_MODEL_PATH=/path/to/LingBot-VLA-RoboTwin-EEF-ckpt1500

模型目录中已经包含 RoboTwin 的默认机器人配置。

运行一个任务
------------

先按 :doc:`configure_planner` 配置模型服务，并用 ``rpent-check-llm`` 检查连接。在已激活的虚拟环境中运行一个任务：

.. code-block:: bash

   # 国内用户可设置 HF_ENDPOINT=https://hf-mirror.com,
   # 因为下面的命令运行过程中会下载相关的memory数据
   rpent --robot robotwin \
      --task-name beat_block_hammer \
      --seed 100000 \
      --planner codex \
      --model gpt-5.5

修改 ``--task-name`` 可以选择其他任务；标准随机化评测使用的 seed 说明见下方。完整参数请运行 ``rpent --robot robotwin --help`` 查看。

.. note::

   ``--seed`` 是 RoboTwin 的精确场景随机种子。使用标准 ``demo_randomized`` 配置进行评测时，请从 `RoboTwin evaluation suite <https://github.com/RLinf/RPent/blob/main/robots/robotwin/eval/demo_randomized.json>`_ 中选择当前任务对应的 5 个已验证 seed。

   这些 seed 已通过 RoboTwin expert 执行筛选；无法稳定初始化或 expert 执行未成功的候选 seed 已被跳过。自定义运行仍可显式指定表中没有的其他 seed。

查看结果
--------

终端会显示服务启动信息、规划器输出和工具调用。默认情况下，运行结果保存在 ``logs/<timestamp>_robotwin_<task-name>_s<seed>/``。排查或复核运行结果时，可以先查看以下文件：

- ``run.log``：RPent 主进程日志。
- ``robotwin_env_server.log`` 和 ``lingbot_vla_server.log``：仿真环境与模型服务的启动和报错信息。
- ``transcript_*.json``：规划器对话和最终回复。

任务是否成功以最新工具结果中的 RoboTwin 原生 ``TASK_ENV.eval_success`` 为准。``finish`` 只负责结束规划器循环，不会另外定义一套成功条件。

添加 ``--dashboard`` 可以在浏览器中查看规划器输出以及头部和腕部相机画面。 Dashboard 启动后，访问地址会显示在终端中。

常用参数
--------

RPent 默认使用 RoboTwin 的 ``demo_randomized`` 任务配置，该配置带环境扰动（随机背景、桌面杂物、光照、桌高）。如需简单、干净的场景，可使用 ``--task-config demo_clean``。

- ``--robotwin-assets-path``：覆盖 ``ROBOTWIN_ASSETS_PATH`` 指定的资源目录。
- ``--vla-model-path``：覆盖 ``LINGBOT_MODEL_PATH`` 指定的模型目录。
- ``--cuda-device``：让仿真环境和 VLA 使用同一张 GPU。
- ``--env-cuda-device`` 和 ``--vla-cuda-device``：让仿真环境和 VLA 使用不同 GPU。这两个参数不能与 ``--cuda-device`` 同时使用。

规划器配置、外部服务和离线参考资料的说明分别见 :doc:`configure_planner`、 :doc:`advanced_deployment` 和 :doc:`memory`。

使用 HF 记忆模式评测时，RPent 会在运行前从公开数据集 `RLinf/RPent-memory <https://huggingface.co/datasets/RLinf/RPent-memory/tree/main/robotwin>`_ 同步可选的 RoboTwin 经验和任务参考。这些内容包含经过验证的操作方法，可以帮助规划器提高任务表现；即使无法下载，任务仍可正常启动。

任务记忆
--------

公开的 RoboTwin 记忆位于数据集的 ``robotwin/`` 目录，HF 模式会将其同步到 ``memory/robotwin/``。

``MEMORY.md`` 索引可跨任务复用的经验，例如感知线索、控制参数和失败恢复方法。规划器按当前任务与遇到的问题选择相关条目。

每个评测任务的 ``task_only/<task>_s0.json`` 描述阶段目标、可观察的完成条件、控制方式及已知失败模式。配套的 ``task_only/<task>_s0_recipe.jsonl`` 保存历史工具调用，供规划器参考动作顺序和 VLA 动作块的执行节奏。

文件名中的 ``_s0`` 是任务参考文件的统一命名，不代表场景 seed 0。实际来源 seed 经 RoboTwin 官方 expert 程序筛选，并保存在元数据中。

这些参考来自成功的 ``demo_clean`` 轨迹，用于指导独立的 ``demo_randomized`` 任务。可以参考操作阶段和控制方式；任务指令、机械臂选择、像素、坐标、姿态、避障空间及接触位置，都必须按当前任务和观测重新确认。

``evidence_status=supported`` 表示有成功的干净场景轨迹作为证据；``experimental`` 表示经验尚不充分。先阅读索引，再选择与本次任务有关的少量笔记。

探索模式
--------

添加 ``--explore`` 后，规划器可以复位环境、重新尝试任务，并将经验写入本地记忆。与 LIBERO 相同，每次运行默认最多包含 3 个规划会话，每个会话最多尝试 5 次：

.. code-block:: bash

   rpent --robot robotwin --task-name beat_block_hammer \
     --task-config demo_randomized --seed 100000 \
     --planner codex \
     --explore --explore-sessions 3 --explore-attempts-per-session 5 \
     --memory-dir /path/to/robotwin-memory

``reset`` 使用环境原生的复位机制，因此每次复位后，规划器都需要重新获取观测并定位。运行器只导出最后一次复位后成功尝试的动作序列。探索产生的记忆写入本地草稿目录（inbox）；运行正常结束、未发生智能体执行错误时，会自动合并草稿。传入 ``--no-auto-merge-memory`` 可关闭自动合并。

实验复现
--------

以下是 :doc:`Harness VLA <../awesome_works/harnessvla>` 在 RoboTwin C2R 上的评测结果。实验使用 `reproduce/robotwin <https://github.com/RLinf/RPent/tree/reproduce/robotwin>`_ 分支、``gpt-5.5`` 模型和 ``xhigh`` 推理强度：

- ``demo_randomized``：62.4%（156/250）

本次评测中，156 次运行成功，58 次任务失败，36 次运行超时。

评测覆盖 RoboTwin 的 50 个任务，每个任务运行 5 次，共计 250 次。每个任务使用的 5 个 seed 来自 ``robots/robotwin/eval/demo_randomized.json``，均经 RoboTwin 官方 expert 程序验证。由于不同任务的可解 seed 可能不同，请根据该文件为每个任务选择对应 seed，不要对所有任务统一使用一组固定 seed。对于表中列出的任务与 seed 组合，RPent 会按该 seed 复位到对应场景，再使用表内的 ``task_language`` 作为任务指令；未列出的自定义 seed 仍使用 RoboTwin 原生环境生成的指令。

单次运行的复现命令如下：

.. code-block:: bash

   rpent --robot robotwin \
     --task-name beat_block_hammer \
     --task-config demo_randomized \
     --seed 100000 \
     --planner codex \
     --model gpt-5.5 \
     --reasoning-effort xhigh \
     --max-turns 100 \
     --planner-timeout-s 4800 \
     --max-episode-steps 10000

本例运行 ``beat_block_hammer`` 的 seed 100000。复现全部成绩时，按 ``demo_randomized.json`` 为每个任务逐一运行其中列出的 seed。运行前还需按照本页前文配置 RoboTwin 仿真资源和 LingBot-VLA checkpoint。每次运行仅在 ``TASK_ENV.eval_success`` 的最终值为 ``true`` 时计为成功；规划器调用 ``finish`` 本身不代表成功。
