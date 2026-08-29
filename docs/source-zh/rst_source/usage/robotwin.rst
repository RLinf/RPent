RoboTwin
========

`RoboTwin <https://robotwin-platform.github.io/>`_ 是一个面向双臂机器人操作的
仿真基准，包含多种桌面操作任务和随机化场景。RPent 通过 RLinf 运行 RoboTwin，
并使用 LingBot-VLA 生成机器人动作。

.. note::

   当前代码尚未完成 RoboTwin 的完整效果对齐验证，完整验证结果将在后续放出。

安装
----

RoboTwin 要求 Python 3.11。宿主机需预先具备兼容的 CUDA toolkit/NVCC、编译
工具链以及 SAPIEN 依赖的系统级 GL/EGL/Vulkan 库。创建虚拟环境并安装
RoboTwin 所需依赖：

.. code-block:: bash

   cd /path/to/RPent
   uv venv --python 3.11
   source .venv/bin/activate
   uv pip install -e ".[robotwin]"

用户不需要运行 RLinf 安装器，也不需要单独克隆 RoboTwin。

国内网络可使用 PyPI 镜像加速：\

.. code-block:: bash

   uv pip install -e ".[robotwin]" \
      --default-index https://mirrors.aliyun.com/pypi/simple \
      --index https://pypi.tuna.tsinghua.edu.cn/simple

.. note::

   ``.[robotwin]`` 使用 SAPIEN 3.0.0b1。其他版本可能改变仿真观测，导致模型
   效果下降。

.. note::

   ``.[robotwin]`` 目前会从 GitHub 的固定提交安装部分依赖，因此即使配置了
   PyPI 镜像，安装时仍需能访问 GitHub。这些依赖在正式发布后将改用版本号
   安装。

下载仿真资源
------------

下载 RPent 支持的 RoboTwin 仿真资源，并设置资源目录：

.. code-block:: bash

   robotwin-download-assets --output ~/.robotwin/assets
   export ROBOTWIN_ASSETS_PATH=~/.robotwin/assets
   # 国内用户可以使用下面的命令
   # HF_ENDPOINT=https://hf-mirror.com robotwin-download-assets --output ~/.robotwin/assets

下载工具会先校验已有文件；如果目标目录中的 RoboTwin 资源已经完整，
则不会重复下载。

下载模型
--------

下载 LingBot 模型并设置模型目录：

.. code-block:: bash

   # 国内用户在可以加上 HF_ENDPOINT=https://hf-mirror.com
   hf download RLinf/LingBot-VLA-RoboTwin-EEF-ckpt1500 \
      --revision e727b46cd220b66981ea4d2fd9ba84adc189e2cc \
      --local-dir /path/to/LingBot-VLA-RoboTwin-EEF-ckpt1500
   export LINGBOT_MODEL_PATH=/path/to/LingBot-VLA-RoboTwin-EEF-ckpt1500

模型目录中已经包含 RoboTwin 的默认机器人配置。

运行任务
--------

激活虚拟环境后运行一个任务：

.. code-block:: bash

   # 国内用户在可以加上 HF_ENDPOINT=https://hf-mirror.com,
   # 因为下面的命令运行过程中会下载相关的memory数据
   rpent --robot robotwin \
      --task-name beat_block_hammer \
      --seed 100000 \
      --planner codex \
      --model gpt-5.5

修改 ``--task-name`` 可以选择其他任务；标准随机化评测使用的 seed 说明见下方。
完整参数请运行 ``rpent --robot robotwin --help`` 查看。

.. note::

   ``--seed`` 是 RoboTwin 的精确场景随机种子。使用标准 ``demo_randomized``
   配置进行评测时，请从
   `RoboTwin evaluation suite
   <https://github.com/RLinf/RPent/blob/main/robots/robotwin/eval/demo_randomized.json>`_
   中选择当前任务对应的 5 个已验证 seed。

   这些 seed 已通过 RoboTwin expert 执行筛选；无法稳定初始化或 expert 执行
   未成功的候选 seed 已被跳过。自定义运行仍可显式指定表中没有的其他 seed。

查看运行结果
------------

终端会显示服务启动信息、规划器输出和工具调用。默认情况下，运行结果保存在
``logs/<timestamp>_robotwin_<task-name>_s<seed>/``。排查或复核运行结果时，
可以先查看以下文件：

- ``run.log``：RPent 主进程日志。
- ``robotwin_env_server.log`` 和 ``lingbot_vla_server.log``：仿真环境与模型
  服务的启动和报错信息。
- ``transcript_*.json``：规划器对话和最终回复。

任务是否成功以最新工具结果中的 RoboTwin 原生
``TASK_ENV.eval_success`` 为准。``finish`` 只负责结束规划器循环，不会另外
定义一套成功条件。

添加 ``--dashboard`` 可以在浏览器中查看规划器输出以及头部和腕部相机画面。
Dashboard 启动后，访问地址会显示在终端中。

常用参数
--------

RPent 默认使用 RoboTwin 的 ``demo_randomized`` 任务配置，该配置带环境
扰动（随机背景、桌面杂物、光照、桌高）。如需简单、干净的场景，可使用
``--task-config demo_clean``。

- ``--robotwin-assets-path``：覆盖 ``ROBOTWIN_ASSETS_PATH`` 指定的资源目录。
- ``--vla-model-path``：覆盖 ``LINGBOT_MODEL_PATH`` 指定的模型目录。
- ``--cuda-device``：让仿真环境和 VLA 使用同一张 GPU。
- ``--env-cuda-device`` 和 ``--vla-cuda-device``：让仿真环境和 VLA 使用不同
  GPU。这两个参数不能与 ``--cuda-device`` 同时使用。

规划器配置、外部服务和离线参考资料的说明分别见 :doc:`configure_planner`、
:doc:`advanced_deployment` 和 :doc:`../development/memory`。

每次运行前，RPent 会自动从公开数据集 `RLinf/RPent-memory
<https://huggingface.co/datasets/RLinf/RPent-memory/tree/main/robotwin>`_ 同步可选的
RoboTwin 经验和任务参考。这些内容包含经过验证的操作方法，可以帮助规划器提高任务表现；
即使无法下载，任务仍可正常启动。

规划器 memory 与 recipe
------------------------

只读规划资源位于数据集的 ``robotwin/`` 目录，并会同步到
``<RPent-clone-path>/resources/robotwin/``。

``memory/MEMORY.md`` 索引可跨任务复用的执行经验，包括感知线索、控制启发、恢复策略、
参数选择建议和常见失败模式。规划器可以通过该索引，只读取与当前任务或已观察到的失败
相关的 memory 条目。

对于每个评测任务，``recipe/<task>_s0.json`` 是从成功轨迹中提炼的语义 recipe，
描述阶段目标、可观察的完成 gate、控制与 VLA 使用建议以及已知失败模式。配套的
``recipe/recipe_<task>_s0.jsonl`` 记录该轨迹中的历史工具调用，用于提供动作顺序、
工具选择和 action chunk 节奏方面的证据。

文件名中的 ``_s0`` 只是统一的 recipe slot 名称，便于 prompt 查找，并不表示 RoboTwin
seed 0。由于部分随机 seed 可能不可解，每份 recipe 的来源 seed 通过 RoboTwin 官方
expert 程序选择；实际来源 seed 记录在 recipe 元数据中。

这些 recipe 来源于成功的 ``demo_clean`` 轨迹，用作独立 ``demo_randomized`` 场景的
策略先验。可迁移的是阶段结构、可观察 gate、控制方式和 VLA chunk 节奏；来源任务语言、
机械臂选择、像素、坐标、姿态、净空与接触点均不是新 episode 的直接命令。当前环境原生
task language 与最新 observation 始终优先，所有几何信息都必须重新定位。

``evidence_status=supported`` 表示 recipe 有成功 clean 轨迹支持；``experimental``
仍然只是弱先验。使用时先阅读 ``memory/MEMORY.md``，再只选与当前任务和失败模式相关的
少量笔记。

结果复现
--------

以下结果复现了 Harness VLA 在 RoboTwin C2R 上的评测。实验使用 ``gpt-5.5`` 模型和
``xhigh`` 推理强度：

- ``demo_randomized``：58.0%（145/250）

评测覆盖 RoboTwin 的 50 个任务，每个任务运行 5 个 episode，共计 250 个 episode。
每个任务使用的 5 个 seed 来自 ``robots/robotwin/eval/demo_randomized.json`` 中的
官方 verified expert seeds。由于不同任务的可解 seed 可能不同，请根据该文件为每个
任务选择对应 seed，不要对所有任务统一使用一组固定 seed。

单个 episode 的复现命令如下：

.. code-block:: bash

   rpent --robot robotwin \
     --task-name "task" \
     --task-config demo_randomized \
     --seed "seed" \
     --planner codex \
     --model gpt-5.5 \
     --reasoning-effort xhigh \
     --max-turns 100 \
     --planner-timeout-s 3600 \
     --max-episode-steps 10000

其中，``task`` 应替换为 ``demo_randomized.json`` 中的任务名，``seed`` 应替换为
该任务对应的一个 verified expert seed。运行前还需按照本页前文配置 RoboTwin assets、
LingBot-VLA checkpoint。任务是否成功以 episode 结束时最新的
``TASK_ENV.eval_success`` 为准，不能仅根据规划器是否调用 ``finish`` 判断。
