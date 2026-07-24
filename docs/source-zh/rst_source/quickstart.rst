快速开始
========

开始前，请先按照 :doc:`installation` 安装 RPent，并下载
LIBERO-PRO 仿真资源。下面以 LIBERO-PRO 和 ``claude_code`` planner
为例，演示如何完成一次运行。

1. 配置 API key 与 checkpoint
------------------------------

设置 Anthropic API key、VLA checkpoint 和 SAM 3.0 checkpoint 路径：

.. code-block:: bash

   # Anthropic 密钥；使用 Anthropic 官方 API 时无需设置 base URL。
   export ANTHROPIC_BASE_URL=https://xxx
   export ANTHROPIC_API_KEY=sk-xxx

   # VLA checkpoint —— 从下面地址下载
   # https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT
   export PI05_CHECKPOINT_PATH=/path/to/rlinf-pi05-libero-130-fullshot-sft

   # SAM 3.0 checkpoint —— 从以下任一地址下载
   # https://huggingface.co/facebook/sam3
   # https://modelscope.cn/models/facebook/sam3
   export SAM3_CHECKPOINT_PATH=/path/to/sam3/sam3.pt

2. 跑一个 LIBERO 任务
---------------------

使用 ``claude_code`` planner 跑单个 LIBERO PRO 任务
（``libero_object_swap``，任务 ``2``，种子 ``0``）：

.. code-block:: bash

   rpent --env libero --suite libero_object_swap --task 2 --seed 0 \
     --planner claude_code --model claude-opus-4-8

若要切换到其他 planner（如 ``codex`` 或 ``api``），请参阅
:doc:`Agentic Planner <usage/configure_planner>`。

3. 通过 Dashboard 查看运行过程
------------------------------

添加 ``--dashboard`` 后，RPent 会启动本地 Dashboard 服务，并在终端输出访问地址。打开该地址后，可以先在启动页面确认配置。运行开始后，Dashboard 会实时显示智能体的推理过程、相机与 Pi0 视图、动作时间线和片段回放。使用 ``--dashboard-language zh-cn`` 可切换到中文界面。

.. code-block:: bash

   rpent --env libero --dashboard --dashboard-language zh-cn \
     --suite libero_object_swap --task 2 --seed 0 \
     --planner claude_code --model claude-opus-4-8

关键 CLI 选项
-------------

下表只列出完成首次运行需要关注的选项。其他通用选项可运行
``rpent --help`` 查看；有关 LIBERO 环境的更多配置，请参阅
:doc:`LIBERO 使用指南 <usage/libero>`。

.. list-table::
   :header-rows: 1
   :widths: 22 15 63

   * - 参数
     - 默认值
     - 说明
   * - ``--env``
     - 必填
     - 环境后端，如 ``libero``
   * - ``--suite``
     - 必填
     - 任务套件，如 ``libero_object_task``、``libero_spatial_swap``
   * - ``--task``
     - 必填
     - 套件内的任务编号
   * - ``--seed``
     - ``0``
     - 随机种子
   * - ``--planner``
     - ``api``
     - ``api`` | ``claude_code`` | ``codex``
   * - ``--model``
     - —
     - 模型 ID；``api`` planner 需要模型提供商前缀
       （``anthropic:…``、``openai:…``、``openai-chat:…``）
   * - ``--dashboard``
     - 关
     - 为本次运行启动本地 Dashboard 服务

运行结果
--------

一次成功的运行会：

1. 终端会先显示 ``env_server``、``vla_server`` 和 ``sam3_server`` 的启动信息。
2. 智能体的逐轮输出和工具调用会显示在终端中；运行结束时还会显示耗时、token 用量和运行记录的路径。
3. 启用 Dashboard 后，智能体的输出、相机视图、动作时间线和片段回放也会实时显示在 Dashboard 中。
4. 默认输出目录为 ``logs/<timestamp>_<suite>_t<task>_s<seed>/``，其中包含 ``transcript_*.json``\ （运行记录）、``states.json``\ （每个环境步的记录）、``recipe_*.jsonl``\ （动作序列）和 ``episode.mp4``\ （回合录像）。

运行结束后，查看 ``states.json`` 的最后一条记录：``libero_terminated`` 为 ``true`` 表示 LIBERO 已判定任务完成；也可以打开 ``episode.mp4`` 复核运行过程。
出问题时，参考 :doc:`installation` 页底部提到的四份日志文件。
