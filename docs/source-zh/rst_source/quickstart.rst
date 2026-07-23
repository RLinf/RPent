快速开始
========

本页是在 ``README.md`` Quick Start 基础上整理的快速入门指南。开始前，
请先按照 :doc:`installation` 克隆 RPent，并执行
``pip install -e ".[full]"``。

1. 配置 API key 与 checkpoint
------------------------------

导出 Anthropic 密钥, 以及 VLA checkpoint 的路径:

.. code-block:: bash

   # Anthropic 密钥; 使用官方端点时无需 export base url。
   export ANTHROPIC_BASE_URL=https://xxx
   export ANTHROPIC_API_KEY=sk-xxx

   # VLA checkpoint —— 从下面地址下载
   # https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT
   export PI05_CHECKPOINT_PATH=/path/to/rlinf-pi05-libero-130-fullshot-sft
   export LIBERO_TYPE=pro
   export CUDA_VISIBLE_DEVICES=0

2. 跑一个 LIBERO 任务
---------------------

使用 ``claude_code`` planner 跑单个 LIBERO PRO 任务
(``libero_object_swap``, 任务 ``2``, 种子 ``0``):

.. code-block:: bash

   rpent --env libero --suite libero_object_swap --task 2 --seed 0 \
     --planner claude_code --model claude-opus-4-8

其他 planner (``api``、``codex``) 与模型提供商的配置见
:doc:`usage/configure_planner`。

3. 用 dashboard 观察运行
------------------------

添加 ``--dashboard`` 后，会启动本地监控服务，并在终端输出访问地址。
打开该地址后，可先在启动页面确认配置；运行开始后，页面会实时显示智能体的
推理过程、相机与 Pi0 视图、动作时间线和片段回放。使用
``--dashboard-language zh-cn`` 可切换到中文界面。

.. code-block:: bash

   rpent --env libero --dashboard --dashboard-language zh-cn \
     --suite libero_goal_task --task 1 --seed 0 \
     --planner claude_code --model claude-opus-4-8

关键 CLI 选项
-------------

``rpent`` 日常最常用的几个 flag:

.. list-table::
   :header-rows: 1
   :widths: 22 15 63

   * - 参数
     - 默认值
     - 说明
   * - ``--env``
     - 必填
     - 环境后端。当前支持 ``libero``。
   * - ``--suite``
     - 必填
     - 任务套件, 如 ``libero_object_task``、``libero_spatial_swap``
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
     - 模型 ID; ``api`` 下要带 provider 前缀 (``anthropic:…``,
       ``openai:…``, ``openai-chat:…``)
   * - ``--max-turns``
     - ``100``
     - 智能体最大轮数
   * - ``--max-tokens``
     - ``8192``
     - LLM 每次回复的最大 token 数
   * - ``--no-images``
     - 关
     - 纯文本模式: 不向模型发送图片字节 (用于不支持图片输入的模型)
   * - ``--max-episode-steps``
     - ``10000``
     - 环境最大步数
   * - ``--libero-type``
     - ``LIBERO_TYPE`` 或 ``pro``
     - LIBERO 变体: ``standard`` | ``pro`` | ``plus``
   * - ``--cuda-device``
     - 继承
     - 暴露给 env / vla server 的 GPU 设备
   * - ``--dashboard``
     - 关
     - 为本次运行启动本地 dashboard
   * - ``--dashboard-language``
     - ``en``
     - Dashboard UI 语言: ``en`` | ``zh-cn``
   * - ``--env-endpoint``
     - —（自动启动）
     - 已在运行的 env_server 的 ``[protocol://]host:port``
       (``protocol=http|socket``, 默认 ``http``). 留空则本地起一个。
   * - ``--vla-endpoint``
     - —（自动启动）
     - 已在运行的 vla_server 的 ``[protocol://]host:port`` (同上).
       留空则本地起一个。

运行结果
--------

一次成功的运行会：

1. ``env_server`` / ``vla_server`` 启动后会分别打印一行
   ``RPC server listening on http://127.0.0.1:<port>``。
2. 智能体每一轮的推理过程会输出到终端，或实时传输到 Dashboard。
3. 当 LLM 调用 ``finish(status="success", summary="任务已完成")`` 时结束；
   或者达到 ``--max-turns`` / ``--max-episode-steps`` 时结束。
4. 生成 ``<output_dir>/transcript_*.json``（完整的逐轮记录）和
   ``<output_dir>/episode.mp4``（渲染得到的回合视频）。

出问题时, 参考 :doc:`installation` 页底部提到的三份日志文件。
