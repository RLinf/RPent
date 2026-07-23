快速开始
========

本页把 ``README.md`` 里的 Quick Start 搬到了文档中。它假设你已经完成了
:doc:`installation` (克隆好 RPent 并执行了
``pip install -e ".[full]"``)。

1. 配置 API key 与 checkpoint
------------------------------

导出 Anthropic 密钥, 以及 VLA 与 SAM3 checkpoint 的路径:

.. code-block:: bash

   # Anthropic 密钥; 使用官方端点时无需 export base url。
   export ANTHROPIC_BASE_URL=https://xxx
   export ANTHROPIC_API_KEY=sk-xxx

   # VLA checkpoint —— 从下面地址下载
   # https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT
   export PI05_CHECKPOINT_PATH=/path/to/rlinf-pi05-libero-130-fullshot-sft
   # SAM 3.0 checkpoint —— 从以下任一地址下载
   # https://huggingface.co/facebook/sam3
   # https://modelscope.cn/models/facebook/sam3
   export SAM3_CHECKPOINT_PATH=/path/to/sam3/sam3.pt
   export LIBERO_TYPE=pro
   export CUDA_VISIBLE_DEVICES=0

2. 跑一个 LIBERO 任务
---------------------

用 ``claude_code`` planner 跑单个 LIBERO PRO 任务
(``libero_object_swap``, 任务 ``2``, 种子 ``0``):

.. code-block:: bash

   rpent --env libero --suite libero_object_swap --task 2 --seed 0 \
     --planner claude_code --model claude-opus-4-8

其他 planner (``api``、``codex``) 与模型提供商的配置见
:doc:`usage/configure_planner`。

1. 用 dashboard 观察运行
------------------------

加上 ``--dashboard`` 会打开浏览器监控页面。它会先展示一个 launcher
页面让你确认配置, 然后开始 streaming: agent 的 reasoning、实时相机
与 Pi0 视图、动作时间线、剪辑回放。加上 ``--dashboard-language zh-cn``
切换到中文 UI。

.. code-block:: bash

   rpent --env libero --dashboard --dashboard-language zh-cn \
     --suite libero_goal_task --task 1 --seed 0 --planner claude_code

关键 CLI 选项
-------------

``rpent`` 日常最常用的几个 flag:

.. list-table::
   :header-rows: 1
   :widths: 22 15 63

   * - Flag
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
     - 套件内的任务 id
   * - ``--seed``
     - ``0``
     - 随机种子
   * - ``--planner``
     - ``api``
     - Reasoning brain: ``api`` | ``claude_code`` | ``codex``
   * - ``--model``
     - —
     - 模型 id; ``api`` 下要带 provider 前缀 (``anthropic:…``,
       ``openai:…``, ``openai-chat:…``)
   * - ``--max-turns``
     - ``100``
     - Agent 最大轮数
   * - ``--max-tokens``
     - ``8192``
     - LLM 每次回复的最大 token 数
   * - ``--no-images``
     - 关
     - 纯文本模式: 不向模型发送图片字节 (用于不支持图片输入的模型)
   * - ``--max-episode-steps``
     - ``10000``
     - Env 最大 step 数
   * - ``--libero-type``
     - ``LIBERO_TYPE`` 或 ``pro``
     - LIBERO 变体: ``standard`` | ``pro`` | ``plus``
   * - ``--cuda-device``
     - 继承
     - 暴露给 env / VLA / SAM3 server 的 GPU 设备
   * - ``--dashboard``
     - 关
     - 为本次运行启动本地 dashboard
   * - ``--dashboard-language``
     - ``en``
     - Dashboard UI 语言: ``en`` | ``zh-cn``
   * - ``--env-endpoint``
     - —(自动 spawn)
     - 已在运行的 env_server 的 ``[protocol://]host:port``
       (``protocol=http|socket``, 默认 ``http``). 留空则本地起一个。
   * - ``--vla-endpoint``
     - —(自动 spawn)
     - 已在运行的 vla_server 的 ``[protocol://]host:port``
       (协议规则同 ``env_server``)。留空则本地起一个。
   * - ``--sam3-endpoint``
     - —(自动 spawn)
     - 已在运行的 RPent SAM3 服务的 ``[protocol://]host:port``
       (协议规则同 ``env_server``)。留空则本地起一个。

跑起来后应该看到什么
--------------------

一次成功的运行:

1. env_server、vla_server 和 sam3_server 就绪后，会在各自的服务日志中
   打印一行 ``RPC server listening on http://127.0.0.1:<port>``；
   主进程确认三个服务就绪后再进入 agent loop。
2. 每一轮 agent 的 reasoning 会输出到终端 (或 stream 到 dashboard)。
3. 当 LLM 调用 ``finish(success=True)`` 时结束; 或者触达
   ``--max-turns`` / ``--max-episode-steps`` 时结束。
4. 写出 ``<output_dir>/transcript_*.json`` (完整 turn-by-turn 记录) 和
   ``<output_dir>/episode.mp4`` (渲染出的 rollout)。

出问题时, 参考 :doc:`installation` 页底部提到的服务与 agent 日志文件。
