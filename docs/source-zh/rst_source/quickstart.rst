快速开始
========

本页把 ``README.md`` 里的 Quick Start 搬到了文档中。它假设你已经完成了
:doc:`installation` (克隆好 RPent 并执行了
``pip install -e ".[full]"``)。

1. 配置 API key 与 checkpoint
------------------------------

导出你要用的 LLM 提供商的 API key, 以及 VLA checkpoint 的路径:

.. code-block:: bash

   # LLM API keys (供 `api` planner 通过 pydantic-ai 使用)
   export ANTHROPIC_BASE_URL=https://xxx
   export ANTHROPIC_API_KEY=sk-xxx
   export OPENAI_BASE_URL=https://xxx
   export OPENAI_API_KEY=sk-xxx

   # VLA checkpoint —— 从下面地址下载
   # https://huggingface.co/datasets/RLinf/rlinf-pi05-libero-130-fullshot-sft
   export PI05_CHECKPOINT_PATH=/path/to/rlinf-pi05-libero-130-fullshot-sft
   export LIBERO_TYPE=pro
   export CUDA_VISIBLE_DEVICES=0

你只需要为实际使用的 provider 设置对应的 key。例如, 只用
``--planner claude_code`` 时, 可以不配置 ``OPENAI_*``。

2. 跑一个 LIBERO 任务
---------------------

用 ``api`` planner + Anthropic 模型, 上限 8192 tokens, 跑单个 LIBERO PRO
任务 (``libero_object_swap``, 任务 ``2``, 种子 ``0``):

.. code-block:: bash

   rpent --env libero --suite libero_object_swap --task 2 --seed 0 \
     --planner api --model anthropic:claude-opus-4-8 --max-tokens 8192

**模型 id 规约。** ``api`` planner 下, ``--model`` 需要带 provider
前缀; ``claude_code`` / ``codex`` 下, 直接写裸模型名:

- OpenAI 兼容 chat 接口 —— ``--model openai-chat:glm-5.2``
- OpenAI Responses 接口 —— ``--model openai:gpt-5.5``
- ``claude_code`` / ``codex`` —— 不加前缀, 如
  ``--model claude-opus-4-8``

完整的 brain 切换指南见 :doc:`usage/configure_planner`。

3. 用 dashboard 观察运行
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
   * - ``--max-episode-steps``
     - ``10000``
     - Env 最大 step 数
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
     - —(自动 spawn)
     - 已在运行的 env_server 的 ``[protocol://]host:port``
       (``protocol=http|socket``, 默认 ``http``). 留空则本地起一个。
   * - ``--vla-endpoint``
     - —(自动 spawn)
     - 已在运行的 vla_server 的 ``[protocol://]host:port`` (同上).
       留空则本地起一个。

跑起来后应该看到什么
--------------------

一次成功的运行:

1. env_server / vla_server 起来后各打印一行
   ``RPC server listening on http://127.0.0.1:<port>``。
2. 每一轮 agent 的 reasoning 会输出到终端 (或 stream 到 dashboard)。
3. 当 LLM 调用 ``finish(success=True)`` 时结束; 或者触达
   ``--max-turns`` / ``--max-episode-steps`` 时结束。
4. 写出 ``<output_dir>/transcript_*.json`` (完整 turn-by-turn 记录) 和
   ``<output_dir>/episode.mp4`` (渲染出的 rollout)。

出问题时, 参考 :doc:`installation` 页底部提到的三份日志文件。
