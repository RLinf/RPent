Agentic Planner
===============

RPent 的 reasoning brain —— 也叫 planner —— 用一个 CLI flag 选择:

.. code-block:: bash

   --planner {api, claude_code, codex}

三种 planner 看到的是同一份 tool schema 和同一份 prompt bundle。它们只在
tool-calling 循环 *如何* 被编排, 以及能触达哪些 LLM / SDK 上有区别。

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - ``--planner``
     - 它是什么
     - 什么时候选它
   * - ``api``
     - 基于 `pydantic-ai <https://ai.pydantic.dev/>`_ 的与 provider
       无关的 tool-calling 循环。支持 Anthropic、OpenAI Responses、
       OpenAI 兼容 chat 接口, 内置 prompt 缓存和历史图片剪枝。
     - 需要最细的调用控制、最广的 provider 覆盖, 或最省钱的
       per-turn 开销。
   * - ``claude_code``
     - `Claude Agent SDK
       <https://docs.claude.com/en/api/agent-sdk/overview>`_。
       把 RPent 的 toolkit 暴露为 in-process MCP server, 由 Claude
       驱动循环。
     - 想用 Claude 的原生 agent runtime (memory、thinking-mode
       预算、健壮的 tool 重试)。
   * - ``codex``
     - OpenAI **Codex SDK**, 通过 HTTP MCP server 桥接到 toolkit。
     - 想用 Codex 的 agent runtime, 或者已经有 OpenAI / Codex
       配额可用。

``api`` planner (自定义 / 轻量)
--------------------------------

``--planner api`` 跑一个手写的 pydantic-ai 循环。它是默认值, 也是
可移植性最好的一个 —— 任何讲 Anthropic Messages API、OpenAI Responses API,
或 OpenAI 兼容 chat API 的 provider 都能用。

通过 ``--model`` 前缀选择 provider:

.. code-block:: bash

   # Anthropic Claude
   rpent --planner api --model anthropic:claude-opus-4-8 ...

   # OpenAI Responses (例如 GPT-5.5)
   rpent --planner api --model openai:gpt-5.5 ...

   # OpenAI 兼容 chat (例如 GLM 5.2, 纯文本)
   rpent --planner api --model openai-chat:glm-5.2 --no-images ...

它读取的环境变量 (需要覆盖时用 ``--base-url``):

- ``anthropic:*`` → ``ANTHROPIC_BASE_URL`` / ``ANTHROPIC_API_KEY``
- ``openai:*`` / ``openai-chat:*`` → ``OPENAI_BASE_URL`` /
  ``OPENAI_API_KEY``

``api`` 专属的调节参数:

- ``--max-tokens`` —— 单次 LLM 回复的 token 上限 (默认 ``8192``)。
- ``--max-turns`` —— tool-calling 轮数上限 (默认 ``100``)。
- ``--no-images`` —— 不向模型发送图片字节; 纯文本模型必须加此参数,
  否则会报 ``400 "message type 'image_url' is not supported"``。此时
  智能体只依赖文本状态推理, 任务表现可能不够理想。

``claude_code`` planner
------------------------

``--planner claude_code`` 把循环委托给 Claude Agent SDK。RPent 的 tools
变成一个 **in-process MCP server**, Claude Code 直接调用; 它看到的工具名
带有 ``mcp__rpent__<name>`` 命名空间。

.. code-block:: bash

   rpent --planner claude_code \
     --model claude-opus-4-8 \
     --suite libero_object_swap --task 2 --seed 0

注意事项:

- ``--model`` **不要** 加 provider 前缀 —— 直接写 ``claude-opus-4-8``。
- 子进程有 wall-clock 上限 (``--planner-timeout-s``, 默认取
  ``CODEX_TIMEOUT_S`` / ``CELL_TIMEOUT_S`` / ``1200``)。
- 通过 ``--claude-code-max-budget-usd`` 设置美元预算 (默认取
  ``MAX_BUDGET_USD`` 环境变量或 ``10``)。
- Claude Code 需要单独安装和登录; 见
  `Claude Agent SDK 文档
  <https://docs.claude.com/en/api/agent-sdk/overview>`_。

``codex`` planner
------------------

``--planner codex`` 通过 ``scripts/codex_proxy/`` 起的 HTTP MCP server
把同一个 toolkit 桥接到 OpenAI Codex SDK。

.. code-block:: bash

   rpent --planner codex \
     --model gpt-5.5 \
     --suite libero_goal_task --task 1 --seed 0

注意事项:

- ``--planner-timeout-s`` 的语义与 ``claude_code`` 相同。
- Codex 用标准的 OpenAI 环境变量做认证。

自带 agent
----------

如果这三种 planner 都不合适 —— 例如想接入内部的 planner、实验性的
研究原型、或另一种 agent SDK —— 继承 ``rpent.planner.base.Planner``,
并在 ``rpent.planner.base.build_planner`` 中注册工厂:

.. code-block:: python

   # rpent/planner/mybrain.py
   from rpent.planner.base import Planner

   class MyPlanner(Planner):
       async def run(self, *, prompt_bundle, toolkit, output_dir, ...):
           # 自己驱动 tool-calling 循环。
           # 用 toolkit.dispatch(tool_name, **kwargs) 调工具。
           ...

任何 planner 必须:

1. 拿到渲染好的 ``prompt_bundle`` (来自
   ``robots/<env>/prompt_bundle.py`` 的 system + user 分节)。
2. 循环处理 LLM 回复、抽出 tool call、通过 ``toolkit.dispatch(...)``
   转发到 toolkit。
3. 把每个 tool 的返回值以 multimodal 上下文 (text + images) 的形式
   喂回 LLM。
4. 遇到 ``finish`` 或达到上限时终止。

因为所有 planner 看到的是同一份 schema 和 prompt, 新增 brain 不需要
改动 tool 或 env server。接口参见 :doc:`../development/architecture`;
想给自定义 brain 暴露新工具, 见 :doc:`../development/add_primitive`。

选择 max-tokens 与 max-turns
----------------------------

两个 knob 圈定每次 planner 运行的规模:

- ``--max-tokens`` 限制 *每次回复* 的 token 数。LIBERO 类任务通常
  ``8192`` 就够; 更长时序的 RoboCasa episode 如果模型支持可以调大。
- ``--max-turns`` 限制 *tool-calling 总轮数*。单个 LIBERO 任务通常
  不会超过 30 轮; RoboCasa 的长时序任务可能接近默认的 ``100``。

两个上限都会以 ``finish(stuck)`` 优雅收尾, 不会硬崩, 因此可以放心
调参 —— transcript 不会丢。
