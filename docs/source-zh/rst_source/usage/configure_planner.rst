Agentic Planner
===============

RPent 通过一个 CLI 参数选择 planner：

.. code-block:: bash

   --planner {api, claude_code, codex}

三种 planner 使用相同的工具 schema 和 prompt bundle，区别主要在于
tool-calling 循环的编排方式，以及各自支持的 LLM 和 SDK。

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - ``--planner``
     - 它是什么
     - 什么时候选它
   * - ``api``
     - 基于 `pydantic-ai <https://ai.pydantic.dev/>`_、不绑定特定
       provider 的 tool-calling 循环。支持 Anthropic、OpenAI Responses
       和 OpenAI 兼容 chat 接口，内置 prompt 缓存和历史图片剪枝。
     - 需要精细控制模型调用、支持更多 provider，或降低单轮调用成本。
   * - ``claude_code``
     - `Claude Agent SDK
       <https://docs.claude.com/en/api/agent-sdk/overview>`_。
       把 RPent 的 toolkit 暴露为 in-process MCP server, 由 Claude
       驱动循环。
     - 想使用 Claude 的原生 agent runtime（memory、thinking-mode
       预算和更完善的工具重试机制）。
   * - ``codex``
     - OpenAI **Codex SDK**, 通过 HTTP MCP server 桥接到 toolkit。
     - 想用 Codex 的 agent runtime, 或者已经有 OpenAI / Codex
       配额可用。

``api`` planner (自定义 / 轻量)
--------------------------------

``--planner api`` 使用基于 pydantic-ai 实现的 tool-calling 循环。它是默认
选项，也具有较好的可移植性：任何支持 Anthropic Messages API、OpenAI
Responses API 或 OpenAI 兼容 chat API 的 provider 都可以接入。

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

``--planner claude_code`` 将 tool-calling 循环交给 Claude Agent SDK。
RPent 将 toolkit 作为进程内 MCP server 提供给 Claude Code，其工具名带有
``mcp__rpent__<name>`` 命名空间。

.. code-block:: bash

   rpent --env libero --planner claude_code \
     --model claude-opus-4-8 \
     --suite libero_object_swap --task 2 --seed 0

注意事项:

- ``--model`` **不要** 加 provider 前缀 —— 直接写 ``claude-opus-4-8``。
- 子进程有最长运行时间限制（``--planner-timeout-s``，默认取
  ``CODEX_TIMEOUT_S`` / ``CELL_TIMEOUT_S`` / ``1200``）。
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

   rpent --env libero --planner codex \
     --model gpt-5.5 \
     --suite libero_goal_task --task 1 --seed 0

注意事项:

- ``--planner-timeout-s`` 的语义与 ``claude_code`` 相同。
- Codex 用标准的 OpenAI 环境变量做认证。

接入自定义 planner
------------------

如果三种内置 planner 都不合适，例如需要接入内部 planner、研究原型或其他
agent SDK，可以继承 ``rpent.planner.base.Planner``，并在
``rpent.planner.base.build_planner`` 中注册工厂：

.. code-block:: python

   # rpent/planner/my_planner.py
   from rpent.planner.base import Planner

   class MyPlanner(Planner):
       async def run(self, *, prompt_bundle, toolkit, output_dir, ...):
           # 自己驱动 tool-calling 循环。
           # 用 toolkit.dispatch(tool_name, **kwargs) 调工具。
           ...

任何 planner 必须:

1. 接收渲染后的 ``prompt_bundle``（来自
   ``robots/<env>/prompt_bundle.py`` 的 system + user 分节）。
2. 循环处理 LLM 回复、提取工具调用，并通过 ``toolkit.dispatch(...)``
   转发到 toolkit。
3. 将每个工具的返回值作为多模态上下文（text + images）返回给 LLM。
4. 遇到 ``finish`` 或达到上限时终止。

因为所有 planner 使用相同的 schema 和 prompt，新增 planner 不需要改动
tool 或 env server。接口参见 :doc:`../development/architecture`；想给
自定义 planner 暴露新工具，见 :doc:`../development/add_primitive`。

选择 max-tokens 与 max-turns
----------------------------

以下两个参数用于限制单次 planner 运行的规模：

- ``--max-tokens`` 限制 *每次回复* 的 token 数。LIBERO 类任务通常
  ``8192`` 就够; 更长时序的 RoboCasa episode 如果模型支持可以调大。
- ``--max-turns`` 限制 *tool-calling 总轮数*。单个 LIBERO 任务通常
  不会超过 30 轮; RoboCasa 的长时序任务可能接近默认的 ``100``。

达到任一上限时，运行都会以 ``finish(stuck)`` 正常结束，而不会直接中断，
因此 transcript 仍会完整保存。
