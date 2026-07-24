Agentic Planner
===============

RPent 通过一个 CLI 参数选择 Agentic Planner 的后端：

.. code-block:: bash

   --planner {api, claude_code, codex}

三种 planner 接收相同的系统提示词和用户提示词，也使用同一套 RPent 工具定义。
它们的区别在于如何将这些工具接入模型、如何组织工具调用循环，以及使用哪个模型
SDK。

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - ``--planner``
     - 它是什么
     - 什么时候选它
   * - ``api``
     - 基于 `Pydantic AI <https://pydantic.dev/docs/ai/>`_ 实现的工具调用循环，
       不绑定特定模型提供商。当前支持 Anthropic Messages API、OpenAI Responses
       API 和 OpenAI 兼容的 Chat Completions API，内置 prompt 缓存和历史图片剪枝。
     - 需要精细控制模型调用、支持更多模型提供商，或降低单轮调用成本。
   * - ``claude_code``
     - `Claude Agent SDK
       <https://code.claude.com/docs/en/agent-sdk/overview>`_。
       把 RPent 的 toolkit 暴露为进程内 MCP 服务，由 Claude Agent SDK
       驱动循环。
     - 想使用 Claude Code 原生提供的 agent 能力（memory、thinking-mode
       预算和更完善的工具重试机制）。
   * - ``codex``
     - OpenAI **Codex Python SDK**。RPent 在进程内启动
       Streamable HTTP MCP 服务，把 toolkit 接入 Codex。
     - 想使用 Codex 原生提供的 agent 能力，或者已有可用的 OpenAI
       或 Codex 配额。

``api`` planner（直接调用模型 API）
-------------------------------------

``--planner api`` 是默认选项。它使用 Pydantic AI 实现工具调用循环，并要求
``--model`` 带有模型提供商前缀。当前项目安装的依赖包含 Anthropic 和 OpenAI
集成，因此可以直接使用 Anthropic Messages API、OpenAI Responses API，
以及 OpenAI 兼容的 Chat Completions API。

通过 ``--model`` 前缀选择模型提供商：

.. code-block:: bash

   # Anthropic Claude
   rpent --planner api --model anthropic:claude-opus-4-8 ...

   # OpenAI Responses (例如 GPT-5.5)
   rpent --planner api --model openai:gpt-5.5 ...

   # OpenAI 兼容的 Chat Completions（例如 GLM 5.2，纯文本）
   rpent --planner api --model openai-chat:glm-5.2 --no-images ...

它读取以下环境变量；需要覆盖 API 地址时使用 ``--base-url``：

- ``anthropic:*`` → ``ANTHROPIC_BASE_URL`` / ``ANTHROPIC_API_KEY``
- ``openai:*`` / ``openai-chat:*`` → ``OPENAI_BASE_URL`` /
  ``OPENAI_API_KEY``

``api`` planner 的相关调节参数：

- ``--max-tokens`` —— 单次 LLM 回复的 token 上限（默认 ``8192``）。
- ``--max-turns`` —— 工具调用轮数上限（默认 ``100``）。
- ``--no-images`` —— 不向模型发送图片字节；纯文本模型必须加此参数。此时
  智能体只依赖文本状态推理，任务表现可能不够理想。

``claude_code`` planner
------------------------

``--planner claude_code`` 将工具调用循环交给 Claude Agent SDK。
RPent 通过 SDK 创建进程内 MCP 服务，并把 toolkit 的工具注册到
``mcp__rpent__<name>`` 命名空间。

.. code-block:: bash

   rpent --env libero --planner claude_code \
     --model claude-opus-4-8 \
     --suite libero_object_swap --task 2 --seed 0

注意事项：

- ``--model`` **不要** 加模型提供商前缀；省略时默认使用 ``sonnet``。
- ``--max-turns`` 会传给 Claude Agent SDK，默认 ``100``。
- 非交互运行受 ``--planner-timeout-s`` 限制；默认读取
  ``CELL_TIMEOUT_S``，未设置时为 ``1200`` 秒。``--interactive`` 模式
  不应用这一时限。
- 通过 ``--claude-code-max-budget-usd`` 设置美元预算（默认取
  ``MAX_BUDGET_USD`` 环境变量或 ``10``）。
- RPent 的依赖中已包含 Claude Agent SDK；该 SDK 自带 Claude Code
  二进制文件，无需单独安装 CLI。认证通常使用 ``ANTHROPIC_API_KEY``，详见
  `Claude Agent SDK 文档
  <https://code.claude.com/docs/en/agent-sdk/overview>`_。

``codex`` planner
------------------

``--planner codex`` 使用 OpenAI Codex Python SDK。每次运行时，RPent
会在当前进程的后台线程中启动本地 Streamable HTTP MCP 服务，Codex 通过
该服务调用同一个 toolkit；无需预先启动 ``scripts/codex_proxy/``。

.. code-block:: bash

   rpent --env libero --planner codex \
     --model gpt-5.5 \
     --suite libero_goal_task --task 1 --seed 0

注意事项：

- ``--model`` 会覆盖 ``CODEX_MODEL``；两者都未设置时使用 Codex SDK
  配置的默认模型。
- ``--planner-timeout-s`` 限制 Codex 运行时间。默认依次读取
  ``CODEX_TIMEOUT_S``、``CELL_TIMEOUT_S``，均未设置时为 ``1200`` 秒。
- 默认情况下，Codex SDK 会复用已有的 Codex 认证。若要接入自定义的
  Responses API 兼容端点，请设置 ``CODEX_BASE_URL`` 和
  ``CODEX_API_KEY``；这里不读取 ``OPENAI_BASE_URL`` 或
  ``OPENAI_API_KEY``。

接入自定义 planner
------------------

如果三种内置 planner 都不合适，例如需要接入内部 planner、研究原型或其他
agent SDK，可以实现 ``rpent.planner.base.Planner`` 协议，并在
``rpent.planner.base.build_planner`` 中增加对应的构造分支：

.. code-block:: python

   # rpent/planner/my_planner.py
   from rpent.planner.base import PlannerResult

   class MyPlanner:
       def solve(
           self,
           *,
           system_prompt,
           user_message,
           toolkit,
           max_turns,
           input_queue=None,
       ):
           tool_specs = toolkit.get_tools_spec()
           # 使用 system_prompt、user_message 和 tool_specs 调用模型。
           # 每次工具调用都通过下面的接口执行：
           tool_result = toolkit.execute_tool(tool_name, arguments)
           ...
           return PlannerResult(
               finish_result=finish_result,
               messages=messages,
               stats=stats,
               error=error,
           )

任何 planner 必须：

1. 接收已经渲染好的 ``system_prompt`` 和 ``user_message``。
2. 从 ``toolkit.get_tools_spec()`` 取得工具定义，并通过
   ``toolkit.execute_tool(name, arguments)`` 执行工具。
3. 将 ``ToolResult.content_blocks`` 中的文本和图片转换成模型 SDK
   所需的格式。
4. 识别 ``ToolResult.is_finish``，并按 ``max_turns`` 等限制终止循环。
5. 返回包含结束状态、消息、统计信息和可选错误的 ``PlannerResult``。

由于 RPent 工具定义和 prompt 渲染流程保持不变，新增 planner 不需要修改
工具或环境服务。接口参见
:doc:`../development/architecture`；想给
自定义 planner 暴露新工具，见 :doc:`../development/add_primitive`。

设置 planner 的运行限制
-----------------------

以下参数的作用范围并不相同：

- ``--max-tokens`` 只限制 ``api`` planner *每次回复* 的 token 数。
  LIBERO 类任务通常 ``8192`` 就够；更长时序的 RoboCasa episode
  如果模型支持可以调大。
- ``--max-turns`` 限制工具调用的总轮数。单个 LIBERO 任务通常
  不会超过 30 轮；RoboCasa 的长时序任务可能接近默认的 ``100``。
- ``--planner-timeout-s`` 限制 planner 的运行时间。

模型调用 ``finish`` 工具后，planner 会记录相应的结束状态。达到轮数上限或
超时时，运行结束，主程序仍会保存 transcript。超时或 SDK 异常会写入
planner 结果，并输出到日志。
