配置规划器与模型服务
==============================

RPent 通过一个 CLI 参数选择 Agentic Planner 的后端：

.. code-block:: text

   --planner {api,claude_code,codex,flash}

三种在线规划器（``api``、``claude_code`` 和 ``codex``）接收相同的系统提示词和用户提示词，也使用同一套 RPent 工具定义。它们的区别在于如何将这些工具接入模型、如何组织工具调用循环，以及使用哪个模型 SDK。

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - ``--planner``
     - 它是什么
     - 什么时候选它
   * - ``api``
     - 基于 `Pydantic AI <https://ai.pydantic.dev/>`_ 实现的工具调用循环，
       不绑定特定模型提供商。当前支持 Anthropic Messages API、OpenAI Responses API 和 OpenAI 兼容的 Chat Completions API，内置 prompt 缓存和历史图片剪枝。
     - 直接调用 Anthropic、OpenAI 或兼容的模型服务。
   * - ``claude_code``
     - `Claude Agent SDK
       <https://code.claude.com/docs/en/agent-sdk/overview>`_。把 RPent 的 toolkit 暴露为进程内 MCP 服务，由 Claude Agent SDK 驱动循环。
     - 通过 Claude Agent SDK 运行工具调用循环。
   * - ``codex``
     - OpenAI **Codex Python SDK**。RPent 在进程内启动
       Streamable HTTP MCP 服务，把 toolkit 接入 Codex。
     - 通过 Codex SDK 运行任务，复用已有的 Codex 认证或配置独立 API。
   * - ``flash``
     - **Flash Mode**，仅用于评测。重放 memory 中保存的成功执行计划，并对每个路点
       的锚点重新定位，使方案能跟随移动过的物体。参见
       :doc:`flash`。
     - 想在新布局上低成本地重跑一个已知可行的方案，无需 LLM 在线规划；仍需要感知和 VLA 服务。

``api`` 规划器（直接调用模型 API）
-------------------------------------

``--planner api`` 是默认选项。它使用 Pydantic AI 实现工具调用循环，并要求 ``--model`` 带有模型提供商前缀。当前项目安装的依赖包含 Anthropic 和 OpenAI 集成，因此可以直接使用 Anthropic Messages API、OpenAI Responses API，以及 OpenAI 兼容的 Chat Completions API。

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

``api`` 规划器的相关参数：

- ``--max-tokens`` —— 单次 LLM 回复的 token 上限（默认 ``8192``）。
- ``--max-turns`` —— 工具调用轮数上限（默认 ``100``）。
- ``--no-images`` —— 不向模型发送图片字节；纯文本模型必须加此参数。此时智能体只依赖文本状态推理，任务表现可能不够理想。

.. _planner-claude-code:

``claude_code`` 规划器
------------------------

``--planner claude_code`` 将工具调用循环交给 Claude Agent SDK。 RPent 通过 SDK 创建进程内 MCP 服务，并把 toolkit 的工具注册到 ``mcp__rpent__<name>`` 命名空间。

RPent 为 Claude 规划会话关闭文件系统配置来源，因此不会自动加载项目的 ``CLAUDE.md`` 和开发 skills。工作目录仍为仓库根目录。

.. code-block:: bash

   rpent --robot libero --planner claude_code \
     --model claude-opus-4-8 \
     --suite libero_object_swap --task 2 --seed 0

注意事项：

- ``--model`` **不要** 加模型提供商前缀；省略时默认使用 ``sonnet``。
- ``--max-turns`` 会传给 Claude Agent SDK，默认 ``100``。
- 非交互运行受 ``--planner-timeout-s`` 限制；默认读取 ``CELL_TIMEOUT_S``，未设置时为 ``1200`` 秒。``--interactive`` 模式不应用这一时限。
- 通过 ``--claude-code-max-budget-usd`` 设置美元预算（默认取 ``MAX_BUDGET_USD`` 环境变量或 ``10``）。
- RPent 的依赖中已包含 Claude Agent SDK；该 SDK 自带 Claude Code 二进制文件，无需单独安装 CLI。认证通常使用 ``ANTHROPIC_API_KEY``，详见 `Claude Agent SDK 文档 <https://code.claude.com/docs/en/agent-sdk/overview>`_。

通过 Claude Code 使用本地模型
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Claude Code 可以连接兼容 Anthropic Messages API 的本地模型服务。假设服务将 Qwen3.6-27B 注册为 ``Qwen/Qwen3.6-27B``，可以这样配置：

.. code-block:: bash

   export ANTHROPIC_BASE_URL=http://127.0.0.1:8000
   export ANTHROPIC_API_KEY=EMPTY

   rpent --robot libero --planner claude_code \
     --model Qwen/Qwen3.6-27B \
     --suite libero_goal_task --task 1 --seed 0

对于无法识别的本地模型名称，Claude Code 默认按 200,000 token 的上下文窗口管理会话。如果本地服务使用其他长度，请参考 `Claude Code 环境变量文档 <https://code.claude.com/docs/en/env-vars>`_ 配置它的上下文和自动压缩参数。

.. _planner-codex:

``codex`` planner
------------------

``--planner codex`` 使用 OpenAI Codex Python SDK。每次运行时，RPent 会在当前进程的后台线程中启动本地 Streamable HTTP MCP 服务，Codex 通过该服务调用同一个 toolkit；无需预先启动 ``scripts/codex_proxy/``。

Codex 规划会话不会自动加载仓库的 ``AGENTS.md`` 和 ``.agents/skills/`` 中的开发 skills。工作目录仍为仓库根目录，机器人指南和 memory 仍可通过已有工具读取。

.. code-block:: bash

   rpent --robot libero --planner codex \
     --model gpt-5.5 \
     --suite libero_goal_task --task 1 --seed 0

注意事项：

- 设置 ``CODEX_SERVICE_TIER=fast`` 可向 Codex 后端传入 fast 服务档位，不改变 ``--reasoning-effort``。未设置时 RPent 不覆盖服务档位。
- ``--model`` 会覆盖 ``CODEX_MODEL``；两者都未设置时使用 Codex SDK 配置的默认模型。
- ``--planner-timeout-s`` 限制 Codex 运行时间。默认依次读取 ``CODEX_TIMEOUT_S``、``CELL_TIMEOUT_S``，均未设置时为 ``1200`` 秒。
- 默认情况下，Codex SDK 会复用已有的 Codex 认证。若要接入自定义的 Responses API 兼容端点，请设置 ``CODEX_BASE_URL`` 和 ``CODEX_API_KEY``；这里不读取 ``OPENAI_BASE_URL`` 或 ``OPENAI_API_KEY``。

通过 Codex 使用本地模型
~~~~~~~~~~~~~~~~~~~~~~~~

Codex 可以连接兼容 OpenAI Responses API 的本地模型服务。下面以通过 vLLM 启动 Qwen3.6-27B 为例：

.. code-block:: bash

   vllm serve /path/to/Qwen3.6-27B \
     --served-model-name Qwen/Qwen3.6-27B \
     --max-model-len 262144 \
     --reasoning-parser qwen3 \
     --enable-auto-tool-choice \
     --tool-call-parser qwen3_coder

然后让 Codex 连接本地服务，并填写该服务实际开放的上下文限制：

.. code-block:: bash

   export CODEX_BASE_URL=http://127.0.0.1:8000
   export CODEX_API_KEY=EMPTY
   export CODEX_MODEL_CONTEXT_WINDOW=262144
   export CODEX_AUTO_COMPACT_TOKEN_LIMIT=230000

   rpent --robot libero --planner codex \
     --model Qwen/Qwen3.6-27B \
     --suite libero_goal_task --task 1 --seed 0

vLLM 在兼容 OpenAI 的 ``/v1/models`` 响应中用 ``max_model_len`` 表示该上限，而 Codex 使用的模型目录格式要求 ``context_window`` 字段。因此，Codex 无法识别 vLLM 返回的模型元数据时会使用备用配置。请将 ``CODEX_MODEL_CONTEXT_WINDOW`` 设置为当前 vLLM 服务的 ``--max-model-len``。这是服务实际接受的上限；为了适应可用显存，它可以低于 checkpoint 配置中标注的最大长度。

``CODEX_AUTO_COMPACT_TOKEN_LIMIT`` 用于设置 Codex 自动压缩会话历史的触发点。该值应小于 ``CODEX_MODEL_CONTEXT_WINDOW``，为下一次回复预留空间；当服务窗口为 ``262144`` token 时，``230000`` 是一个示例值。这两个变量都是可选的；如果未设置，Codex 将使用自身的默认值。

RPent 的 ``--model`` 必须与 vLLM 的 ``--served-model-name`` 保持一致。使用其他模型时，请按照对应的 vLLM 部署说明设置解析参数。

.. _planner-check:

验证你的配置
------------

在启动完整任务前，先用 ``rpent-check-llm`` 检查模型服务的连接与认证配置。它会向所选后端发送其支持的最小真实请求，不携带工具或图像，也不启动机器人运行环境：

.. code-block:: bash

   rpent-check-llm --planner api --model anthropic:claude-opus-4-8
   rpent-check-llm --planner claude_code
   rpent-check-llm --planner codex --json

成功时退出码为 ``0``，任何失败为 ``1``，并将失败归类为 ``missing_config``、``unsupported_provider``、``missing_api_key``、 ``auth_failed``、``network_error``、``provider_error``、``sdk_error`` 之一。脚本与 CI 建议使用 ``--json``。``--base-url`` 覆盖后端端点， ``--timeout-s`` 覆盖诊断超时（``api`` 为 30 秒，两个 SDK 后端为 90 秒；运行时的 ``1200`` 秒默认值不会被复用）。

使用 Dashboard 时，也请先在终端运行上述检查，并使用准备运行任务的规划器与模型配置。Dashboard 从命令行接收配置，打开后直接显示运行监控页面。启动方法见 :doc:`dashboard`。

检查通过只能证明认证与网络可达。它并不能证明模型会接受图像块（参见 ``--no-images``）、你的工具 schema，或你的上下文长度。

.. _planner-custom:

添加规划器
---------------

自定义规划器的接口、接入步骤和验证要求见 :doc:`../development/add_planner`。

设置规划器的运行限制
-----------------------

以下参数的作用范围并不相同：

- ``--max-tokens`` 只限制 ``api`` 规划器每次回复的 token 数。LIBERO 类任务通常使用 ``8192`` 即可；RoboCasa 的长时序任务可以在模型支持的范围内调大。
- ``--max-turns`` 限制工具调用的总轮数。单个 LIBERO 任务通常不会超过 30 轮；RoboCasa 的长时序任务可能接近默认的 ``100``。
- ``--planner-timeout-s`` 限制规划器的运行时间。

模型调用 ``finish`` 工具后，规划器会记录相应的结束状态。达到轮数上限或超时时，运行结束，主程序仍会保存对话记录。超时或 SDK 异常会写入规划器结果，并输出到日志。
