规划器配置
==========

本页说明如何选择规划器、配置模型和认证，并在运行任务前检查连接。先完成 RPent 安装；下方任务命令以已按 :doc:`../get_started/quickstart` 配置好的 LIBERO 为例。

用 ``--planner`` 选择后端，再按对应章节完成配置：

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - 规划器
     - 适用场景
     - 认证与服务地址
   * - :ref:`api <planner-api>` （默认）
     - 直接连接模型 API，按服务协议选择模型前缀。
     - 环境变量；可用 ``--base-url`` 覆盖地址。
   * - :ref:`claude_code <planner-claude-code>`
     - 使用 Claude Code；支持 Anthropic Messages 兼容服务。
     - Claude 登录，或 ``ANTHROPIC_API_KEY`` / ``ANTHROPIC_BASE_URL``。
   * - :ref:`codex <planner-codex>`
     - 使用 Codex；支持 OpenAI Responses 兼容服务。
     - Codex 登录，或 ``CODEX_API_KEY`` / ``CODEX_BASE_URL``。
   * - :ref:`gemini_cli <planner-gemini-cli>`
     - 使用已配置好的 Gemini CLI。
     - 在 Gemini CLI 中配置。
   * - :ref:`opencode <planner-opencode>`
     - 使用已配置好的 OpenCode 模型服务。
     - 在 OpenCode 中配置。
   * - :doc:`flash`
     - 重放已有计划，无需在线模型规划。
     - 无需模型 API；仍需感知和 VLA 服务，见 :doc:`flash`。

``--base-url`` 仅适用于 ``api``，其他在线规划器会拒绝此参数。使用 CLI 后端时，RPent 会自动启动并管理所选工具；无需手动启动后台进程。

.. _planner-api:

API 规划器（直接调用模型 API）
-------------------------------------

按服务的 API 协议选择 ``--model`` 前缀，冒号后填写服务提供的模型 ID：

.. list-table::
   :header-rows: 1
   :widths: 20 25 30 25

   * - API 协议
     - ``--model`` 示例
     - 密钥环境变量
     - 自定义地址环境变量
   * - Anthropic Messages
     - ``anthropic:claude-opus-4-8``
     - ``ANTHROPIC_API_KEY``
     - ``ANTHROPIC_BASE_URL``
   * - Gemini
     - ``google:Argon``
     - ``GOOGLE_API_KEY`` 优先，否则 ``GEMINI_API_KEY``
     - ``GEMINI_BASE_URL``
   * - OpenAI Responses
     - ``openai:gpt-5.5``
     - ``OPENAI_API_KEY``
     - ``OPENAI_BASE_URL``
   * - OpenAI Chat Completions
     - ``openai-chat:glm-5.2``
     - ``OPENAI_API_KEY``
     - ``OPENAI_BASE_URL``

例如，设置 Anthropic 密钥，检查连接后运行一个任务：

.. code-block:: bash

   export ANTHROPIC_API_KEY="your-api-key"
   rpent-check-llm --planner api --model anthropic:claude-opus-4-8
   rpent --robot libero --planner api --model anthropic:claude-opus-4-8 \
     --suite libero_goal_task --task 1 --seed 0

使用自定义服务时，设置表中的地址变量，或在检查和运行命令中都添加 ``--base-url``；命令行参数优先。纯文本模型需加 ``--no-images``，此时规划器只能依赖文本状态，可能影响任务表现。

.. _planner-claude-code:

Claude Code 规划器
------------------------

RPent 已包含 Claude Code 所需依赖，无需单独安装 CLI。先配置 Claude 登录或 ``ANTHROPIC_API_KEY``，再运行：

.. code-block:: bash

   rpent-check-llm --planner claude_code --model claude-opus-4-8
   rpent --robot libero --planner claude_code --model claude-opus-4-8 \
     --suite libero_goal_task --task 1 --seed 0

``--model`` 不加提供商前缀，省略时使用 ``sonnet``。需要限制费用时，设置 ``--claude-code-max-budget-usd``；默认读取 ``MAX_BUDGET_USD``，未设置时为 10 美元。认证方式见 `Claude Agent SDK 文档 <https://code.claude.com/docs/en/agent-sdk/overview>`_。

.. dropdown:: 通过 Claude Code 使用本地模型

   本地服务需要兼容 Anthropic Messages API。例如，服务已在端口 8000 提供 ``Qwen/Qwen3.6-27B``：

   .. code-block:: bash

      export ANTHROPIC_BASE_URL=http://127.0.0.1:8000
      export ANTHROPIC_API_KEY=EMPTY
      rpent --robot libero --planner claude_code --model Qwen/Qwen3.6-27B \
        --suite libero_goal_task --task 1 --seed 0

   ``EMPTY`` 仅用于无需认证的本地服务。Claude Code 对未知模型默认按 200,000 token 上下文窗口处理；服务限制不同时，按 `Claude Code 环境变量文档 <https://code.claude.com/docs/en/env-vars>`_ 配置上下文和自动压缩参数。

.. _planner-codex:

.. _codex-planner:

Codex 规划器
------------------

默认复用已有的 Codex 登录。自定义服务需兼容 OpenAI Responses API，并设置 ``CODEX_BASE_URL`` 和 ``CODEX_API_KEY``；此规划器不读取 ``OPENAI_BASE_URL`` 或 ``OPENAI_API_KEY``。

.. code-block:: bash

   rpent-check-llm --planner codex --model gpt-5.5
   rpent --robot libero --planner codex --model gpt-5.5 \
     --suite libero_goal_task --task 1 --seed 0

模型按 ``--model``、``CODEX_MODEL``、Codex 默认模型的顺序选择。可设置 ``CODEX_SERVICE_TIER=fast`` 请求 fast 服务档位；它不改变推理强度。

.. _planner-codex-isolation:

Codex 独立配置与登录
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

若需要将 RPent 与日常 Codex 的配置、登录和会话记录分开，可使用独立目录。在运行 RPent 的终端中使用 SDK 依赖自带的 Codex 程序：

.. code-block:: bash

   cd /path/to/RPent
   source .venv/bin/activate
   export CODEX_HOME="$PWD/.codex-rpent-live"
   export CODEX_SQLITE_HOME="$CODEX_HOME"
   export CODEX_BIN="$(python -c 'from codex_cli_bin import bundled_codex_path; print(bundled_codex_path())')"
   mkdir -p "$CODEX_HOME"
   unset CODEX_API_KEY CODEX_BASE_URL OPENAI_API_KEY OPENAI_BASE_URL CODEX_ACCESS_TOKEN

首次登录前，在 ``$CODEX_HOME/config.toml`` 中设置：

.. code-block:: toml

   cli_auth_credentials_store = "file"

然后登录并确认状态：

.. code-block:: bash

   "$CODEX_BIN" login --device-auth
   "$CODEX_BIN" login status

每个新终端都需设置上述环境变量；已有 ``sqlite_home`` 配置时，也应指向独立目录。自定义 API 用户应改为设置本次部署的 ``CODEX_API_KEY`` 和 ``CODEX_BASE_URL``。更多登录方式见 `Codex 认证说明 <https://developers.openai.com/codex/auth/>`_。

该配置不会隔离工作区或迁移旧会话；任务记忆另由 ``--memory-dir`` 选择。不要提交凭据或该目录。

.. dropdown:: 通过 Codex 使用本地模型

   例如，使用 vLLM 提供 Qwen3.6-27B 的 Responses API：

   .. code-block:: bash

      vllm serve /path/to/Qwen3.6-27B \
        --served-model-name Qwen/Qwen3.6-27B \
        --max-model-len 262144 \
        --reasoning-parser qwen3 \
        --enable-auto-tool-choice \
        --tool-call-parser qwen3_coder

   在运行 RPent 的终端设置地址和服务实际支持的上下文长度：

   .. code-block:: bash

      export CODEX_BASE_URL=http://127.0.0.1:8000
      export CODEX_API_KEY=EMPTY
      export CODEX_MODEL_CONTEXT_WINDOW=262144
      export CODEX_AUTO_COMPACT_TOKEN_LIMIT=230000
      rpent --robot libero --planner codex --model Qwen/Qwen3.6-27B \
        --suite libero_goal_task --task 1 --seed 0

   vLLM 返回的上下文字段是 ``max_model_len``，Codex 读取的却是 ``context_window``，遇到未知模型 ID 会采用默认元数据。

   ``--model`` 必须与 ``--served-model-name`` 一致。``CODEX_MODEL_CONTEXT_WINDOW`` 应匹配服务的 ``--max-model-len``；自动压缩阈值应小于该值，为后续回复留出空间。两个变量均可省略，省略时使用 Codex 默认值。使用其他模型时，按其部署说明选择解析参数。

.. _planner-gemini-cli:

Gemini CLI 规划器
----------------------

安装支持 ``--acp`` 的 `Gemini CLI <https://github.com/google-gemini/gemini-cli>`_，并先在其中完成认证和服务地址配置：

.. code-block:: bash

   npm install -g @google/gemini-cli
   gemini

确认 Gemini CLI 能正常对话后退出，再从 RPent 检查并运行：

.. code-block:: bash

   rpent-check-llm --planner gemini_cli
   rpent --robot libero --planner gemini_cli \
     --suite libero_goal_task --task 1 --seed 0

省略 ``--model`` 时使用 Gemini CLI 默认模型；指定时填写服务支持的原始模型 ID，例如 ``--model Argon``，不加 ``google:``。检查和运行应使用相同模型。若可执行文件不在 ``PATH`` 中或不叫 ``gemini``，通过 ``GEMINI_BIN`` 指定路径。

.. _planner-opencode:

OpenCode 规划器
--------------------------------

按 `OpenCode 安装指南 <https://opencode.ai/docs/>`_ 安装，配置模型提供商、认证和服务地址，并确认能独立对话。运行 ``opencode acp --help``，确认包含 ``--pure`` 选项；已测试版本为 v1.18.18。先检查安装版本和可用选项，再运行 RPent：

.. code-block:: bash

   opencode --version
   opencode acp --help
   rpent-check-llm --planner opencode
   rpent --robot libero --planner opencode \
     --suite libero_goal_task --task 1 --seed 0

省略 ``--model`` 时使用 OpenCode 默认模型；切换模型时，将 ``opencode models`` 列出的完整 ``provider/model`` ID 传给检查和运行命令，先确认所选模型可用。若可执行文件不在 ``PATH`` 中，通过 ``OPENCODE_BIN`` 指定路径。

.. _planner-check:

验证你的配置
------------

运行任务或打开 ``--dashboard`` 前，先在终端使用相同的规划器和模型执行 ``rpent-check-llm``。它会发送一次真实模型请求，但不启动机器人，也不调用工具或发送图像。响应较慢时增加诊断时限：

.. code-block:: bash

   rpent-check-llm --planner gemini_cli --timeout-s 120 --json

成功退出码为 ``0``，失败为 ``1``；``--json`` 输出便于脚本读取。失败时按提示检查认证、模型 ID、服务地址或网络。检查通过不代表工具调用、图像输入或机器人任务已验证。

诊断默认时限：``api`` 30 秒，``opencode`` 90 秒；Claude Code、Codex、Gemini CLI 在启动环境中有对应密钥时为 90 秒，否则为 25 秒。CLI 仍可使用保存的登录；Claude Code/Codex 若未检测到密钥或登录配置，会直接报告缺少认证。``--timeout-s`` 可覆盖诊断时限，与任务的运行时限独立。

.. dropdown:: JSON 状态值

   脚本可根据 ``status`` 字段判断结果，并读取 ``detail`` 获取诊断信息：

   - ``ok``：模型已回复。
   - ``missing_config``：模型配置缺失或不完整。
   - ``unsupported_provider``：不支持该 API 提供商前缀。
   - ``missing_api_key``：缺少凭据或需要登录。
   - ``invalid_model``：服务拒绝了模型 ID。
   - ``auth_failed``：认证被拒绝。
   - ``network_error``：连接、DNS、TLS 错误或请求超时。
   - ``provider_error``：服务拒绝请求，例如额度不足或触发限流。
   - ``sdk_error``：SDK、CLI 进程或未分类错误。

.. _planner-interaction:

交互输入与配置隔离
------------------

Claude Code、Codex、Gemini CLI、OpenCode 支持终端 ``--tui`` 和 ``--dashboard``，追加消息默认排队，手动中断方法见 :doc:`dashboard`。

规划任务不会自动加载 Claude/Codex/Gemini 的项目开发指令和 skills。OpenCode 禁用项目配置、Claude 配置与外部 skills，保留自身全局配置。Gemini CLI 和 OpenCode 开放 RPent 工具包，关闭扩展或外部插件；添加工具请使用 :ref:`external-mcp-servers`。这些会话设置不修改全局配置，也不提供操作系统级沙箱隔离。

.. _planner-limits:

设置规划器的运行限制
-----------------------

常用参数如下，完整选项见 :doc:`cli`：

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - 参数
     - 用途
   * - ``--max-turns``
     - 规划轮数上限，默认 ``100``；不是机器人动作数，见下表。
   * - ``--planner-timeout-s``
     - 任务运行时限；未指定时读取 ``CELL_TIMEOUT_S``，再回退到 1200 秒。Codex 优先读取 ``CODEX_TIMEOUT_S``，Gemini CLI 优先读取 ``GEMINI_CLI_TIMEOUT_S``。终端交互式 API/Claude 会话不受此限。
   * - ``--max-tokens``
     - 仅 ``api``：单次回复 token 上限，默认 ``8192``。
   * - ``--reasoning-effort``
     - API/Claude Code/Codex 可选 ``none``、``low``、``medium``、``high``、``xhigh``，实际支持取决于模型。API 默认 ``high``，其余默认 ``none``；Gemini CLI/OpenCode 仅接受 ``none``。

一次模型回复可能调用多个工具，因此各后端的轮数不能直接当作动作数比较：

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - 后端
     - 一轮的含义
   * - API
     - 一次模型请求，包含重试和追加消息；预算覆盖整段对话。
   * - Claude Code
     - 模型请求工具，RPent 执行后返回结果；无工具的最终答复不占预算。每次追加输入获得新的预算。
   * - Codex / Gemini CLI
     - 一次模型请求或回复；同一回复中的多个工具不分别计数。
   * - OpenCode
     - 一次模型执行步骤；同一步骤中的多个工具不分别计数。每次追加输入获得新的预算。

达到轮数上限后，API 正常结束且不报错，探索模式可继续下一次尝试；Claude Code 返回 ``error_max_turns``；OpenCode 生成不再调用工具的最终回复。达到上限不代表任务成功。``flash`` 不运行模型循环，不使用该预算。

Gemini CLI 和 OpenCode 的运行统计不包含 ``turns_used``，因为 ACP 不提供可靠的模型轮数。

直接执行动作
--------------------

LIBERO、RoboCasa、RoboTwin 可加 ``--enable-direct-action``，让规划器调用 ``execute_action(values=[...])`` 直接提交控制数值；仍可使用 VLA 和脚本工具，具体数值个数和范围由当前环境的工具说明给出。

.. _planner-custom:

添加规划器
---------------

接入其他规划器的开发接口见 :doc:`../development/add_planner`。
