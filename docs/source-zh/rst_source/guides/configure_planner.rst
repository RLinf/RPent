规划器配置
==========

RPent 可以直接调用模型 API，也可以通过 Claude Code、Codex 等工具规划任务。选好规划器后，需要配置模型与认证，再检查连接。下面的任务命令以 LIBERO 为例；运行前请按 :doc:`../get_started/quickstart` 完成安装和环境配置。

用 ``--planner`` 选择后端，再按对应章节完成配置：

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - 规划器
     - 适用场景
     - 认证与服务地址
   * - :ref:`api <planner-api>`\ （默认）
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

不同后端的配置方式如下：

- ``--base-url`` 仅适用于 ``api``，传给其他在线规划器会报错。
- 使用 CLI 后端时，RPent 会自动启动并管理对应的 CLI，因此无需手动启动后台进程。

.. _planner-interaction:

如果需要在运行中补充指令，可使用终端 ``--tui`` 或 ``--dashboard``；所有在线规划器都支持这两种模式。其中 Claude Code、Codex、Gemini CLI 和 OpenCode 的追加消息默认排队，而 ``api`` 的终端交互方式有所不同；具体用法和手动中断方法见 :doc:`dashboard`。

.. _planner-api:

API 规划器（直接调用模型 API）
-------------------------------------

``api`` 直接调用模型服务，通过环境变量读取密钥和服务地址。使用时，按服务的 API 协议选择 ``--model`` 前缀，冒号后填写服务支持的模型 ID：

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

使用自定义服务时，设置表中的地址变量，或在检查和运行命令中都添加 ``--base-url``。两者同时设置时，以 ``--base-url`` 为准。

如果模型只支持文本，需加 ``--no-images``。规划器此时只能依赖文本状态，因此可能影响任务表现。

.. _planner-claude-code:

Claude Code 规划器
------------------------

RPent 已包含 Claude Code 所需依赖，因此无需单独安装 CLI。先配置 Claude 登录或 ``ANTHROPIC_API_KEY``，再运行：

.. code-block:: bash

   rpent-check-llm --planner claude_code --model claude-opus-4-8
   rpent --robot libero --planner claude_code --model claude-opus-4-8 \
     --suite libero_goal_task --task 1 --seed 0

``--model`` 不加提供商前缀，省略时使用 ``sonnet``。

需要限制费用时，设置 ``--claude-code-max-budget-usd``。未指定时读取 ``MAX_BUDGET_USD``，该变量也未设置时使用 10 美元。

其他认证方式见 `Claude Agent SDK 文档 <https://code.claude.com/docs/en/agent-sdk/overview>`_。

.. dropdown:: 通过 Claude Code 使用本地模型

   本地服务需要兼容 Anthropic Messages API。例如，服务已在端口 8000 提供 ``Qwen/Qwen3.6-27B``：

   .. code-block:: bash

      export ANTHROPIC_BASE_URL=http://127.0.0.1:8000
      export ANTHROPIC_API_KEY=EMPTY
      rpent --robot libero --planner claude_code --model Qwen/Qwen3.6-27B \
        --suite libero_goal_task --task 1 --seed 0

   ``EMPTY`` 仅用于无需认证的本地服务。

   Claude Code 对未知模型默认按 200,000 token 上下文窗口处理。如果服务的上下文长度与此不同，需按 `Claude Code 环境变量文档 <https://code.claude.com/docs/en/env-vars>`_ 配置上下文和自动压缩参数。

.. _planner-codex:

.. _codex-planner:

Codex 规划器
------------------

Codex 规划器默认复用已有登录。使用自定义服务时，先确认服务兼容 OpenAI Responses API，再设置 ``CODEX_BASE_URL`` 和 ``CODEX_API_KEY``。Codex 规划器不读取 ``OPENAI_BASE_URL`` 或 ``OPENAI_API_KEY``。

.. code-block:: bash

   rpent-check-llm --planner codex --model gpt-5.5
   rpent --robot libero --planner codex --model gpt-5.5 \
     --suite libero_goal_task --task 1 --seed 0

模型按 ``--model``、``CODEX_MODEL``、Codex 默认模型的顺序选择。

需要 fast 服务档位时，可设置 ``CODEX_SERVICE_TIER=fast``，但该变量不会改变推理强度。

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

   这里手动设置上下文长度，是因为 vLLM 返回的字段为 ``max_model_len``，而 Codex 读取的是 ``context_window``。两者不一致，Codex 遇到未知模型 ID 时便会回退到默认元数据。

   ``--model`` 必须与 ``--served-model-name`` 一致。``CODEX_MODEL_CONTEXT_WINDOW`` 应匹配服务的 ``--max-model-len``；``CODEX_AUTO_COMPACT_TOKEN_LIMIT`` 应小于该值，为后续回复留出空间。这两个上下文环境变量均可省略，省略时使用 Codex 默认值。

   如果使用其他模型，请按其部署说明选择解析参数。

.. _planner-codex-isolation:

Codex 独立配置与登录
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

若需要将 RPent 与日常 Codex 的配置、登录和会话记录分开，可使用独立目录。先在运行 RPent 的终端中指定目录，并定位 SDK 自带的 Codex 程序：

.. code-block:: bash

   cd /path/to/RPent
   source .venv-libero/bin/activate
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

每个新终端都需设置上述环境变量；已有 ``sqlite_home`` 配置时，也应指向独立目录。若使用自定义 API，则改为设置本次部署的 ``CODEX_API_KEY`` 和 ``CODEX_BASE_URL``。更多登录方式见 `Codex 认证说明 <https://developers.openai.com/codex/auth/>`_。

独立配置目录不会隔离工作区或迁移旧会话；任务记忆仍由 ``--memory-dir`` 选择。另外，不要将凭据或配置目录提交到版本库。

.. _planner-gemini-cli:

Gemini CLI 规划器
----------------------

RPent 复用 Gemini CLI 的认证和服务地址配置。如果尚未安装，先安装支持 ``--acp`` 的 `Gemini CLI <https://github.com/google-gemini/gemini-cli>`_，再启动 CLI 完成配置：

.. code-block:: bash

   npm install -g @google/gemini-cli
   gemini

确认 Gemini CLI 能正常对话后退出，再从 RPent 检查并运行：

.. code-block:: bash

   rpent-check-llm --planner gemini_cli
   rpent --robot libero --planner gemini_cli \
     --suite libero_goal_task --task 1 --seed 0

省略 ``--model`` 时使用 Gemini CLI 默认模型；指定时填写服务支持的原始模型 ID，例如 ``--model Argon``，不加 ``google:``。检查和运行应使用相同模型。

若可执行文件不在 ``PATH`` 中或不叫 ``gemini``，通过 ``GEMINI_BIN`` 指定路径。

.. _planner-opencode:

OpenCode 规划器
--------------------------------

RPent 复用 OpenCode 已有的模型提供商、认证和服务地址配置。如果尚未配置，请按 `OpenCode 安装指南 <https://opencode.ai/docs/>`_ 完成安装和配置，并确认能独立对话。

RPent 需要 OpenCode 的 ``acp`` 子命令支持 ``--pure``，已测试版本为 v1.18.18。运行 RPent 前，先检查版本和可用选项：

.. code-block:: bash

   opencode --version
   opencode acp --help

确认帮助信息包含 ``--pure`` 后，检查连接并运行任务：

.. code-block:: bash

   rpent-check-llm --planner opencode
   rpent --robot libero --planner opencode \
     --suite libero_goal_task --task 1 --seed 0

省略 ``--model`` 时使用 OpenCode 默认模型；切换模型时，将 ``opencode models`` 列出的完整 ``provider/model`` ID 传给检查和运行命令，先确认所选模型可用。

若可执行文件不在 ``PATH`` 中，通过 ``OPENCODE_BIN`` 指定路径。

.. _planner-check:

验证配置
------------

运行任务或打开 ``--dashboard`` 前，先在终端使用相同的在线规划器和模型执行 ``rpent-check-llm``。检查会发送一次真实模型请求，但不启动机器人，也不调用工具或发送图像。

如果服务响应较慢，可用 ``--timeout-s`` 延长等待时间。下面的命令同时用 ``--json`` 输出诊断结果，便于脚本读取：

.. code-block:: bash

   rpent-check-llm --planner gemini_cli --timeout-s 120 --json

成功退出码为 ``0``，失败为 ``1``。失败时按提示检查认证、模型 ID、服务地址或网络。不过，即使检查通过，也只验证了基本连通性；工具调用、图像输入和机器人任务仍需单独验证。

.. dropdown:: 诊断时限与 JSON 状态值

   诊断默认时限为：``api`` 30 秒，``opencode`` 90 秒。Claude Code、Codex、Gemini CLI 在启动环境中有对应密钥时为 90 秒；未设置密钥时，等待时间缩短为 25 秒，但仍可使用保存的 CLI 登录。

   Claude Code 和 Codex 若既未检测到密钥，也未检测到登录配置，会直接报告缺少认证。``--timeout-s`` 可覆盖诊断时限，与任务的运行时限独立。

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

配置与工具隔离
------------------

为避免本地开发配置干扰机器人任务，RPent 在规划会话中限制自动加载的指令和工具：

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - 规划器
     - 指令与 skills
     - 可用工具
   * - Claude Code
     - 不加载用户和项目的 Claude Code 配置。
     - RPent 工具，以及默认开放的 Bash、Read、Write、Glob、Grep。
   * - Codex
     - 不加载项目开发指令，并禁用 RPent 仓库中的 skills。
     - RPent 工具和 Codex 原生工具；原生工具的权限受 ``--codex-sandbox`` 限制。
   * - Gemini CLI
     - 不加载项目开发指令，禁用 skills。
     - 只开放 RPent 工具，禁用扩展。
   * - OpenCode
     - 禁用项目配置、Claude Code 配置与外部 skills，保留自身全局配置。
     - 只开放 RPent 工具，禁用外部插件。

如需添加工具，见 :ref:`external-mcp-servers`；如需让规划器直接提交控制数值，见 :ref:`planner-direct-actions`。

这些会话设置是 RPent 内部的配置，不修改全局配置。

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
     - 整个任务的 RPent 工具调用上限，默认 ``100``。追加消息和中断均不重置额度。
   * - ``--planner-timeout-s``
     - 在线规划器运行的墙钟超时（秒；终端交互式 API/Claude 会话不受限）。默认按优先级取 ``CODEX_TIMEOUT_S``\ （仅 ``codex``）、``GEMINI_CLI_TIMEOUT_S``\ （仅 ``gemini_cli``）、``CELL_TIMEOUT_S`` 或 ``1200``。
   * - ``--max-tokens``
     - 仅 ``api``：单次回复 token 上限，默认 ``8192``。
   * - ``--reasoning-effort``
     - 可选 ``none``、``low``、``medium``、``high``、``xhigh``，实际支持取决于后端和模型。API 默认 ``high``，其余默认 ``none``。

Gemini CLI 的推理强度最高支持 ``high``；OpenCode 使用所选模型提供的档位。这两个后端的 ``none`` 表示沿用 CLI 配置。

每次获准执行的 RPent 工具调用计一次，包括只读工具、``finish`` 和执行失败的调用。同一回复中的多个调用分别计数。执行前被拒绝的无效调用、规划器 CLI 自带的工具不计入额度。所有在线规划器的 ``turns_used`` 都记录这个计数。

达到上限后，RPent 等待已获准的调用执行完毕，再结束规划，不报错。额度用完不代表任务成功。每次规划会话使用独立额度；``flash`` 不使用此额度。

.. _planner-custom:

添加规划器
---------------

接入其他规划器的开发接口见 :doc:`../development/add_planner`。
