命令行参考
==========

使用 ``rpent --robot <name> --help`` 查看所选环境的完整参数。共享参数由 RPent 提供，任务选择、资源路径和服务参数由各环境补充。

.. code-block:: bash

   rpent --help
   rpent --robot libero --help
   rpent --robot robocasa --help
   rpent --robot robotwin --help

常用参数
------------

这些参数控制规划器、记忆、输出目录和 Dashboard。

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - 参数
     - 默认值
     - 说明
   * - ``--robot``
     - ``—``
     - 选择已发现的机器人或仿真环境。
   * - ``--planner``
     - ``api``
     - ``api``、``claude_code``、``codex`` 或 ``flash``。
   * - ``--model``
     - ``—``
     - ``api`` 要求服务提供方前缀；SDK 规划器使用各自默认值。
   * - ``--max-turns``
     - ``100``
     - 规划器轮数上限；``api`` 按整段对话的模型请求次数计算，包括重试和后续输入。
   * - ``--max-tokens``
     - ``8192``
     - 仅 ``api``：每次模型回复的 Token 上限。
   * - ``--reasoning-effort``
     - ``api``：``high``；``claude_code``/``codex``：``none``
     - ``api``、``claude_code`` 与 ``codex`` 的推理强度：``none``、``low``、``medium``、``high`` 或 ``xhigh``。在我们的 LIBERO Pro Long 评测中，关闭 reasoning 将平均运行时间从约 13.2 分钟缩短至 7.9 分钟（约 40%）。较高强度可能提升任务成功率；实际支持的档位取决于所选模型。
   * - ``--planner-timeout-s``
     - ``—``
     - 默认使用 ``CODEX_TIMEOUT_S`` （仅 Codex）、``CELL_TIMEOUT_S`` 或 1200 秒；终端交互式 API/Claude 会话不受此限。
   * - ``--base-url``
     - ``—``
     - 仅 ``api``：覆盖模型服务地址。SDK 规划器使用各自的环境变量。
   * - ``--no-images``
     - ``false``
     - 仅 ``api``：不向模型发送图像。
   * - ``--claude-code-max-budget-usd``
     - ``—``
     - Claude Code 美元预算；默认取 ``MAX_BUDGET_USD`` 或 10。
   * - ``--output-dir``
     - ``—``
     - 保存本次运行文件的目录；默认目录名由环境生成。
   * - ``--memory-profile``
     - ``—``
     - 通常评测默认为 ``hf``；探索和 WAM 评测使用 ``local``。
   * - ``--memory-dir``
     - ``—``
     - 本地记忆目录；与本地模式或探索一起使用。
   * - ``--memory-repo``
     - ``—``
     - HF 数据集 ID 或 URL，可附加 @revision；仅用于评测。
   * - ``--explore``
     - ``false``
     - 在支持的环境中启用探索。
   * - ``--tui``、``-t``
     - ``false``
     - 在终端输入指令，见 :ref:`terminal-interaction`。
   * - ``--dashboard``
     - ``false``
     - 启动 Dashboard 会话。
   * - ``--dashboard-host``
     - ``127.0.0.1``
     - Dashboard 监听地址。
   * - ``--dashboard-port``
     - ``0``
     - 0 表示自动分配可用端口。
   * - ``--dashboard-language``
     - ``en``
     - ``en`` 或 ``zh-cn``。
   * - ``--verbose``
     - ``false``
     - 启用 DEBUG 日志。

环境参数
------------

LIBERO 使用 ``--suite``、``--task``、``--seed`` 和 ``--libero-type`` 选择任务；默认类型为 ``LIBERO_TYPE`` 或 ``pro``。RoboCasa365 和 RoboTwin 使用 ``--task-name``。Franka 使用 ``--task-id`` 和 ``--robot-config``。具体选项见 :doc:`../simulators/libero`、:doc:`../simulators/robocasa`、:doc:`../simulators/robotwin`、:doc:`../real_world_robots/franka` 和 :doc:`../real_world_robots/dual_franka`。

``--env-endpoint``、``--am-endpoint`` 和 LIBERO 的 ``--sam3-endpoint`` 分别连接已有的环境、动作模型（VLA 或 WAM）和 SAM3 服务。RPent RPC 地址格式为 ``[protocol://]host:port``，默认协议为 ``http``；LingBot 使用原生 WebSocket，见 :doc:`../simulators/robotwin`。GPU 分配和远程服务见 :doc:`advanced_deployment`。

``--am-backend`` 选择当前平台支持的一个动作模型；LIBERO 和 RoboTwin 默认分别为 ``pi05`` 和 ``lingbot``。VLA 和 WAM 共用 ``--am-endpoint`` 连接已有服务，共用 ``--am-checkpoint``、``--am-python``、``--am-root`` 和 ``--am-cuda-device`` 配置本地部署。使用 endpoint 时，显式传入本地部署参数会报错。

WAM 采用 ``local`` 记忆模式并注册记忆文件工具，默认目录按后端区分，可通过 ``--memory-dir`` 指定目录；目前不支持 HF 记忆、Flash 和探索模式。模型启动参数及支持的组合见 :doc:`../development/add_action_model`。

``--am-checkpoint`` 按后端要求提供本地权重或服务模型标识。``--am-python`` 指定已准备好的模型解释器，VLA 默认使用当前解释器；``--am-root`` 指定工作目录，默认使用 RPent 仓库目录，Cosmos Policy 则要求其源码目录。Franka 和 YAM 仅提供外部服务连接。

``--predict-future`` 在本地部署时开启未来状态和价值输出，目前仅 Cosmos Policy 支持，其他后端会拒绝此参数。使用 ``--am-endpoint`` 连接已有服务时，应在启动 worker 时配置。RPent 尚不利用这些输出进行规划。

``--am-cuda-device`` 独立选择本地模型的 GPU。LIBERO、RoboTwin、RoboCasa、RoboDojo 和 dual-Franka 支持本地模型部署；外部模型服务的 GPU 在部署时选择。LIBERO、RoboTwin、RoboCasa、RoboDojo 使用 ``--env-cuda-device`` 选择环境 GPU。LIBERO、RoboDojo 和 dual-Franka 使用 ``--sam3-cuda-device`` 选择本地 SAM3 的 GPU。

显存允许时，各服务可共用同一张 GPU。连接外部 endpoint 时，设备由该服务启动时的配置决定。

RPent CLI 保留原有 ``--cuda-device`` 作为本地服务的共享默认值；各服务专属参数优先，与参数顺序无关，两者均未设置时继承 ``CUDA_VISIBLE_DEVICES``。例如，``--cuda-device 0 --am-cuda-device 1`` 让环境和 SAM3 使用 GPU 0，WAM worker 使用 GPU 1。独立服务的启动命令仍可使用其自身支持的 ``--cuda-device``。

``rpent --robot libero --am-backend fast-wam --help`` 按连接服务、本地部署和所选模型展示参数。模型专属参数保留模型名前缀，传入其他模型的参数会报错。动作模型接入指南分别介绍一次部署模型服务和反复运行任务，并提供连接外部服务、由 RPent 启动服务两种方式的示例。

以下默认值适用于 LIBERO，其他环境以各自的 ``--help`` 为准。

.. list-table::
   :header-rows: 1

   * - LIBERO 参数
     - 默认值
     - 说明
   * - ``--seed``
     - ``0``
     - 环境随机种子。
   * - ``--memory-version``
     - ``auto``
     - HF 记忆版本：``auto``、``GPT_5.5_xhigh`` 或 ``GPT_6_astra_low``。详见 :doc:`memory`；显式版本不能与本地记忆或探索模式同时使用。
   * - ``--max-episode-steps``
     - ``10000``
     - 环境步数上限，与规划器的 ``--max-turns`` 分开计算。

动作模型参数兼容
~~~~~~~~~~~~~~~~

编写新命令时使用下表左列的当前参数。右列仅作为兼容旧命令的隐藏别名保留，不显示在 ``--help`` 中，并写入与左列相同的配置字段。

通用部署参数统一使用 ``--am-*``。模型专属参数使用 ``--pi05-*``、``--cosmos-*``、``--cosmos3-*``、``--fast-wam-*`` 或 ``--lingbot-*``；其中 ``fast-wam`` 是模型名称，不是旧的通用 ``--wam-*`` 前缀。

.. list-table:: 模型参数名称
   :header-rows: 1

   * - 当前参数
     - 兼容旧参数（隐藏别名）
   * - ``--am-backend``
     - ``--vla-backend``、``--wam-backend``
   * - ``--am-backend pi05``
     - RoboDojo 的 ``--policy-backend rlinf``
   * - ``--am-endpoint``
     - ``--vla-endpoint``、``--wam-endpoint``
   * - ``--am-checkpoint``
     - ``--vla-model-path``、``--wam-checkpoint``
   * - ``--am-python``
     - ``--pi05-python``、``--wam-python``
   * - ``--am-root``
     - ``--wam-root``
   * - ``--am-cuda-device``
     - ``--vla-cuda-device``、``--wam-cuda-device``
   * - ``--pi05-repo-id``
     - ``--vla-repo-id``
   * - ``--cosmos-text-embeddings``
     - ``--wam-text-embeddings``
   * - ``--cosmos-cached-instructions-only``
     - ``--wam-cached-instructions-only``
   * - ``--fast-wam-config``
     - ``--wam-config``
   * - ``--fast-wam-dataset-stats``
     - ``--wam-dataset-stats``

模型服务环境变量
------------------------

在启动 RPent 的终端中设置所选规划器的认证信息和服务器地址。

.. list-table::
   :header-rows: 1

   * - 规划器
     - API Key
     - 服务地址
   * - ``api`` / Anthropic; ``claude_code``
     - ``ANTHROPIC_API_KEY``
     - ``ANTHROPIC_BASE_URL``
   * - ``api`` / OpenAI
     - ``OPENAI_API_KEY``
     - ``OPENAI_BASE_URL``
   * - ``codex``
     - ``CODEX_API_KEY``
     - ``CODEX_BASE_URL``

Codex 也可复用已有的 Codex 认证。模型选择、地址优先级和本地服务示例见 :doc:`configure_planner`。连接检查使用 ``rpent-check-llm``；记忆管理使用 ``rpent-memory``，操作方法见 :doc:`memory`。

模型与资源路径由各环境读取，例如 LIBERO 的 ``PI05_CHECKPOINT_PATH``、``SAM3_CHECKPOINT_PATH``，RoboCasa365 的 ``ROBOCASA_ASSETS_PATH``，以及 RoboTwin 的 ``ROBOTWIN_ASSETS_PATH``、``LINGBOT_MODEL_PATH``。

.. _run-output-files:

输出文件
------------

``--output-dir`` 指定本次运行的输出目录。LIBERO 未指定该参数时，默认保存在仓库下的 ``logs/<timestamp>_<suite>_t<task>_s<seed>/``；其他环境的目录命名见各环境页。

下表以 LIBERO 为例。实际生成的文件取决于环境、录像和任务运行结果。

.. list-table::
   :header-rows: 1

   * - 文件或路径
     - 用途
   * - ``transcript_*.json``
     - 规划器对话、工具调用及结束信息。
   * - ``episode.mp4``
     - 本次任务的录像。
   * - ``run.log``
     - RPent 运行日志。
   * - ``env_server.log`` / ``vla_server.log`` / ``wam_server.log`` / ``sam3_server.log``
     - 由本次运行启动的环境和模型服务器日志。
   * - ``*_recipe.jsonl``
     - 成功任务导出的动作序列；失败任务不保证生成该文件。
   * - ``states.json``
     - ``EnvState`` 的内部状态索引。通过 Dashboard 或 ``view_env_state`` 查看状态，调用方不要直接解析此文件。
   * - ``agentview_depth.npz/00.npz``
     - 逐步保存的观测示例：目录名是逻辑文件名，目录内按步数编号，如 ``00.npz``、``01.npz``。

整次运行的文件保存在输出目录根部，逐步观测保存在各自的子目录中。成功判定以各环境页的原生判定为准；动作序列、录像和规划器总结本身都不能替代该判定。
