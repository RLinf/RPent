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
     - ``none``
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
     - 评测默认为 ``hf``，探索默认为 ``local``。
   * - ``--memory-dir``
     - ``—``
     - 本地记忆目录；与本地模式或探索一起使用。
   * - ``--explore``
     - ``false``
     - 在支持的环境中启用探索。
   * - ``--interactive``、``-i``
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

``--env-endpoint``、``--vla-endpoint`` 和 LIBERO 的 ``--sam3-endpoint`` 可连接已启动的环境、VLA 或 SAM3 服务器，地址格式为 ``[protocol://]host:port``，默认协议为 ``http``。GPU 分配和远程服务见 :doc:`advanced_deployment`。

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
   * - ``env_server.log`` / ``vla_server.log`` / ``sam3_server.log``
     - 由本次运行启动的环境和模型服务器日志。
   * - ``*_recipe.jsonl``
     - 成功任务导出的动作序列；失败任务不保证生成该文件。
   * - ``states.json``
     - ``EnvState`` 的内部状态索引。通过 Dashboard 或 ``view_env_state`` 查看状态，调用方不要直接解析此文件。
   * - ``agentview_depth.npz/00.npz``
     - 逐步保存的观测示例：目录名是逻辑文件名，目录内按步数编号，如 ``00.npz``、``01.npz``。

整次运行的文件保存在输出目录根部，逐步观测保存在各自的子目录中。成功判定以各环境页的原生判定为准；动作序列、录像和规划器总结本身都不能替代该判定。
