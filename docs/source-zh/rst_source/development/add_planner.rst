添加规划器
===============

规划器负责调用模型、执行机器人工具并返回任务结果。接入新的后端有两种方式：直接对接 API 或 SDK 时，继承 ``Planner``；对接支持 ACP（Agent Client Protocol）的 CLI 时，继承 ``AcpPlanner`` 复用会话执行和交互逻辑。两种方式共用 RPent 工具包，完成实现后都需要注册并验证。

自定义规划器
--------------

接入模型 API 或 SDK 时，继承 ``Planner`` 并实现 ``solve``。

以下为接口示意，省略了模型请求和工具循环，不能直接作为实现运行。完整示例可参考 ``rpent/planner/api_loop.py``。

.. code-block:: python

   # rpent/planner/my_planner.py
   from rpent.planner.base import Planner, PlannerResult

   class MyPlanner(Planner):
       def solve(
           self,
           *,
           system_prompt,
           user_message,
           toolkit,
           max_turns,
           input_queue=None,
           dashboard_interaction=None,
       ):
           tools = toolkit.list_tools()
           # 使用 system_prompt、user_message 和 tools 调用模型。
           # 每次工具调用都通过下面的接口执行：
           tool_result = toolkit.execute_tool(tool_name, arguments)
           ...
           return PlannerResult(
               finish_result=finish_result,
               messages=messages,
               stats=stats,
               error=error,
           )

任何规划器都必须：

1. 接收已经渲染好的 ``system_prompt`` 和 ``user_message``。
2. 从 ``toolkit.list_tools()`` 取得工具的 ``name``、``description`` 和 ``input_schema``，并通过 ``toolkit.execute_tool(name, arguments)`` 执行工具。
3. 将 ``ToolResult.to_text()`` 的文本和 ``ToolResult.images`` 中的 PNG 字节转换成模型 SDK 所需的格式。
4. 识别 ``ToolResult.data.get("_finish")``，并按 ``max_turns`` 等限制终止循环。
5. 返回包含结束状态、消息、统计信息和可选错误的 ``PlannerResult``。

由于 RPent 的工具定义和提示词渲染流程保持不变，新增规划器无需修改工具或环境服务。接口参见 :doc:`architecture`；添加工具见 :doc:`add_primitive`。

ACP CLI 适配器
--------------

如果 CLI 已支持 ACP，可以复用 ``rpent/planner/utils/acp.py`` 中的 ``AcpPlanner``。继承后设置 ``backend`` 和显示名称 ``label``。在 ``_create_session`` 中准备启动命令、环境和工具策略，并返回 ``AcpSession``。

完整示例见 ``rpent/planner/gemini_cli.py`` 和 ``rpent/planner/opencode.py``；两者均通过 ACP 管理会话，通过 HTTP MCP 调用 RPent 工具。

``AcpPlanner`` 负责会话管理、事件记录、连接探测、消息排队、中断及进程清理。适配器需按 CLI 的要求处理以下配置：

- MCP 接入：通过 ``AcpSession`` 的 ``mcp_url`` 传入地址，如 OpenCode；或写入工作区配置，如 Gemini CLI。
- 模型选择：使用 CLI 的 ACP 模型选择器时，才向 ``AcpSession`` 传入模型 ID。
- 认证：沿用 CLI 的认证配置。
- 轮数限制：ACP 没有统一的模型轮数上限，因此需使用 CLI 原生的限制选项。

接入与验证
---------------

实现完成后，需要让 RPent 能构建规划器并检查连接。Dashboard 复用命令行选定的规划器，因此无需单独接入。两种实现的注册位置如下：

- 自定义 ``Planner``：在 ``rpent/planner/base.py`` 的 ``build_planner`` 中添加构建分支。在线后端还需在 ``rpent/planner/check.py`` 中注册到 ``CHECK_PLANNERS``，设置诊断超时和连接探测；CLI 从该列表生成在线规划器选项。自行配置模型服务的 CLI/SDK 规划器应加入 ``EXTERNALLY_CONFIGURED_PLANNERS``。
- ``AcpPlanner`` 适配器：在 ``rpent/planner/check.py`` 中加入 ``ACP_PLANNERS``，设置诊断超时，并在 ``build_planner`` 和 ``_check_acp`` 中选择对应的类。``ACP_PLANNERS`` 统一用于在线规划器注册、外部配置处理以及仅接受 ``--reasoning-effort none`` 的校验。

注册后，参考 ``tests/unit_tests/rpent/planner/`` 编写离线测试，覆盖工具调用、文本和图像返回、真实的 ``finish`` 调用、轮数上限、错误结果、消息排队、中断及进程清理。离线测试通过后，再单独验证真实模型连接。
