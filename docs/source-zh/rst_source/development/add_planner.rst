添加规划器
===============

接入模型后端需要实现 ``Planner``，负责接收提示词、调用工具并返回任务结果。如果 CLI 支持 ACP（Agent Client Protocol），可继承 ``AcpPlanner`` 复用会话执行和交互逻辑。两条接入路径使用相同的工具包和注册步骤。

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

RPent 的工具定义和提示词渲染流程保持不变，新增规划器不需要修改工具或环境服务。接口参见 :doc:`architecture`；添加工具见 :doc:`add_primitive`。

ACP CLI 适配器
--------------

接入支持 ACP 的 CLI 时，继承 ``rpent/planner/utils/acp.py`` 中的 ``AcpPlanner``，设置 ``backend`` 和显示名称 ``label``。实现 ``_create_session``，准备命令、环境与工具策略，并返回 ``AcpSession``。参考 ``gemini_cli.py`` 和 ``opencode.py``：两者均通过 ACP 管理会话，通过 HTTP MCP 调用 RPent 工具。适配器以 CLI 的名称注册。

基类处理会话、事件记录、连接探测、消息排队、中断及进程清理。OpenCode 通过 ``AcpSession`` 传入 MCP 地址，Gemini CLI 则写入工作目录配置。使用 CLI 的 ACP 模型选择器时，才向 ``AcpSession`` 传入模型 ID。适配器负责认证和原生轮数限制；ACP 没有统一的模型轮数上限。

接入与验证
---------------

在 ``rpent/planner/base.py`` 的 ``build_planner`` 中添加构建分支。在线后端还需在 ``rpent/planner/check.py`` 中加入 ``CHECK_PLANNERS``，设置诊断超时并接入连接探测；CLI 从该列表生成在线规划器选项。自行配置模型服务的 CLI/SDK 规划器还需加入 ``EXTERNALLY_CONFIGURED_PLANNERS``。Dashboard 会话使用命令行选定的规划器。

接入 ACP 适配器时，将名称加入 ``ACP_PLANNERS``，并在 ``build_planner`` 和 ``_check_acp`` 中选择对应的类。共享列表统一用于在线规划器注册、外部配置处理以及仅接受 ``--reasoning-effort none`` 的校验。

验证工具调用、文本和图像返回、真实的 ``finish`` 调用、轮数上限、错误结果、消息排队、中断及进程清理。复用 ``tests/unit_tests/rpent/planner/`` 中的测试模式；真实模型连接需要单独验证。
