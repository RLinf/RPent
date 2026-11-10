添加规划器
===============

规划器负责接收提示词、选择工具并处理工具结果。

ACP Harness
------------

ACP 是共享驱动，不是可选的规划器。接入 ACP harness 时，继承
``rpent/planner/utils/acp.py`` 中的 ``AcpPlanner``，设置 ``backend`` 和显示名称
``label``。实现 ``_create_session``，准备命令、环境与工具策略，并返回
``AcpSession``。标准 HTTP MCP 接入可参考 ``opencode.py``，工作目录配置可参考
``gemini_cli.py``。按下方步骤以 harness 名称注册适配器。

基类统一处理会话、事件记录和连接探测，并复用 ``utils/agent_session.py`` 的交互控制、
HTTP MCP、日志与清理逻辑。对声明支持 HTTP MCP 的 agent，将 MCP URL 传给
``AcpSession``；Gemini 则通过工作目录配置传入。仅在使用 agent 的 ACP 模型选择器时，
向会话传入模型 ID。认证和原生轮次限制由 harness 适配器负责，ACP 没有统一的模型轮次上限。
除连接检查外，还应验证文本和图像工具返回、真实 ``finish`` 调用、消息排队、
中断及进程清理。

自定义规划器
--------------

对于其他执行协议，继承 ``Planner`` 并实现 ``solve``，然后将其加入构建入口和 CLI 选项。

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

任何 planner 必须：

1. 接收已经渲染好的 ``system_prompt`` 和 ``user_message``。
2. 从 ``toolkit.list_tools()`` 取得工具的 ``name``、``description`` 和 ``input_schema``，并通过 ``toolkit.execute_tool(name, arguments)`` 执行工具。
3. 将 ``ToolResult.to_text()`` 的文本和 ``ToolResult.images`` 中的 PNG 字节转换成模型 SDK 所需的格式。
4. 识别 ``ToolResult.data.get("_finish")``，并按 ``max_turns`` 等限制终止循环。
5. 返回包含结束状态、消息、统计信息和可选错误的 ``PlannerResult``。

由于 RPent 工具定义和 prompt 渲染流程保持不变，新增 planner 不需要修改工具或环境服务。接口参见 :doc:`architecture`；想给
自定义 planner 暴露新工具，见 :doc:`add_primitive`。

接入与验证
---------------

在 ``rpent/planner/base.py`` 的 ``build_planner`` 中添加构建分支。在线后端还需在 ``rpent/planner/check.py`` 中加入 ``CHECK_PLANNERS``，设置诊断超时并接入连接探测；CLI 从该列表生成在线规划器选项。由 harness 自行配置模型服务的后端，还需加入 ``EXTERNALLY_CONFIGURED_PLANNERS``。Dashboard 会话使用命令行选定的规划器。

验证工具调用、文本和图像返回、``finish``、轮数上限、错误结果，以及需要支持的中断和交互路径。复用 ``tests/unit_tests/rpent/planner/`` 中的测试模式；真实模型连接需要单独验证。
