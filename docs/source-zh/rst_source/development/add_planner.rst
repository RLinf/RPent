添加规划器
===============

规划器负责接收提示词、选择工具并处理工具结果。

ACP Harness
------------

对于支持 stdio 上的 ACP v1 和 HTTP MCP 的 harness，使用 ``--planner acp``
和 ``RPENT_ACP_COMMAND`` 即可，无需新增 planner 类或注册项。
配置方法与限制见 :ref:`planner-acp`。

``rpent/planner/acp.py`` 包含共享 planner、ACP 会话驱动和事件记录器，并复用
``utils/agent_session.py`` 的交互控制、HTTP MCP、日志与清理逻辑。
``gemini_cli.py`` 继承 ``AcpPlanner``，只负责 Gemini 的命令、环境和工作目录配置。

其他 harness 的准备工作可放在预先配置的启动脚本中。若需要 RPent 管理这些
设置，可参考 Gemini 的小型 ``_create_session`` 覆盖方法；会话通信和事件处理
仍留在公共驱动中。模型选择器是可选能力，任务执行则必须支持 HTTP MCP。
使用图像观测时，harness 还需能处理 RPent 工具返回的 MCP 图像结果。

接入新 harness 时，应验证文本和图像工具返回、真实 ``finish`` 调用、消息排队、
中断及进程清理。连接检查成功不代表这些行为都已验证。

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
4. 识别 ``ToolResult.data.get("_finish")``，并按支持的执行限制终止循环。ACP harness 的模型轮次上限由自身配置，RPent 负责总运行超时。
5. 返回包含结束状态、消息、统计信息和可选错误的 ``PlannerResult``。

由于 RPent 工具定义和 prompt 渲染流程保持不变，新增 planner 不需要修改工具或环境服务。接口参见 :doc:`architecture`；想给
自定义 planner 暴露新工具，见 :doc:`add_primitive`。

接入与验证
---------------

在 ``rpent/planner/base.py`` 的 ``build_planner`` 中添加构建分支，并在 ``rpent/cli/main.py`` 的 ``--planner`` 选项中加入名称。需要 Dashboard 或连接诊断时，还应接入对应的规划器选择与 ``rpent/planner/check.py``，不能仅添加一个类。

验证工具调用、文本和图像返回、``finish``、轮数上限、错误结果，以及需要支持的中断和交互路径。复用 ``tests/unit_tests/rpent/planner/`` 中的测试模式；真实模型连接需要单独验证。
