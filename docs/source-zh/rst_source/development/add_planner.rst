添加规划器
===============

规划器负责接收提示词、选择工具并处理工具结果。接入自定义规划器时，继承 ``Planner`` 并实现 ``solve``，然后将其加入构建入口和 CLI 选项。

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
5. 返回包含结束状态、消息、统计信息，以及可选 ``error`` 和 ``stop_reason`` 字段的 ``PlannerResult``。

规划器可通过 ``stop_info`` 提供原始 ``reason`` 和公共 ``category``：
``finish``、``budget``、``timeout``、``cancelled``、``error`` 或 ``unknown``。
``normalize_stop_info`` 保留后端原始原因，并为通用原因补充分组；未知原因仍归为 ``unknown``。
结果写入方应优先使用 ``stop_info.reason``，其次是 ``stop_reason``，最后兼容 ``stats`` 中的旧原因字段。
执行是否确定、结果是否交付、清理是否完成，以及是否允许开始新会话，应分别记录。
开始新会话要求 ``execution_certain``、``result_delivery_complete``、
``cleanup_complete`` 和 ``allow_new_session`` 均明确为 ``True``；
缺少证据或仅凭分组不能允许继续执行。现有规划器可以不提供 ``stop_info``，保留原有恢复行为。

``get_planner_capabilities(name)`` 提供无需导入可选 SDK 的静态运行要求。
``PlannerCapabilities`` 的三个标志默认均为 false：探索时的 ``requires_memory_copy``、
手动动作后恢复前的 ``refresh_manual_observations``，以及排队消息撤回能力 ``withdraw_queued``。
接入后端时，应声明其要求，并在相应运行入口读取和落实这些标志；仅声明不会自动启用行为。

由于 RPent 工具定义和 prompt 渲染流程保持不变，新增 planner 不需要修改工具或环境服务。接口参见 :doc:`architecture`；想给
自定义 planner 暴露新工具，见 :doc:`add_primitive`。

接入与验证
---------------

在 ``rpent/planner/base.py`` 的 ``build_planner`` 中添加构建分支，并在 ``rpent/cli/main.py`` 的 ``--planner`` 选项中加入名称。需要 Dashboard 或连接诊断时，还应接入对应的规划器选择与 ``rpent/planner/check.py``，不能仅添加一个类。

验证工具调用、文本和图像返回、``finish``、轮数上限、错误结果，以及需要支持的中断和交互路径。复用 ``tests/unit_tests/rpent/planner/`` 中的测试模式；真实模型连接需要单独验证。
