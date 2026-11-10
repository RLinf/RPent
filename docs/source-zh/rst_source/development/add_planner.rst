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
5. 返回包含结束状态、消息、统计信息和可选错误的 ``PlannerResult``。

由于 RPent 工具定义和 prompt 渲染流程保持不变，新增 planner 不需要修改工具或环境服务。接口参见 :doc:`architecture`；想给
自定义 planner 暴露新工具，见 :doc:`add_primitive`。

接入与验证
---------------

在 ``rpent/planner/base.py`` 的 ``build_planner`` 中添加构建分支，并在 ``rpent/cli/main.py`` 的 ``--planner`` 选项中加入名称。需要 Dashboard 或连接诊断时，还应接入对应的规划器选择与 ``rpent/planner/check.py``，不能仅添加一个类。

验证工具调用、文本和图像返回、``finish``、轮数上限、错误结果，以及需要支持的中断和交互路径。复用 ``tests/unit_tests/rpent/planner/`` 中的测试模式；真实模型连接需要单独验证。

.. _deepseek-runtime-bridge:

DeepSeek Harness 运行时桥接
--------------------------------------

固定版本的 Harness SDK 可通过 ``HttpMcpServer(..., include_structured=True)`` 使用 RPent 工具。
这个显式启用的结构化结果保留超出文本截断上限的字段，将有限负零规范为零以满足 Harness 的 JSON 约束，
并拒绝非有限数值。现有 MCP 消费方默认仍接收原有的文本和图像表示。
``ToolResult.execution_uncertain`` 是内部执行来源信息：处理函数或观测失败前，环境可能已经发生变化。

``deepseek_tools._ToolBoundary`` 串行执行原生工具，保留实际接受的 ``finish`` 和结果交付依据，
并在执行结果不确定时关闭后续调用入口。 ``deepseek_bridge.mjs`` 在运行时内约束同一边界：
只允许明确列出的 RPent 工具、串行执行、完整且不插值的系统提示、准确的根模型步骤预算，
以及根请求和摘要请求一致的模型、提供方与推理强度。
它在允许下一动作前校验工具最终结果是否交付，以及图像数量和像素尺寸是否保持一致。
这些检查本身不证明任务成功或物理动作安全。

调用方必须关闭默认的非 RPent 工具提供方和额外 Agent，将 ``agent-loop.maxParallelToolCalls``
设为 ``1``，并把 ``rpentBridgeReady`` 加入 ``sdk-jsonrpc-server`` 的启动依赖。
调用方还负责取消、等待 Toolkit 活动调用结束、回收 SDK 进程和解释最终控制记录。
桥接组件不提供 host 控制通道或持久化会话恢复。
``tests/e2e_tests/planner/test_deepseek_foundation.py`` 展示了直接组合 SDK 与 MCP 的方式，
使用确定性的本地模型验证启动、结果交付、图像、上下文压缩和资源清理。
