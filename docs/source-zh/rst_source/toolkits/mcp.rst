.. _external-mcp-servers:

外部 MCP 工具
=============

通过 ``--mcp-config`` 传入 YAML 配置文件，即可将外部 Streamable HTTP MCP 服务的工具注册到当前机器人的 ``toolkit`` 中。``api``、``claude_code``、``codex`` 和 ``gemini_cli`` 共用同一套工具过滤、超时控制、取消机制与轨迹记录。

前置条件：

- 外部 MCP 服务提供 Streamable HTTP 端点（``http://`` 或 ``https://``）。RPent 不支持通过 ``command`` 启动 stdio 服务。
- 已完成对应机器人与规划器的基础配置（见 :doc:`../get_started/quickstart`\ ）。

快速开始
--------

1. 启动本地测试服务。将下面的脚本保存为 ``power_meter.py``：

   .. dropdown:: 示例服务端（``power_meter.py``）

      .. code-block:: python

         from mcp.server.fastmcp import FastMCP

         mcp = FastMCP("power-meter", host="127.0.0.1", port=8765)


         @mcp.tool()
         def read_power_meter() -> str:
             """Return the current power meter reading."""
             return "battery 82%, outlet idle"


         if __name__ == "__main__":
             mcp.run(transport="streamable-http")

   RPent 依赖中已包含 ``mcp``。运行脚本启动服务（默认监听 ``http://127.0.0.1:8765/mcp``）：

   .. code-block:: bash

      python power_meter.py

2. 创建 ``power.yaml`` 指定服务地址。未设置 ``tools`` 时默认开放该服务的全部工具：

   .. code-block:: yaml

      mcp_servers:
        power:
          url: http://127.0.0.1:8765/mcp

3. 启动任务时传入 ``--mcp-config``：

   .. code-block:: bash

      rpent --robot libero --mcp-config power.yaml \
        --suite libero_goal_task --task 1 --seed 0 --planner claude_code

任务启动后，RPent 会连接 ``power`` 服务，并在 ``toolkit`` 中注册 ``power__read_power_meter``。模型调用该工具后会收到 ``battery 82%, outlet idle``。

配置项与认证
------------

``mcp_servers`` 下每个服务支持以下字段：

.. list-table::
   :header-rows: 1
   :widths: 22 10 68

   * - 字段
     - 必填
     - 说明
   * - ``url``
     - 是
     - Streamable HTTP 端点（``http://`` 或 ``https://``）。URL 中不能包含用户名或密码。
   * - ``tools``
     - 否
     - 要开放的工具范围：``"*"``\ （默认，开放服务端全部工具）或工具名列表。所有规划器都只会看到这里列出的工具。
   * - ``headers``
     - 否
     - 每次调用附带的 HTTP 请求头。加载配置文件时会解析 ``${oc.env:VAR_NAME}``。
   * - ``sync_with_robot``
     - 否
     - ``false``\ （默认）或 ``true``，控制是否与机器人动作串行执行并采集观测（见下节）。

请求头中的凭据仅保留在 RPent 进程内，不会出现在任何子进程命令行中；写入任务记录时，请求头取值以及 URL 中的查询参数与锚点都会自动脱敏。

对接需要认证、且只开放部分工具的服务示例：

.. code-block:: yaml

   mcp_servers:
     power:
       url: https://power.example.com/mcp
       tools: [read_power_meter]
       headers:
         Authorization: "Bearer ${oc.env:POWER_API_TOKEN}"

运行前先导出 ``POWER_API_TOKEN`` 环境变量。若该变量未设置，RPent 会在启动机器人服务之前直接报配置错误并退出。

与机器人动作同步
----------------

默认配置（``sync_with_robot: false``）下，外部 MCP 工具按只读工具执行：可与其他只读工具并发、不触发机器人状态采集，直接将远端返回的文本、结构化数据和 PNG 图像交给规划器（仍受 ``toolkit`` 读写锁约束，不会与机器人动作同时执行；耗时较长的外部调用也会延后排队的机器人动作）。

如果外部服务控制的设备会影响真实场景或机器人安全（例如充电底座、安全互锁或灯光），将 ``sync_with_robot`` 设为 ``true``：

.. code-block:: yaml

   mcp_servers:
     dock:
       url: http://127.0.0.1:8766/mcp
       tools: [set_charging_dock]
       sync_with_robot: true

启用 ``sync_with_robot: true`` 后：

- 工具调用会按顺序与机器人动作独占执行，避免在机械臂运动过程中触发外部设备。
- 每次调用完成后，RPent 会采集一次最新的机器人观测，并将远端返回内容保留在观测结果的 ``tool_result`` 字段中。
- 服务端返回的 ``image/png`` 图像会与机器人相机观测图像一并返回给模型。

如果同一个服务同时提供只读工具和需要与机器人同步的设备控制工具，可以在 ``mcp_servers`` 下声明两个不同名称的条目指向同一个 ``url``\ （分别配置 ``tools`` 与 ``sync_with_robot``），或在机器人接入代码中分组注册（详见 :ref:`add-robot-toolkit`\ ）。

工具命名、超时与内容限制
------------------------

- **命名规则：** 每个外部工具在 ``toolkit`` 中的名称为 ``<server>__<tool>``\ （如 ``power__read_power_meter``）。各规划器在此基础上附加各自的 ``toolkit`` 前缀：``api`` 直接使用 ``power__read_power_meter``，``claude_code`` 与 ``codex`` 为 ``mcp__rpent__power__read_power_meter``，``gemini_cli`` 为 ``mcp_rpent_power__read_power_meter``。服务名必须以字母或数字开头，其余字符仅限字母、数字、``_`` 或 ``-``，不能包含 ``__``，不能以 ``_`` 结尾，且不能为 ``rpent``。
- **启动校验与工具白名单：** 构建 ``toolkit`` 时，RPent 会连接每个服务并校验 ``tools`` 中列出的工具是否存在。服务不可达或工具名缺失会终止启动，并在错误信息中列出服务端实际提供的工具。省略 ``tools``\ （或设为 ``"*"``）会在所有规划器中开放并自动批准该服务的全部工具；若服务包含有副作用或管理类工具，建议在 ``tools`` 中显式列出允许调用的工具名。
- **无状态会话与静态发现：** 每次调用都会建立新的无状态 MCP 会话，不会跨调用保留服务端会话状态。RPent 仅开放启动时发现的 MCP 工具，不转发 Resources、Prompts 或动态的 ``tools/list_changed`` 通知。
- **超时与取消：** 单次 MCP 请求默认 60 秒超时，可通过环境变量 ``MCP_CALL_TIMEOUT_S``\ （单位为秒）调整。取消任务或中断 Dashboard 运行时，正在执行的 MCP 调用会立即中止，不会等待远端执行结束。
- **内容转发：** 文本、结构化 JSON 和 ``image/png`` 图像块会转发给模型，文本长度受 RPent 工具输出字节上限约束；其他媒体或资源块只保留元数据，启用 ``--no-images``\ （仅 ``api``）时同样会省略外部 PNG 图像。
