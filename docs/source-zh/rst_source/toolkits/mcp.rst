.. _external-mcp-servers:

外部 MCP 工具
=============

通过 ``--mcp-config`` 传入 YAML 配置文件，可将 Streamable HTTP MCP 服务接入 RPent，使规划器在调用机器人动作的同时访问外部传感器、实验仪器或其他外部设备。同一份配置适用于 ``api``、``claude_code``、``codex`` 和 ``gemini_cli``。

前置条件：

- 外部 MCP 服务提供 Streamable HTTP 端点（``http://`` 或 ``https://``）。RPent 不支持通过 ``command`` 启动 stdio 服务；如果现有服务只提供 stdio 接口，可先用 ``mcp-proxy``、``supergateway`` 等代理将其转为 Streamable HTTP 端点，再将地址填入 ``url``。
- 已完成对应机器人与规划器的基础配置（见 :doc:`../get_started/quickstart`\ ）。

快速开始
--------

本示例启动一个本地模拟功率计 MCP 服务（提供 ``read_power_meter`` 工具），通过 ``--mcp-config power.yaml`` 接入 RPent，并在终端交互模式（``--tui``）下验证规划器调用该工具。

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

   运行脚本启动服务（默认监听 ``http://127.0.0.1:8765/mcp``）：

   .. code-block:: bash

      python power_meter.py

2. 创建 ``power.yaml`` 指定服务地址。未设置 ``tools`` 时默认开放该服务的全部工具：

   .. code-block:: yaml

      mcp_servers:
        power:
          url: http://127.0.0.1:8765/mcp

3. 启动交互式任务并通过 ``--mcp-config`` 传入 YAML 文件路径（可将 ``--robot libero`` 与 ``--planner claude_code`` 替换为目标机器人和在线规划器）：

   .. code-block:: bash

      rpent --robot libero --mcp-config power.yaml \
        --suite libero_goal_task --task 1 --seed 0 --planner claude_code --tui

任务启动后，RPent 会连接 ``power`` 服务并注册 ``power__read_power_meter``。在终端输入指令让规划器读取功率计（例如 ``Read the power meter``），即可看到它调用 ``power__read_power_meter`` 并返回 ``battery 82%, outlet idle``，调用记录也会写入输出目录的 ``transcript_*.json``\ （见 :ref:`run-output-files`\ ）。如果将 ``--tui`` 换为 ``--dashboard``，也可在 Dashboard 界面中查看该工具及调用结果。

如果服务连不上，或配置了服务端不存在的工具，RPent 会在规划开始前退出，并在终端和 ``run.log`` 中打印报错信息以及服务端实际提供的工具列表。

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
   * - ``timeout_s``
     - 否
     - 单次工具发现与调用的超时时间（正数，单位为秒），优先于环境变量 ``MCP_CALL_TIMEOUT_S`` 和默认 60 秒。

.. warning::

   省略 ``tools`` 或设为 ``"*"`` 时，服务端的所有工具都会直接开放给规划器，调用前不会额外确认。如果服务包含写操作、硬件控制或管理类接口，建议在 ``tools`` 中显式列出允许调用的工具名。

下面的示例演示如何配置认证请求头、只开放指定工具并自定义超时时间：

.. code-block:: yaml

   mcp_servers:
     power:
       url: https://power.example.com/mcp
       tools: [read_power_meter]
       timeout_s: 30
       headers:
         Authorization: "Bearer ${oc.env:POWER_API_TOKEN}"

运行前先导出 ``POWER_API_TOKEN`` 环境变量。若该变量未设置，RPent 会在启动机器人服务之前直接报配置错误并退出。

与机器人动作同步
----------------

默认（``sync_with_robot: false``）按只读工具执行：可与其他只读工具并发，不触发机器人状态采集，并直接将远端返回的文本、结构化数据和 PNG 图像传给规划器。这类调用仍受 ``toolkit`` 读写锁约束，不会与机器人动作并发执行；如果远端耗时较长，排队的机器人动作会等待其完成。

如果外部服务控制的设备会影响真实场景或机器人安全（例如充电底座、安全互锁或灯光），将 ``sync_with_robot`` 设为 ``true``：

.. code-block:: yaml

   mcp_servers:
     dock:
       url: http://127.0.0.1:8766/mcp
       tools: [set_charging_dock]
       sync_with_robot: true

启用 ``sync_with_robot: true`` 后：

- 与机器人动作互斥执行，避免机械臂运动时误触发外部设备。
- 每次调用完成后，RPent 会采集一次最新的机器人观测，并将远端返回内容与更新后的状态和相机画面一并交给规划器。
- 服务端返回的 ``image/png`` 图像会与机器人相机观测图像一并返回给模型。

如果同一个服务同时提供只读工具和需要与机器人同步的控制工具，可在 ``mcp_servers`` 下写两条不同名称的配置指向同一个 ``url``，分别指定各自的 ``tools`` 和 ``sync_with_robot``。

参考细节
--------

外部工具默认以 ``<server>__<tool>`` 命名，单次请求超时按 YAML 中的 ``timeout_s``、环境变量 ``MCP_CALL_TIMEOUT_S``、默认 60 秒依次生效，并复用 RPent 的取消机制与轨迹记录。

.. dropdown:: 工具命名与规划器前缀

   - 每个外部工具在 ``toolkit`` 中的名称为 ``<server>__<tool>``\ （如 ``power__read_power_meter``）。
   - 各规划器在此基础上附加各自的 ``toolkit`` 前缀：``api`` 直接使用 ``power__read_power_meter``，``claude_code`` 与 ``codex`` 为 ``mcp__rpent__power__read_power_meter``，``gemini_cli`` 为 ``mcp_rpent_power__read_power_meter``。
   - 服务名必须以字母或数字开头，其余字符仅限字母、数字、``_`` 或 ``-``，不能包含 ``__``，不能以 ``_`` 结尾，且不能为 ``rpent``。

.. dropdown:: 启动校验、配置合并与会话限制

   - **共享配置与机器人配置合并：** CLI 会合并 ``--mcp-config`` 所指定 YAML 文件中的服务和机器人 ``RunConfig.mcp_servers`` 自带的服务，再统一注册到 ``toolkit``。两处来源的服务名不能重复，重名会在启动时报错。
   - **启动校验：** 构建 ``toolkit`` 时，RPent 会连接每个服务并校验 ``tools`` 中列出的工具是否存在。服务不可达或工具名缺失会终止启动，并在错误信息中列出服务端实际提供的工具。
   - **无状态会话与静态发现：** 每次调用都会建立新的无状态 MCP 会话，不会跨调用保留服务端会话状态。RPent 仅开放启动时发现的 MCP 工具，不转发 Resources、Prompts 或动态的 ``tools/list_changed`` 通知。

.. dropdown:: 超时、内容限制与脱敏

   - **超时与取消：** 单次 MCP 请求的超时时间按 YAML 中的 ``timeout_s``\ （按服务配置）、环境变量 ``MCP_CALL_TIMEOUT_S``\ （全局，单位为秒）、默认 60 秒依次生效。取消任务或中断 Dashboard 运行时，正在执行的 MCP 调用会立即中止，不会等待远端执行结束。
   - **内容转发与大小上限：** 文本、结构化 JSON 和 ``image/png`` 图像块会转发给模型，其中文本输出受 RPent 单次工具输出 60,000 字节上限约束；其他媒体或资源块只保留元数据，启用 ``--no-images``\ （仅 ``api``）时同样会省略外部 PNG 图像。
   - **凭据处理与脱敏：** 请求头中的凭据仅保留在 RPent 进程内，不会出现在任何子进程命令行中；写入 ``task.json`` 和 Dashboard 任务记录时，请求头取值以及 URL 中的查询参数与锚点都会自动脱敏。

如需在机器人接入代码中显式控制 MCP 工具注册，见 :ref:`add-robot-toolkit`。
