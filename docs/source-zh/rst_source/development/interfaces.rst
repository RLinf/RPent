核心接口
========

给新机器人或新 primitive 接入 RPent 时，需要对接的接口如下。具体操作见
:doc:`add_robot`、:doc:`add_primitive`；仓库分层见 :doc:`architecture`。

机器人入口
----------

把包放到 ``robots/<robot>/`` 后，包的 ``__init__.py`` 会重导出
``robot_spec.py`` 中实现的两个函数，供 ``main.py`` 调用：

.. code-block:: python

   def get_robot_spec() -> RobotSpec: ...
   def get_toolkit(
       *,
       runtime_kwargs,
       dashboard_events: DashboardEventSink,
       config: RunConfig,
   ): ...

``get_robot_spec`` 返回 ``RobotSpec``，其中你需要提供：

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - 字段或钩子
     - 你要做什么
   * - ``name``
     - 机器人名，对应 ``--robot``。
   * - ``prompts``
     - ``PromptBundle``：``system`` 与 ``user`` 两套 prompt 工厂（见
       ``robots/<robot>/prompt_bundle.py``）。
   * - ``dashboard``
     - 可选的 Dashboard 描述。设为 ``None`` 时，该机器人不支持 Dashboard 控制；
       否则由该 spec 定义任务命令与字段、runtime components 和 frame channels。
   * - ``add_cli_args``
     - 注册本机器人的 CLI 参数（如 ``--suite``、``--env-endpoint``）。
   * - ``parse_config``
     - 校验参数并返回 ``RunConfig``；``recipe_tag``、``output_dir``、``prompt_vars``
       三项需由你正确填写（供 prompt 模板插值）。
   * - ``init_runtime``
     - 启动或连接全部 runtime components，或只处理指定名称的子集，并构造对应的
       ``runtime_kwargs``。普通 CLI 传 ``None``；Dashboard 从 spec 得到显式声明
       的 shared 和 unique 子集后分别传入。``DashboardEventSink`` 用于上报运行时状态。

``get_toolkit`` 用 ``runtime_kwargs`` 中的客户端构造机器人 toolkit；
运行配置 ``config`` 和 Dashboard 事件接收器 ``dashboard_events`` 也由 runner 提供。它需要构造一个
:class:`~rpent.memory.MemoryManager`（root 取自
``config.prompt_vars["memory_dir"]``，未设置时回退到
``get_memory_dir(robot_name)``）并传给 toolkit。Memory 访问权限在
``MemoryManager`` 上配置。如果某个机器人还需要额外参数，可以继续声明
keyword-only 参数；例如 LIBERO 还使用 ``mode``、``attempts_per_session`` 和
``state_output_dir``。

参考实现：``robots/libero/robot_spec.py``。

Planner
-------

多数用户用内置 ``api``、``claude_code``、``codex``，见
:doc:`../usage/configure_planner`。自定义 planner 才需实现
``rpent.planner.base.Planner``：

.. code-block:: python

   def solve(
       self,
       *,
       system_prompt: str,
       user_message: str,
       toolkit: Toolkit,
       max_turns: int,
       input_queue=None,
       dashboard_interaction=None,
   ) -> PlannerResult: ...

Planner 先通过 ``toolkit.list_tools()`` 获取工具定义，将每个工具的 ``name``、
``description`` 和 ``input_schema`` 交给模型。模型发起调用后，使用
``toolkit.execute_tool(name, arguments)`` 执行，再将返回的文本和图片送回模型，
继续下一轮推理。当 ``toolkit.finish_result`` 有值，或达到运行限制时，返回
``PlannerResult``。

如果 planner 使用异步调用，可以通过 ``rpent.planner.base.execute_tool``
在线程中执行工具。这个辅助函数也会处理取消，等待工具完成清理后再退出。

工具集
------

在 ``robots/<robot>/toolkit.py`` 中继承 ``Toolkit``，构造时传入机器人使用的
客户端、状态和工具集合：

.. code-block:: python

   super().__init__(
       state=state,
       memory=memory,
       robot=runtime,
       output_dir=output_dir,
       tools=MYROBOT_TOOLS,
       dashboard_events=dashboard_events,
   )

``MYROBOT_TOOLS`` 是由 ``@tool`` 声明组成的元组。编写工具时，主要需要了解
以下三部分，它们都可以从 ``rpent.tools`` 导入：

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - 接口
     - 用法
   * - ``@tool``
     - 将函数声明为工具。函数名就是工具名，Google 风格 docstring 提供说明，
       参数类型和 ``Field`` 约束用于生成校验模型及 JSON schema。
   * - ``ToolContext``
     - 工具通过必填的仅限关键字参数 ``ctx`` 获取上下文。其中 ``robot`` 是
       运行时对象，``state`` 和 ``memory`` 分别管理观测与记忆，``output_dir``
       指向输出目录。``ctx`` 由 toolkit 提供，不出现在模型可见的参数中。
   * - ``ToolResult``
     - 工具函数的返回值。``data`` 保存执行结果，``images`` 保存 PNG 字节，
       出错时填写 ``error``。Planner 使用 ``to_text()`` 和 ``images`` 读取
       文本与图片，通过 ``is_error`` 判断调用是否失败。

基类会自动加入公共文件工具，文件访问权限由 ``MemoryManager`` 检查。
其中 ``read_image`` 供 API planner 使用；Claude Code 和 Codex 使用各自内置的
图片读取工具。工具函数的完整示例见 :doc:`add_primitive`。

默认情况下，工具独占执行，完成后由机器人子类的 ``_capture_observation``
保存状态并返回新的观测。观测会替换动作返回的数据，因此需要保留的执行详情
应写入观测中的日志；动作错误仍会保留。即使工具函数出错，toolkit 也会尝试
捕获观测，让 planner 了解当前环境。

读取已有观测的工具可以在 ``@tool`` 下方添加 ``@readonly``，跳过自动捕获。
每个 toolkit 同时只允许一个调用；重叠的直接调用会返回错误。API 和 MCP
适配器按顺序执行工具调用。

公共工具和 ``finish`` 不触发观测捕获。``write_text_file`` 和 ``finish``
不设置 readonly，因此独占执行但不新增观测。LIBERO 的 ``segment`` 也不设置
readonly，独占执行，并在保存分割附件后自动捕获一次观测。

长时间运行的工具应在安全的动作边界调用 ``ctx.check_cancelled()``。收到中断后，
``cancel_active_and_wait()`` 向当前调用发送取消信号，并等待它退出。后续调用
使用新的取消信号。工具通过 ``ctx.record_frame`` 提交录像帧；机器人 toolkit
在捕获观测时保存动作片段，并重写 ``close()`` 保存回合录像。

每个机器人提供自己的 ``finish`` 工具。调用成功后，toolkit 将其中的 ``status``
和 ``summary`` 保存到 ``finish_result``，供 planner 结束循环；环境是否真正成功，
则由 ``solved()`` 判断。Recipe 的导出由 ``write_recipe(recipe_tag)`` 完成，
memory 的使用与发布见 :doc:`memory`。

进程间通信
----------

接已有server或写 ``env_server`` / ``vla_server`` 时关注下面两点。

客户端端点（在 ``add_cli_args`` 中暴露，并在适用的普通 CLI 或 Dashboard
runtime 钩子中解析）：

.. code-block:: text

   [protocol://]host:port    # 未写协议时默认为 http

常见：``--env-endpoint``、``--vla-endpoint``。``http`` 为默认，走 ``POST /call``
传 JSON，其中 NumPy 数组编码为 ``{"__ndarray__": <base64>, "dtype": ..., "shape": ...}``；
NumPy 标量编码为 ``{"__npscalar__": <value>, "dtype": ...}`` 以保留精确 dtype；
观测数据很大、或是多帧堆叠的嵌套 NumPy 字典时可改 ``socket``，用带长度前缀的
pickle 数据帧传输，省掉反复的 JSON 编解码。pickle 不适合不可信输入，socket 只应连接可信端点。

环境和 VLA client 通常应分别继承 ``BaseEnvClient``、``BaseVLAClient``；服务端
分别继承 ``BaseEnvFacade``、``BaseVLAFacade``，并通过 ``_register_rpc`` 注册
扩展路由。这些基类在 ``RpcFacade`` 之上提供公共路由和锁。只有尚无专用基类的
服务类型才直接继承 ``RpcFacade``。业务子类不必实现 ``healthz`` / ``shutdown``。

细节见 :doc:`add_robot` 中的 env_server 与 vla_server 章节。
