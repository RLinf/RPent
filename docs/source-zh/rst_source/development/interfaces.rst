核心接口
============

将新机器人或动作原语接入 RPent 时，需要实现以下接口。具体操作见 :doc:`add_robot`、:doc:`add_primitive`；代码目录与模块职责见 :doc:`architecture`。

机器人入口
----------

把包放到 ``robots/<robot>/`` 后，包的 ``__init__.py`` 会重导出 ``robot_spec.py`` 中实现的两个函数，供 ``main.py`` 调用：

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
       否则由该配置定义任务命令与字段、运行组件和图像通道。
   * - ``add_cli_args``
     - 注册本机器人的 CLI 参数（如 ``--suite``、``--env-endpoint``）。
   * - ``parse_config``
     - 校验参数并返回 ``RunConfig``；``recipe_tag``、``output_dir``、``prompt_vars``
       三项需由你正确填写（供 prompt 模板插值）。
   * - ``init_runtime``
     - 启动或连接全部 runtime components，或只处理指定名称的子集，并构造对应的
       ``runtime_kwargs``。普通 CLI 传 ``None``；Dashboard 从 spec 得到显式声明的 shared 和 unique 子集后分别传入。``DashboardEventSink`` 用于上报运行时状态。

``get_toolkit`` 一般只需把 ``runtime_kwargs`` 传给机器人子类； ``dashboard_events`` 和 ``config`` 由当前 runner 传入。它需要构造一个
:class:`~rpent.memory.MemoryManager`（root 取自
``config.prompt_vars["memory_dir"]``，未设置时回退到 ``get_memory_dir(robot_name)``）并传给 toolkit。Memory 访问权限在 ``MemoryManager`` 上配置。如果某个机器人还需要额外参数，可以继续声明 keyword-only 参数；例如 LIBERO 还使用 ``mode``、``attempts_per_session`` 和 ``state_output_dir``。

参考实现：``robots/libero/robot_spec.py``。

Planner
-------

多数用户用内置 ``api``、``claude_code``、``codex``，见 :doc:`../guides/configure_planner`。自定义 planner 才需实现
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

规划器通过 ``toolkit.list_tools()`` 获取工具定义并传给模型，通过 ``toolkit.execute_tool(name, input_dict)`` 执行模型发出的工具调用，再将工具返回结果传给模型。调用 ``finish`` 或达到轮数上限时，返回 ``PlannerResult``。

工具集
------

在 ``robots/<robot>/toolkit.py`` 里继承 ``Toolkit``，用 ``add_tool`` 注册机器人工具：

.. code-block:: python

   def add_tool(self, tool: Tool, *, replace: bool = False) -> None: ...

用 ``@tool`` 声明函数或方法，参数类型注解和 Google 风格 docstring 提供 schema 与说明；``@tool(readonly=True)`` 允许只读工具并行执行，并跳过动作后的状态采集。处理函数返回 ``ToolResult(data=..., images=...)``：``data`` 保存 JSON 数据，按需包含 ``error`` 或 ``_finish``；``images`` 保存 PNG 字节。

基类已注册公共文件工具；子类 ``super().__init__()`` 后追加本机器人工具即可。逐步状态与 ``view_env_state`` 见 :doc:`add_primitive`。

未设置 ``readonly=True`` 的工具按进入调度队列的顺序独占执行。等待中的独占调用优先于新的只读调用，独占范围包括观测采集和 Dashboard 发布。只读工具如果会更新共享缓存，应自行同步这些更新。``EnvState.save`` 会串行保存附件。

``cancel_active_and_wait()`` 暂停接收新调用，向排队中和运行中的调用发送取消信号，并等待清理完成。长时间运行的处理函数在安全边界检查 ``raise_if_cancelled()``。规划器也排空旧请求后，通过 ``resume_calls()`` 恢复接收调用。``close()`` 永久关闭调用入口；子类应先调用 ``super().close()``，再保存录像或释放资源。

进程间通信
----------

连接已启动的服务器，或编写 ``env_server`` / ``vla_server`` 服务端程序时，需要遵循以下通信约定。

客户端端点（在 ``add_cli_args`` 中暴露，并在适用的普通 CLI 或 Dashboard runtime 钩子中解析）：

.. code-block:: text

   [protocol://]host:port    # 未写协议时默认为 http

常见：``--env-endpoint``、``--vla-endpoint``。``http`` 为默认，走 ``POST /call`` 传 JSON，其中 NumPy 数组编码为 ``{"__ndarray__": <base64>, "dtype": ..., "shape": ...}``； NumPy 标量编码为 ``{"__npscalar__": <value>, "dtype": ...}`` 以保留精确 dtype；观测数据很大、或是多帧堆叠的嵌套 NumPy 字典时可改 ``socket``，用带长度前缀的 pickle 数据帧传输，省掉反复的 JSON 编解码。pickle 不适合不可信输入，socket 只应连接可信端点。

环境和 VLA client 通常应分别继承 ``BaseEnvClient``、``BaseVLAClient``；服务端分别继承 ``BaseEnvFacade``、``BaseVLAFacade``，并通过 ``_register_rpc`` 注册扩展路由。这些基类在 ``RpcFacade`` 之上提供公共路由和锁。只有尚无专用基类的服务类型才直接继承 ``RpcFacade``。业务子类不必实现 ``healthz`` / ``shutdown``。

细节见 :doc:`add_robot` 中的 env_server 与 vla_server 章节。
