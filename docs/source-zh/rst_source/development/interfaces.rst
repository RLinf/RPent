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

``get_toolkit`` 一般只需把 ``runtime_kwargs`` 传给机器人子类；
``dashboard_events`` 和 ``config`` 由当前 runner 传入。它需要构造一个
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

约定：从 ``toolkit.list_tools()`` 获取工具，通过
``toolkit.execute_tool(name, input_dict)`` 执行调用，并把结果交回模型；
当 ``toolkit.finish_result`` 已设置或达到轮次上限时，返回 ``PlannerResult``。
SDK 格式转换和 schema 占位符的处理见 :doc:`../usage/configure_planner`。

工具集
------

在 ``robots/<robot>/toolkit.py`` 中继承 ``Toolkit``。基类已注册公共文件工具
和 ``finish``；为原语方法添加 ``@tool`` 后，注册实例上的绑定方法：

.. code-block:: python

   self.add_tool(self._primitives.move_to)
   # 也可以注册所有带 @tool 的方法：
   self.add_tools(iter_tools(self._primitives))

``add_tool(declaration, replace=True)`` 覆盖同名工具，否则重复注册会报错。
``declaration.with_handler(handler)`` 可包装处理函数，保留 schema 和
``readonly`` 设置。

参数由类型注解、``Field`` 约束和 Google 风格 docstring 定义。函数默认值
在运行时生效；如需写入 schema，使用 ``Field(json_schema_extra={"default": value})``。

处理函数返回 ``ToolResult``：``data`` 是结果字典，``images`` 是 PNG 字节列表，
``error`` 是错误信息或 ``None``。``execute_tool`` 校验参数、执行处理函数，
再通过 ``get_env_state`` 采集观测。``@tool(readonly=True)`` 跳过观测采集；
直接调用 Python 方法不经过这些校验和采集步骤。工具声明和观测示例见
:doc:`add_primitive`。

``finish`` 被接受后，``toolkit.finish_result`` 保存去除内部 ``_finish`` 标记的
结果。调用出错或返回 ``_finish=False`` 时，不会结束任务。

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
