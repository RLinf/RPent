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

约定：从 ``toolkit.list_tools()`` 获取原生声明，将 ``name``、
``description`` 和 ``input_schema`` 转换成模型 SDK 所需的格式。发送前通过
``rpent.utils.templates.substitute`` 替换 schema 中的占位符。对已注册的 toolkit
工具，通过 ``toolkit.execute_tool(name, input_dict)`` 执行；当 ``toolkit.finish_result``
设置完成或达到轮次上限时，返回 ``PlannerResult``。

工具集
------

在 ``robots/<robot>/toolkit.py`` 中继承 ``Toolkit``。基类构造器已注册公共文件
工具和 ``finish``。在原语成员方法上添加 ``@tool``，再注册实例上的绑定方法：

.. code-block:: python

   self.add_tool(self._primitives.move_to)
   # 也可以收集并注册整个原语对象的工具声明：
   self.add_tools(iter_tools(self._primitives))

原生 ``Tool`` 包含 ``name``、``description``、``args_schema``、``handler``
和 ``readonly``，通过 ``input_schema`` 获取生成的 JSON Schema。
Google 风格 docstring 的 ``Args`` 段提供参数说明，类型注解和 ``Field`` 约束
定义参数校验。Python 默认值用于补齐省略的参数；如需在 schema 中公开默认值，
使用 ``Field(json_schema_extra={"default": value})`` 显式声明。``self`` 不暴露
给模型，环境和模型客户端仍由实例持有。

``add_tool(declaration, replace=True)`` 显式覆盖同名工具，否则重复注册会报错。
``declaration.with_handler(handler)`` 可以绑定内部资源或包装执行逻辑，同时
保留 schema 和只读标记。资源注入示例见 :doc:`add_primitive`。

原生结果与执行
~~~~~~~~~~~~~~

处理函数和 ``get_env_state`` 均返回 ``ToolResult``：``data`` 是结果字典，
``images`` 是按顺序排列的 PNG 字节列表，``error`` 是错误文本或 ``None``。
结构化输出使用 ``to_dict()``，发给模型的限长文本使用 ``to_text()``，
失败状态使用 ``is_error``。各 planner 适配器负责生成对应 SDK 的内容块。

planner 和 Dashboard 均通过 ``execute_tool`` 调用已注册工具，在处理函数执行前
统一校验参数。状态推进工具执行后通过 ``get_env_state(command, result,
elapsed_s)`` 采集新观测，返回观测数据、工具图片和执行错误。
``@tool(readonly=True)`` 跳过这一步自动采集。Primitives 类管理机器人运行状态
和视频帧缓冲，``EnvState`` 管理已记录的步骤和工件。
公共文件工具调用 ``MemoryManager.authorize_read`` / ``authorize_write``
判断路径访问权限。

``finish`` 被接受后，``toolkit.finish_result`` 保存完整结果，并去除内部
``_finish`` 标记。API、Claude Code 和 Codex 读取该结果。错误或
``_finish=False`` 不会设置结束状态，机器人特有的操作员反馈字段会完整保留。

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
