添加动作原语
==================

在 RPent 中，*动作原语* 负责将一次工具调用转换为环境可执行的动作。它既可以基于 VLA、WAM 或 Diffusion Policy，也可以是 ``move_to``、 ``open_gripper`` 等脚本化例程。本页介绍工具声明、注册与执行。

两类原语
--------

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - 类别
     - 运行位置
     - 例子
   * - **基于模型的**
       （VLA / WAM / Diffusion Policy / …）
     - 在独立模型服务进程中运行，通过 toolkit 持有的
       *model client* 调用。
     - Pi0.5（LIBERO）、RLDX-1（RoboCasa）
   * - **脚本化**
       （运动学 / 启发式）
     - 在 agent 进程内运行；需要进行运动学计算时，可能通过一次
       server 侧 RPC 完成。不需要加载模型权重。
     - ``move_to``、``rotate_wrist``、``release``、
       ``back_project``

从 LLM 的视角看，两类原语采用相同的接口：一份工具定义、一个 primitives 方法，以及调用完成后的状态快照。区别仅在于方法的具体实现。

添加一个脚本化原语
------------------

在机器人的 primitives 类中声明方法，并注册绑定到实例的工具：

.. code-block:: python

   from rpent.tools import ToolResult, iter_tools, tool

   class MyRobotPrimitives:
       @tool
       def open_drawer(self, dx: float = 0.15) -> ToolResult:
           """Pull the grasped drawer handle backwards.

           Args:
               dx: Pull distance in meters.
           """
           self._env.step(build_open_drawer_chunk(dx))
           return ToolResult(data={"ok": True, "dx": dx})

   # 在 toolkit 中构造 self._primitives 后注册：
   self.add_tools(iter_tools(self._primitives))

参数类型注解和 docstring 生成模型可见的 schema。Toolkit 校验参数，并在执行后通过 ``get_env_state`` 采集状态。``view_env_state``、``back_project`` 等读取工具使用 ``@tool(readonly=True)``。模块级工具可用 ``@tool(exclude=("state",))`` 排除内部资源参数，再通过 ``Tool.with_handler`` 绑定。在线 planner 共用注册后的工具声明。

.. _add-primitive-model-based:

基于模型的工具
----------------

VLA/WAM 的服务、客户端、adapter 与会话生命周期见 :doc:`add_action_model`。平台工具调用模型客户端，通过环境执行返回的原生动作块，再返回 ``ToolResult``。工具声明与注册仍使用上文的 ``@tool`` 和 ``iter_tools``。

判断报错的工具是否执行了动作
----------------------------

夹爪可能已经合上了，相机却没拍到动作后的画面。遇到这种情况，Toolkit 会在 ``execute_tool`` 返回的 ``ToolResult`` 上设置 ``execution_uncertain=True``。调用方可以用它判断是否需要再观察一次。机器人本身无法确认动作结果时，工具也可以主动返回这个标记。模型和轨迹里看不到它。

默认情况下，动作工具开始执行后抛出异常，这个标记为真。只读工具报错时，标记为假。``write_text_file`` 写文件时不会移动机器人，所以它用 ``readonly=True`` 跳过自动观测，也可以和其他只读调用并行。它用 ``@tool(readonly=True, uncertain_on_error=True)`` 声明写到一半失败时，文件可能已经改变。

写入前被拒绝、参数有误，或调用在工具启动前被取消时，动作或写入还没开始，``execution_uncertain`` 为假。如果返回的 ``ToolResult.data["code"]`` 是 ``tool_result_serialization_failed``，说明结果没能完整传回。这个错误码本身无法说明动作有没有发生。

在两次调用之间查看环境
----------------------

如果上一次没拍到夹爪的动作，可以用 ``Toolkit.refresh_observation()`` 查看夹爪现在的位置，再决定是否重试。刷新会等正在运行的工具结束，也不占规划器的工具调用次数。Toolkit 已暂停或关闭时，刷新会抛出 ``ToolCancelled``。

只有 ``get_env_state`` 新增了观测，轨迹里才会多一步。采集出错时，调用方会收到异常。如果新观测已经记下，随后收到取消请求，这一步仍会发布，再抛出 ``ToolCancelled``。

回放时跳过 ``{"action": "refresh_observation"}``。这一步只记录观测，没有机器人动作需要重做。动作的 ``StepRecord.command`` 只存工具名和参数，操作来源另行记录。

单线程 serve（EGL 渲染后端）
----------------------------

大多数后端直接使用基类继承的 ``serve``：transport server 为每个请求开一个工作线程并发处理。但如果你的服务器进程用 EGL 渲染（如 robosuite / MuJoCo 的 offscreen 渲染，见 ``render_camera``），EGL context 必须留在同一线程，并发 dispatch 会破坏 context 亲和。

这时把 :class:`~rpent.utils.rpc.main_thread_serve.MainThreadServeMixin` 混入你的 facade 类（**先于** ``BaseEnvFacade`` / ``BaseVLAFacade``），直接继承它覆盖的 ``serve`` 即可——它在守护线程跑 transport server，但在调用 ``serve`` 的线程（通常是主线程）串行执行每个 dispatch，通过 work queue 把请求从 transport 线程交给该线程：

.. code-block:: python

   from rpent.utils.rpc.main_thread_serve import MainThreadServeMixin
   from rpent.robots.components.env_facade_base import BaseEnvFacade

   class MyEnvFacade(MainThreadServeMixin, BaseEnvFacade):
       ...

   facade.serve(transport="http", host=host, port=port)  # dispatch 在主线程串行

mixin 覆盖的 ``serve`` 与 :class:`~rpent.utils.rpc.RpcFacade` 的 ``serve`` 契约一致：同样支持 ``healthz`` / ``shutdown``、父进程退出检测和会话管理（构造时传入 ``enable_sessions=True``，则调用 ``serve`` 时仍须传入 ``session_sweep_s``）。子类直接继承即可，无需重写 ``serve`` 再转发调用，参考 ``robots/robocasa/env_server.py`` 中的 ``RoboCasaEnvFacade``。没有 EGL 单线程要求的后端直接使用基类默认的 ``serve``。

新原语的设计原则
----------------

- **工具名称应描述意图，而非底层动作序列。** 例如使用 ``pi0_pick``，而不是 ``execute_action_chunk_of_length_20``。
- **改变状态的工具在返回规划器前采集状态。** Toolkit 在处理函数执行后调用 ``get_env_state``，期间继续保持独占执行。``readonly`` 工具跳过这一步，并可与其他只读工具并行。
- **保持结果数据简短。** 返回值会以文本形式提供给 LLM；图像、深度数据和其他大型观测应通过 ``EnvState.save`` 保存；``EnvState`` 会把每个逻辑基础文件名自动加入其持有的 ``StepRecord.artifacts`` 集合。图像通过 ``view_env_state`` 提供，几何数据通过环境工具访问，不返回原始路径。
- **安全限制由 ``env_server`` 强制执行。** LLM 可能使用任意参数调用工具，因此工作空间边界和安全限制不能只依赖 toolkit。
