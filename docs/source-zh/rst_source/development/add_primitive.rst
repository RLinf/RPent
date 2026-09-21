添加动作原语
============

在 RPent 中，*动作原语* 负责将一次工具调用转换为环境可执行的动作。
它既可以基于 VLA、WAM 或 Diffusion Policy，也可以是 ``move_to``、
``open_gripper`` 等脚本化例程。本页分别介绍这两类原语的添加方法。

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
     - 在独立进程（``vla_server``）中运行，通过 toolkit 持有的
       *model client* 调用。
     - Pi0.5（LIBERO）、RLDX-1（RoboCasa）
   * - **脚本化**
       （运动学 / 启发式）
     - 在 agent 进程内运行；需要进行运动学计算时，可能通过一次
       server 侧 RPC 完成。不需要加载模型权重。
     - ``move_to``、``rotate_wrist``、``release``、
       ``back_project``

从 LLM 的视角看，两类原语采用相同的接口：一份工具定义、一个
primitives 方法，以及调用完成后的状态快照。区别仅在于方法的具体实现。

从 Python 签名声明工具
------------------------------

``@tool`` 从函数签名和 Google 风格 docstring 生成工具说明、JSON Schema
和参数校验模型。普通函数和实例方法都可以使用，已有 primitives 对象继续持有
客户端与运行状态：

.. code-block:: python

   from typing import Annotated

   from pydantic import Field

   from rpent.tools import ToolResult, iter_tools, tool

   class MyPrimitives:
       def __init__(self, env):
           self.env = env

       @tool
       def move_delta(
           self,
           delta_xyz: Annotated[list[float], Field(min_length=3, max_length=3)],
       ) -> ToolResult:
           """Move the TCP by a base-frame offset.

           Args:
               delta_xyz: XYZ displacement in metres.
           """
           return ToolResult(data=self.env.move_delta(delta_xyz))

   # 在机器人 toolkit 的 super().__init__(...) 之后注册：
   self._primitives = MyPrimitives(env)
   self.add_tools(iter_tools(self._primitives))

注册时传入实例上的 **绑定方法**。``self`` 不出现在 schema 中，每个实例使用
各自的资源。方法仍可从 Python 直接调用，包括类内的 ``self.move_delta(...)``
和继承的方法。已有方法也可以在注册时包装：
``self.add_tool(tool(self._primitives.move_delta))``。

公开参数必须有类型注解，并能以关键字传入。约束通过
``Annotated[..., Field(...)]`` 声明。函数签名中的默认值在运行时补齐参数；
需要在 schema 中公开时，用 ``Field(json_schema_extra={"default": value})``
显式声明。``Toolkit.execute_tool`` 在调用工具和采集观测之前执行 Pydantic
严格参数校验，拒绝未知参数和非有限数值。直接从 Python 调用方法时，沿用普通
Python 的参数处理方式。

工具返回 ``ToolResult(data=..., images=..., error=...)``。结构化数据放在
``data`` 中，PNG 字节放在 ``images`` 中，错误通过 ``error`` 表达。
原语内部直接调用其他工具时，取得的同样是 ``ToolResult``。
只读工具使用 ``@tool(readonly=True)``，跳过自动观测采集。
通过 ``Toolkit.execute_tool`` 调用时，``@tool`` 和 ``@tool()`` 声明的工具
默认都会采集观测；直接从 Python 调用时不会自动采集。
``readonly`` 仅控制这一步采集，不禁止文件写入，也不允许工具并发执行。
``iter_tools`` 从指定实例或模块中收集装饰过的工具，包括继承的方法。
新增原语时装饰其成员方法即可，无需再维护 schema 或工具名称列表。
普通方法和 property 不会被收集。模式筛选、资源绑定和执行保护仍由 Toolkit
负责。对于 ``state`` 等内部注入参数，声明 ``exclude=("state",)``，再用
``declaration.with_handler(partial(declaration, state=self.state))`` 绑定后注册。
装饰器不会创建环境或模型客户端。

.. _add-primitive-model-based:

添加一个 VLA（或其他基于模型的原语）
------------------------------------

由于模型运行在独立进程中，添加基于模型的原语还需要以下组件：

1. **编写 ``vla_server.py``。** 该进程只持有模型权重和 CUDA 上下文。
   继承 :class:`rpent.robots.components.vla_facade_base.BaseVLAFacade`，实现
   ``predict``，并通过扩展 ``_register_rpc`` 注册其他模型 RPC：

   - 默认传输方式为 **HTTP**，通过 ``POST /call`` 传输 JSON，适合
     LIBERO/Pi0.5 使用的扁平 ``image + state`` 数据。
   - 当观测数据包含多帧历史信息或采用嵌套数据结构时，可以切换到
     **socket RPC**\ （``--transport socket``），避免重复进行 JSON 编码。

   ``BaseVLAFacade`` 会注册 ``vla.predict`` 并串行化模型调用；继承的
   ``RpcFacade.serve`` 负责绑定传输层、处理 ``healthz`` 和 ``shutdown``、
   检测父进程退出并清理资源。

2. **编写 model client。** 继承
   :class:`rpent.robots.components.vla_client_base.BaseVLAClient`；它已经提供
   公共的 ``vla.predict`` 调用，子类只需增加环境专用的输入 / 输出适配。
   LIBERO 的实现可参考
   ``rpent.robots.components.pi05_vla_client.Pi05VLAClient``。

3. **在 primitives 中添加方法。** 在当前机器人的 primitives
   类中调用 model client，将其返回的动作块交给环境执行，并用
   ``ToolResult(data=...)`` 返回动作日志。
   model client 的接口是
   :meth:`rpent.robots.components.pi05_vla_client.Pi05VLAClient.predict`，
   指令从 ``env_obs["task_descriptions"]`` 中读取；返回 ``[chunk, action_dim]``
   的 numpy 动作块（已剥掉 batch 维）：

   .. code-block:: python

      def mymodel_pick(self, target: str) -> ToolResult:
          env_obs = self._env.get_obs()
          env_obs["task_descriptions"] = f"pick {target}"
          chunk = self._model.predict(env_obs)
          self._env.chunk_step(chunk)
          return ToolResult(data={"model": "mymodel", "target": target})

4. **为方法添加 ``@tool``，并注册实例上的绑定方法。** 按上面的示例，
   使用类型注解和 docstring 的 ``Args`` 段声明参数。

5. **在 ``robot_spec.py`` 中连接各组件。** 机器人的 ``get_toolkit`` 使用
   ``runtime_kwargs`` 构造 toolkit：

   .. code-block:: python

      def get_toolkit(*, runtime_kwargs, dashboard_events):
          from robots.myrobot.toolkit import MyRobotToolkit
          return MyRobotToolkit(
              runtime_kwargs=runtime_kwargs,
              dashboard_events=dashboard_events,
          )

   机器人包中的 ``_init_runtime`` 则负责构造 ``runtime_kwargs``，例如
   ``{"env": MyRobotEnvClient(...), "model": MyModelClient(...)}``，再由
   toolkit 构造器将其转发给 primitives。

在多次运行之间复用 vla_server
-----------------------------

模型服务进程通常需要较长的启动时间，因此 runner 可以通过
``--vla-endpoint`` 连接已经在运行的实例：

.. code-block:: bash

   rpent --robot libero --vla-endpoint http://vla-host:8000 ...

如果模型会保存每个回合的内部状态，应提供 ``vla_reset`` RPC，并在任务之间
调用它完成重置。这样，同一个服务进程就能安全地复用于多次连续运行。

带会话状态的 VLA 后端（按客户端隔离策略状态）
------------------------------------------------

大多数 VLA 后端是无状态的：``predict`` 只做推理，不保存各客户端的
中间状态，``session_id`` 可以忽略。但有些模型带按客户端隔离的策略状态
（如 RLDX-1 的 memory/RTC），同一个 ``vla_server`` 服务多个客户端时，
不同客户端的策略状态会互相污染，必须按 session 隔离。接入分三块：

- **facade 侧**：构造 ``BaseVLAFacade`` 子类时传 ``enable_sessions=True``
  和 ``session_timeout_s``，并实现 ``_on_session_drop``——session 结束
  （客户端调用 ``session.close`` RPC 或空闲超时）时在这里清理该客户端
  的策略状态。需要显式重置时，额外提供 ``reset_session`` RPC（只清策略
  状态，不销毁 session）。``serve`` 必须传 ``session_sweep_s`` （> 0），
  让后台线程定期回收过期 session。

- **client 侧**：model client 内部的 ``RpcClient`` 以
  ``enable_sessions=True`` 构造，连接时自动向 server 注册 session。
  ``session_id`` 由 facade 从连接派生并注入 server 端 handler，客户端
  **不传**，也不应在 ``predict`` 的 ``options`` 里伪造 ``session_ids``。

- **primitives 侧**：任务开始前调用 ``reset_session`` 清空上一回合残留
  的策略状态，保证连续多次运行之间状态不串。

单线程 serve（EGL 渲染后端）
----------------------------

大多数后端直接使用基类继承的 ``serve``：transport server 为每个请求
开一个工作线程并发处理。但如果你的服务器进程用 EGL 渲染（如
robosuite / MuJoCo 的 offscreen 渲染，见 ``render_camera``），EGL context
必须留在同一线程，并发 dispatch 会破坏 context 亲和。

这时把 :class:`~rpent.utils.rpc.main_thread_serve.MainThreadServeMixin`
混入你的 facade 类（**先于** ``BaseEnvFacade`` / ``BaseVLAFacade``），
直接继承它覆盖的 ``serve`` 即可——它在守护线程跑 transport server，但
在调用 ``serve`` 的线程（通常是主线程）串行执行每个 dispatch，通过 work
queue 把请求从 transport 线程交给该线程：

.. code-block:: python

   from rpent.utils.rpc.main_thread_serve import MainThreadServeMixin
   from rpent.robots.components.env_facade_base import BaseEnvFacade

   class MyEnvFacade(MainThreadServeMixin, BaseEnvFacade):
       ...

   facade.serve(transport="http", host=host, port=port)  # dispatch 在主线程串行

mixin 覆盖的 ``serve`` 与 :class:`~rpent.utils.rpc.RpcFacade` 的
``serve`` 契约一致：同样支持 ``healthz`` / ``shutdown``、parent-watch
和 session 支持（构造传 ``enable_sessions=True`` 时，``serve`` 仍须传
``session_sweep_s``）。子类**不需要**重写 ``serve`` 来委托——直接继承
即可（参考 ``robots/robocasa/env_server.py`` 的
``RoboCasaEnvFacade``）。不需要 EGL 单线程的后端直接继承基类用默认
``serve``。

新原语的设计原则
----------------

- **工具名称应描述意图，而非底层动作序列。** 例如使用 ``pi0_pick``，
  而不是 ``execute_action_chunk_of_length_20``。
- **每个工具执行结束后都要保存新的状态快照。** 下一轮需要读取动作执行后的
  环境状态，因此原语不能在渲染完成前返回。
- **保持 ``ToolResult.data`` 简短。** 返回值会以文本形式提供给 LLM；图像、深度数据和
  其他大型观测应通过 ``EnvState.save`` 保存；``EnvState`` 会把每个逻辑基础
  文件名自动加入其持有的 ``StepRecord.artifacts`` 集合。图像通过
  ``view_env_state`` 提供，几何数据通过环境工具访问，不返回原始路径。
- **安全限制由 ``env_server`` 强制执行。** LLM 可能使用任意参数调用工具，
  因此工作空间边界和安全限制不能只依赖 toolkit。

其他基于模型的原语
------------------

同样的架构也适用于非 VLA 的模型原语：

- **World Action Model (WAM)** —— 根据模型预测生成 rollout 和执行计划，
  再交给环境执行。其接入方式与 VLA 相同：使用独立进程和独立 client。
- **Diffusion Policy / MPC** —— 接口形式相同，但工具返回的动作可能是一段
  trajectory，而非单个 chunk，并由 ``env_server`` 按顺序执行。
- **多个原语共享一个 server** —— 一个 ``vla_server`` 可以承载
  多个模型，由工具通过 ``predict`` 的 ``model`` kwarg 选择要调用的模型
  或输出 head。

无论具体实现如何，框架的契约都保持不变：模型进程 → model client →
primitives 方法 → 工具定义 → ``Toolkit.add_tool``。
