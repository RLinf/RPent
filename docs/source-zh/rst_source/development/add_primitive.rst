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

从 LLM 的视角看，两类原语采用相同的接口：一份工具定义、一个执行函数，
以及调用完成后的状态快照。区别仅在于函数内部是调用模型，还是执行脚本化动作。

添加一个脚本化原语
------------------

添加脚本化原语通常需要以下两个步骤：

1. **编写工具函数。** 在 ``robots/<robot>/tools.py`` 中添加函数，用 ``@tool``
   将它声明为工具。函数通过 ``ctx.robot`` 访问当前机器人的环境和模型客户端，
   执行动作后返回 ``ToolResult``。例如，下面的 LIBERO 工具会保持当前位姿，
   并在指定步数后停止：

   .. code-block:: python

      from typing import Annotated

      from pydantic import Field

      from rpent.tools import ToolContext, ToolResult, tool

      @tool
      def hold_pose(
          steps: Annotated[int, Field(ge=1, le=100)] = 10,
          *,
          ctx: ToolContext,
      ) -> ToolResult:
          """Hold the current pose with the gripper closed.

          Args:
              steps: Number of environment steps.
          """
          runtime = ctx.robot
          for _ in range(steps):
              ctx.check_cancelled()
              obs, _, terminated, truncated, _ = runtime.env.step(
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
              )
              runtime.executed_steps += 1
              runtime.set_obs(obs)
              ctx.record_frame(obs["main_images"])
              if terminated or truncated:
                  break
          return ToolResult(data={"steps_requested": steps})

   ``@tool`` 根据函数签名生成参数 schema，并从 Google 风格 docstring 中读取
   工具说明。示例中的 ``Field`` 限定了模型可传入的步数；``ctx`` 则由 toolkit
   提供，不需要模型填写。在机器人代码中，可以用 ``ToolContext[LiberoRuntime]``
   进一步标明上下文中的机器人类型。

   工具执行后，toolkit 会自动保存新的状态快照。对于 ``view_env_state``、
   ``back_project`` 等读取已有观测的工具，可以在 ``@tool`` 下方添加
   ``@readonly``，省去这次状态捕获；调用仍按顺序执行。公共工具和 ``finish``
   不触发状态捕获；``write_text_file`` 和 ``finish`` 不设置 readonly，独占执行。

2. **将工具加入 toolkit。** 把函数声明加入该机器人的工具集合，例如 LIBERO 的
   ``LIBERO_TOOLS``。Toolkit 在构造时接收这组工具，并统一处理参数校验和调用。

完成以上步骤后，``api``、``claude_code`` 和 ``codex`` 三种 planner
都可以调用该工具，无需分别编写适配代码。执行和取消的约定参见
:doc:`interfaces` 中的工具集说明。

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

3. **编写工具函数。** 在函数中通过 ``ctx.robot`` 调用 model client，将返回的
   动作块交给环境执行，并用 ``ToolResult`` 返回执行结果。以 Pi0.5 为例，
   指令从 ``env_obs["task_descriptions"]`` 中读取，模型返回
   ``[chunk, action_dim]`` 的 NumPy 动作块（已去掉 batch 维）。具体实现可参考
   ``robots/libero/tools.py`` 中的 ``pi0_pick``，以及
   ``robots/robocasa/tools.py`` 中的 ``rldx_skill``。

4. **将工具加入 toolkit。** 和脚本化原语一样，使用 ``@tool`` 声明工具，
   再将它加入机器人的工具集合。执行后的状态捕获仍由 toolkit 负责。

5. **在 ``robot_spec.py`` 中连接各组件。** 机器人包中的 ``_init_runtime``
   负责创建环境和模型客户端，并通过 ``runtime_kwargs`` 返回，例如
   ``{"env": MyRobotEnvClient(...), "model": MyModelClient(...)}``。
   ``get_toolkit`` 将这些客户端连同输出目录和 ``MemoryManager`` 传给 toolkit，
   由后者构造本次会话的运行时对象。完整的工厂示例见 :doc:`add_robot`。

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

- **工具侧**：在任务开始或环境重置时调用 ``reset_session``，清空上一回合残留
  的策略状态，避免影响后续任务。

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
- **动作执行后要有新的状态快照。** Toolkit 会在工具函数执行完毕后捕获观测，
  再将结果交给 planner，让下一轮推理能看到动作后的环境。只读工具可以复用已有观测。
- **返回简短的执行结果。** 将动作摘要放在 ``ToolResult.data`` 中，供 planner
  以文本形式读取。图像、深度等大型观测通过 ``EnvState.save`` 保存，文件名会
  自动记入 ``StepRecord.artifacts``。需要向模型展示图片时，使用
  ``ToolResult.images``；历史观测仍可通过 ``view_env_state`` 读取。
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

无论使用哪种模型，都可以沿用上述接入方式：由客户端连接模型服务，再把调用模型
和执行动作的过程写成工具函数，交给 toolkit 调用。
