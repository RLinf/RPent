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
       driver 侧 RPC 完成。不需要加载模型权重。
     - ``move_to``、``rotate_wrist``、``release``、
       ``back_project``

从 LLM 的视角看，两类原语采用相同的接口：一份工具定义、一个
primitive driver 方法，以及调用完成后的状态快照。区别仅在于方法的具体实现。

添加一个脚本化原语
------------------

添加脚本化原语通常需要以下三个步骤：

1. **在 primitive driver 中添加方法。** 在当前环境的 primitive
   driver 类（如 ``LiberoPrimitives``、``MyRobotPrimitives``）中添加
   一个方法。该方法接收工具调用的参数，执行一次或多次
   ``self._env.step(...)``，并返回一个简短的日志字典。

   .. code-block:: python

      def open_drawer(self, dx: float = 0.15) -> dict:
          # 保持夹爪闭合，沿 -x 方向后拉 dx 米。
          for _ in range(N):
              self._env.step(build_open_drawer_chunk(dx))
          return {"ok": True, "dx": dx}

2. **添加工具定义。** 在 ``toolkit.py`` 的 ``TOOLS_SPEC`` 中新增一项：

   .. code-block:: python

      {
          "name": "open_drawer",
          "description": "Pull the currently-grasped drawer handle "
                         "backwards by ``dx`` meters.",
          "input_schema": {
              "type": "object",
              "properties": {"dx": {"type": "number"}},
              "required": [],
          },
      }

3. **在 toolkit 中注册工具。** 通过 toolkit 的 ``_step`` 辅助函数运行
   该工具，使其在执行结束后自动重新渲染状态：

   .. code-block:: python

      self.add_tool("open_drawer", OPEN_DRAWER_SPEC,
                    lambda **kw: self._step("open_drawer", **kw))

完成以上步骤后，``api``、``claude_code`` 和 ``codex`` 三种 planner
都可以调用该工具，无需修改其他代码。

.. _add-primitive-model-based:

添加一个 VLA（或其他基于模型的原语）
------------------------------------

由于模型运行在独立进程中，添加基于模型的原语还需要以下组件：

1. **编写 ``vla_server.py``。** 该进程只持有模型权重和 CUDA 上下文。
   继承 :class:`rpent.utils.rpc.RpcFacade`，并通过 ``_dispatch`` 暴露
   模型方法（如 ``predict``）：

   - 默认传输方式为 **HTTP**，通过 ``POST /call`` 传输 JSON，适合
     LIBERO/Pi0.5 使用的扁平 ``image + state`` 数据。
   - 当观测数据包含多帧历史信息或采用嵌套数据结构时，可以切换到
     **socket RPC**\ （``--transport socket``），避免重复进行 JSON 编码。

   ``RpcFacade.serve`` 负责绑定传输层、处理 ``healthz`` 和 ``shutdown``、
   检测父进程退出并清理资源；这里只需实现与模型相关的方法。

2. **编写 model client。** 创建一个轻量的类，封装
   :class:`rpent.utils.rpc.RpcClient`
   （:class:`HttpRpcClient` 或 :class:`SocketRpcClient`），并提供模型调用
   接口。LIBERO 的实现可参考 ``rpent.utils.vla_client.VLAClient``。

3. **在 primitive driver 中添加方法。** 在当前环境的 primitive driver
   类中调用 model client，将其返回的动作块交给环境执行，并返回日志字典：

   .. code-block:: python

      def mymodel_pick(self, target: str) -> dict:
          obs = self._env.get_obs()
          chunk = self._model.predict(obs, instruction=f"pick {target}")
          self._env.chunk_step(chunk)
          return {"model": "mymodel", "target": target}

4. **添加工具定义并在 toolkit 中注册。** 具体做法与脚本化原语相同。

5. **在 ``__init__.py`` 中连接各组件。** 环境的 ``get_toolkit`` 使用
   ``primitives_kwargs`` 构造 toolkit：

   .. code-block:: python

      def get_toolkit(*, primitives_kwargs, video_path=None):
          from robots.myrobot.toolkit import MyRobotToolkit
          return MyRobotToolkit(
              primitives_kwargs=primitives_kwargs,
              video_path=video_path,
          )

   环境包中的 ``_init_runtime`` 则负责构造 ``primitives_kwargs``，例如
   ``{"env": MyRobotEnvClient(...), "model": MyModelClient(...)}``，再由
   toolkit 构造器将其转发给 primitive driver。

在多次运行之间复用 vla_server
-----------------------------

模型服务进程通常需要较长的启动时间，因此 runner 可以通过
``--vla-endpoint`` 连接已经在运行的实例：

.. code-block:: bash

   rpent --env libero --vla-endpoint http://vla-host:8000 ...

如果模型会保存每个回合的内部状态，应提供 ``vla_reset`` RPC，并在任务之间
调用它完成重置。这样，同一个服务进程就能安全地复用于多次连续运行。

新原语的设计原则
----------------

- **工具名称应描述意图，而非底层动作序列。** 例如使用 ``pi0_pick``，
  而不是 ``execute_action_chunk_of_length_20``。
- **每个工具执行结束后都要保存新的状态快照。** 下一轮需要读取动作执行后的
  环境状态，因此原语不能在渲染完成前返回。
- **工具只返回简短的字典。** 返回值会以文本形式提供给 LLM；图像、深度数据和
  ``states.json`` 等较大的内容则通过状态快照提供。
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
  多个模型，由工具通过 ``vla_infer`` 的 ``model`` kwarg 选择要调用的模型
  或输出 head。

无论具体实现如何，框架的契约都保持不变：模型进程 → model client →
primitive driver 方法 → 工具定义 → ``Toolkit.add_tool``。
