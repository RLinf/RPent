添加 Action Primitive
=====================

在 RPent 中，*action primitive* 将一次工具调用转换成环境可以执行的动作。
它既可以是 VLA、WAM 或扩散模型等学习策略，也可以是 ``move_to``、
``open_gripper`` 等脚本化程序。本页分别介绍这两类 primitive 的
接入方法。

两类 primitive
--------------

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - 类别
     - 运行位置
     - 例子
   * - **基于模型的**
       (VLA / WAM / 扩散模型 / …)
     - 自己的进程 (``vla_server``)。通过 toolkit 持有的 *model
       client* 调用。
     - Pi0.5 (LIBERO)、RLDX-1 (RoboCasa)
   * - **脚本化的**
       (运动学 / 启发式)
     - Agent 进程内; 需要运动学时可能走一次 driver 侧 RPC。
       没有模型权重。
     - ``move_to``、``rotate_wrist``、``release``、
       ``back_project``

两类 primitive 向 LLM 提供相同的接口：一份工具 schema、一个
primitive driver 方法，以及调用完成后的状态快照。区别仅在于方法内部的实现。

添加一个脚本化 primitive
------------------------

添加脚本化 primitive 通常包含以下三个步骤：

1. **在 primitive driver 上加一个方法。** 在你 env 的 primitive
   driver 类 (如 ``LiberoPrimitives``、``MyRobotPrimitives``) 上加
   一个方法。方法接收工具调用的参数，执行一次或多次
   ``self._env.step(...)``，并返回一个简短的日志字典。

   .. code-block:: python

      def open_drawer(self, dx: float = 0.15) -> dict:
          # 保持夹爪闭合, 沿 -x 方向后拉 dx 米。
          for _ in range(N):
              self._env.step(build_open_drawer_chunk(dx))
          return {"ok": True, "dx": dx}

2. **写 tool schema。** 在 ``toolkit.py`` 的 ``TOOLS_SPEC`` 里加一条:

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

3. **在 toolkit 中注册。** 让 tool 走 toolkit 的 ``_step`` 辅助函数,
   这样跑完后会自动重新渲染状态:

   .. code-block:: python

      self.add_tool("open_drawer", OPEN_DRAWER_SPEC,
                    lambda **kw: self._step("open_drawer", **kw))

完成以上步骤后，``api``、``claude_code`` 和 ``codex`` 三种 planner 都可以
调用该工具，无需修改其他代码。

添加一个 VLA（或其他基于模型的 primitive）
-----------------------------------------

基于模型的 primitive 需要增加一些组件，因为模型运行在独立进程中：

1. **写一个 ``vla_server.py``。** 只持有模型权重和 CUDA 上下文。
   继承 :class:`rpent.utils.rpc.RpcFacade`, 通过 ``_dispatch`` 暴露
   模型方法 (如 ``predict``):

   - 默认使用 **HTTP**，通过 ``POST /call`` 传输 JSON，适合 LIBERO/Pi0.5
     使用的扁平 ``image + state`` 数据。
   - 当观测数据包含多帧历史信息或采用嵌套数据结构时，可以切换到
     **socket RPC**（``--transport socket``），避免重复进行 JSON 编码。

   ``RpcFacade.serve`` 负责 transport 绑定、``healthz``、``shutdown``、
   检测父进程退出并执行资源清理；这里只需实现模型相关的方法。

2. **编写一个 model client。** 用于封装
   :class:`rpent.utils.rpc.RpcClient`
   （:class:`HttpRpcClient` 或 :class:`SocketRpcClient`），并提供模型的
   调用接口。LIBERO 的实现可以参考 ``rpent.utils.vla_client.VLAClient``。

3. **在 primitive driver 上添加一个方法。** 在环境的 primitive driver
   类中调用 model client，将返回的动作块交给环境执行，并返回日志字典：

   .. code-block:: python

      def mymodel_pick(self, target: str) -> dict:
          obs = self._env.get_obs()
          chunk = self._model.predict(obs, instruction=f"pick {target}")
          self._env.chunk_step(chunk)
          return {"model": "mymodel", "target": target}

4. **加 tool schema** 并在 toolkit 里注册 (跟脚本化那一节的做法一样)。

5. **在 ``__init__.py`` 中完成连接。** 环境的 ``get_toolkit`` 使用正确的
   ``primitives_kwargs`` 构造 toolkit：

   .. code-block:: python

      def get_toolkit(*, primitives_kwargs, video_path=None):
          from robots.myrobot.toolkit import MyRobotToolkit
          return MyRobotToolkit(
              primitives_kwargs=primitives_kwargs,
              video_path=video_path,
          )

   ``rpent/cli/main.py`` 会传入 ``{"env": MyRobotEnvClient(...),
   "model": MyModelClient(...)}``。

在多次运行间复用 vla_server
---------------------------

模型 server 的启动时间主要用于加载权重。Runner 可以通过
``--vla-endpoint`` 连接已经运行的实例：

.. code-block:: bash

   rpent --env libero --vla-endpoint http://vla-host:8000 ...

应将 ``vla_server`` 设计为在不同 task 之间保持无状态，并通过显式的
``vla_reset`` RPC 清除每个 episode 的状态。这样，同一个进程便可以连续
服务多次运行。

新 primitive 的设计原则
-----------------------

- **工具名称应描述意图，而非底层动作序列。** 例如使用 ``pi0_pick``，
  而不是 ``execute_action_chunk_of_length_20``。LLM 会根据名称选择工具，
  因此名称需要清楚表达用途。
- **每个工具执行结束后都要保存新的状态快照。** 下一轮需要读取动作执行后的
  环境状态，因此 primitive 不能在渲染完成前返回。
- **工具只返回简短的字典。** 返回值会以文本形式提供给 LLM；图像、深度和
  ``states.json`` 等较大数据通过状态快照提供。
- **安全限制由 ``env_server`` 强制执行。** LLM 可能使用任意参数调用工具，
  因此工作空间边界和安全限制不能只依赖 toolkit。

其他基于模型的 primitive
------------------------

同样的架构也适用于非 VLA 的模型 primitive：

- **World Action Model (WAM)** —— 根据模型预测生成 rollout 和执行计划，
  再交给环境执行。接入方式与 VLA 相同：使用独立进程和独立 client。
- **扩散模型 / MPC** —— 接口形式相同，但工具返回的动作可能是一段
  trajectory，而非单个 chunk，并由 ``env_server`` 按顺序执行。
- **多个 primitive 共享一个 server** —— 一个 ``vla_server`` 可以承载
  多个模型，由工具通过 ``vla_infer`` 的 ``model`` kwarg 选择要调用的模型
  或输出 head。

无论具体实现如何，框架的契约不变：模型进程 → model client →
primitive-driver 方法 → tool schema → ``Toolkit.add_tool``。
