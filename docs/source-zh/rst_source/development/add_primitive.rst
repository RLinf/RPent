添加 Action Primitive
=====================

RPent 中的 *action primitive* 就是把一次 tool 调用变成 environment 可
执行动作的东西。它可以是一个学出来的策略 (VLA、WAM、扩散 planner),
也可以是一段脚本化例程 (``move_to``、``open_gripper``)。本页说明这
两类各自怎么加。

两种 primitive 形状
-------------------

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - 类别
     - 跑在哪里
     - 例子
   * - **基于模型的**
       (VLA / WAM / 扩散 / …)
     - 自己的进程 (``vla_server``)。通过 toolkit 持有的 *model
       client* 调用。
     - Pi0.5 (LIBERO)、RLDX-1 (RoboCasa)
   * - **脚本化的**
       (运动学 / 启发式)
     - Agent 进程内; 需要运动学时可能走一次 driver 侧 RPC。
       没有模型权重。
     - ``move_to``、``rotate_wrist``、``release``、
       ``back_project``

两种形状对 LLM 呈现的方式相同: 一份 tool schema、一个
primitive-driver 方法、调用后一次状态 dump。差异只在 *方法内部做什么*。

添加一个脚本化 primitive
------------------------

脚本化 primitive 最快能加进来。套路:

1. **在 primitive driver 上加一个方法。** 在你 env 的 primitive
   driver 类 (如 ``LiberoPrimitives``、``MyRobotPrimitives``) 上加
   一个方法。接受 tool 的 kwargs, 做实事 —— 通常是一次或多次
   ``self._env.step(...)`` —— 返回一个小的 ``dict`` 日志。

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

到这里, ``api``、``claude_code``、``codex`` 三种 cerebrum 都能调它 ——
不需要改别的。

添加一个 VLA (或其他基于模型的 primitive)
-----------------------------------------

基于模型的 primitive 要多一些脚手架, 因为模型跑在自己的进程。套路:

1. **写一个 ``vla_server.py``。** 只持有模型权重和 CUDA 上下文。
   暴露 ``vla_load`` / ``vla_infer`` / ``vla_reset``:

   - 观测是扁平的 ``image + state`` 时走 **HTTP** (LIBERO / Pi0.5
     模式)。
   - 观测是带历史堆叠的嵌套 numpy dict 时走 **socket RPC** (RoboCasa /
     RLDX-1 模式)。

   server 开始监听时, 在 stdout 上输出一条 ``transport_ready`` JSON
   事件; ``rpent/cli/main.py`` 会阻塞等这个事件。

2. **写一个 model client。** 一个小类, 提供 ``predict(obs)`` (HTTP) 或
   ``vla_infer(obs)`` (socket), toolkit 会和 env client 一起持有它。
   可以参考 ``rpent.utils.vla_client.VLAClient`` 这个 LIBERO 用例。

3. **在 primitive-driver 上加一个方法。** 在 env 的 primitive-driver
   类里调 model client, 把返回的 chunk 转给 env, 并返回一个日志 dict:

   .. code-block:: python

      def mymodel_pick(self, target: str) -> dict:
          obs = self._env.get_obs()
          chunk = self._model.predict(obs, instruction=f"pick {target}")
          self._env.chunk_step(chunk)
          return {"model": "mymodel", "target": target}

4. **加 tool schema** 并在 toolkit 里注册 (跟脚本化那一节的做法一样)。

5. **在 ``__init__.py`` 里串起来。** env 的 ``get_toolkit`` 用正确的
   ``primitives_kwargs`` 构造 toolkit:

   .. code-block:: python

      def get_toolkit(*, primitives_kwargs, video_path=None):
          from robots.myrobot.toolkit import MyRobotToolkit
          return MyRobotToolkit(
              primitives_kwargs=primitives_kwargs,
              video_path=video_path,
          )

   ``rpent/cli/main.py`` 会传入 ``{"env": MyRobotEnvClient(...),
   "model": MyModelClient(...)}``。

跨 run 复用同一个 vla_server
----------------------------

模型 server 起动很贵 (加载权重是大头)。Runner 支持指向已经在跑的实例:

.. code-block:: bash

   python rpent/cli/main.py --vla-endpoint http://vla-host:8000 ...

把你的 ``vla_server`` 设计成 **task 无关**——用一个显式的 ``vla_reset``
RPC 清 per-episode 状态——这样一个进程就能安全服务很多次连续 run。

新 primitive 的设计原则
-----------------------

- **Tool 描述意图, 不描述动作。** 好的 tool 名叫 ``pi0_pick``,
  不叫 ``execute_action_chunk_of_length_20``。LLM 是按名字挑 tool 的,
  名字要能自解释。
- **每个 tool 结束时都要 dump 状态。** 下一轮依赖 dump 反映动作后的
  世界; 别在渲染完成前让 primitive 提前 return。
- **返回小 dict。** Tool 返回值以文本形式喂回 LLM。控制在几百字节
  内; 大 payload (图像、深度、``states.json``) 走 state dump, 以
  image content block 的形式回传。
- **护栏 (guardrail) 属于 env_server**, 不属于 toolkit。LLM 会用任意
  参数调任意 tool; 工作空间边界和安全钳位必须在 driver 侧强制执行。

超越 VLA
--------

同样的模式扩展到非 VLA 的模型 primitive:

- **World Action Model (WAM)** —— 基于想象的 rollout, 产出一个计划
  让 env 去执行。接法和 VLA 完全一样: 自己的进程、自己的 client。
- **扩散 planner / MPC** —— 形状一样; 只是 tool 返回的 "动作" 可能
  是一段 trajectory 而非单个 chunk, 由 ``env_server`` 一步步走完。
- **多个 primitive 共享一个 server** —— 一个 ``vla_server`` 可以承载
  多个模型; 由 tool 通过 ``vla_infer`` 的 ``model`` kwarg 选调用哪个
  head。

无论形状如何, 框架的契约不变: 模型进程 → model client →
primitive-driver 方法 → tool schema → ``Toolkit.add_tool``。
