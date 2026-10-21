添加动作模型
================

接入 VLA 或 WAM 服务，并将预测结果交给平台的原生动作执行器。工具声明与状态采集见 :doc:`add_primitive`，平台发现和运行时接线见 :doc:`add_robot`。

共同职责边界
----------------

模型包负责权重与模型专用预处理；平台负责物理观测、控制方式和执行工具。公共 AM 基类复用 RPC 传输与推理转发。VLA 保留模型专用消息格式，WAM 使用 ``am_contracts.py`` 中的物理观测与原生控制声明。

添加 VLA
----------

由于模型运行在独立进程中，添加基于模型的原语还需要以下组件：

1. **编写 ``vla_server.py``。** 该进程只持有模型权重和 CUDA 上下文。继承 :class:`rpent.robots.components.am_facade_base.BaseVLAFacade`，实现 ``predict``，并通过扩展 ``_register_rpc`` 注册其他模型 RPC：

   - 默认传输方式为 **HTTP**，通过 ``POST /call`` 传输 JSON，适合
     LIBERO/Pi0.5 使用的扁平 ``image + state`` 数据。
   - 当观测数据包含多帧历史信息或采用嵌套数据结构时，可以切换到
     **socket RPC**\ （``--transport socket``），避免重复进行 JSON 编码。

   ``BaseVLAFacade`` 会注册 ``vla.predict`` 并串行化模型调用；继承的 ``RpcFacade.serve`` 负责绑定传输层、处理 ``healthz`` 和 ``shutdown``、检测父进程退出并清理资源。

2. **编写 model client。** 继承
   :class:`rpent.robots.components.am_client_base.BaseVLAClient`；它已经提供
   公共的 ``vla.predict`` 调用，子类只需增加环境专用的输入 / 输出适配。 LIBERO 的实现可参考 ``rpent.robots.components.pi05_vla_client.Pi05VLAClient``。

3. **在 primitives 中添加方法。** 在当前机器人的 primitives 类中调用 model client，将其返回的动作块交给环境执行，并返回 ``ToolResult``。 model client 的接口是
   :meth:`rpent.robots.components.pi05_vla_client.Pi05VLAClient.predict`，
   指令从 ``env_obs["task_descriptions"]`` 中读取；返回形状为 ``[chunk, action_dim]`` 的 NumPy 动作块（已移除 batch 维度）：

   .. code-block:: python

      @tool
      def mymodel_pick(self, target: str) -> ToolResult:
          """使用模型抓取指定目标。"""
          env_obs = self._env.get_obs()
          env_obs["task_descriptions"] = f"pick {target}"
          chunk = self._model.predict(env_obs)
          self._env.chunk_step(chunk)
          return ToolResult(data={"model": "mymodel", "target": target})

4. **在 toolkit 中注册声明的工具。** 具体做法与脚本化原语相同。

5. **在 ``robot_spec.py`` 中连接各组件。** 机器人的 ``get_toolkit`` 使用 ``runtime_kwargs`` 构造 toolkit：

   .. code-block:: python

      def get_toolkit(*, runtime_kwargs, dashboard_events, config):
          from robots.myrobot.toolkit import MyRobotToolkit
          from rpent.memory import MemoryManager

          return MyRobotToolkit(
              runtime_kwargs=runtime_kwargs,
              dashboard_events=dashboard_events,
              memory=MemoryManager(config.prompt_vars["memory_dir"]),
          )

   机器人包中的 ``_init_runtime`` 则负责构造 ``runtime_kwargs``，例如 ``{"env": MyRobotEnvClient(...), "model": MyModelClient(...)}``，再由 toolkit 构造器将其转发给 primitives。

在多次运行之间复用 vla_server
-----------------------------

模型服务进程通常需要较长的启动时间，因此 runner 可以通过 ``--am-endpoint`` 连接已经在运行的实例：

.. code-block:: bash

   rpent --robot libero --am-endpoint http://vla-host:8000 ...

如果模型会保存每个回合的内部状态，应提供 ``vla_reset`` RPC，并在任务之间调用它完成重置。这样，同一个服务进程就能安全地复用于多次连续运行。

带会话状态的 VLA 后端（按客户端隔离策略状态）
------------------------------------------------

大多数 VLA 后端是无状态的：``predict`` 只做推理，不保存各客户端的中间状态，``session_id`` 可以忽略。但有些模型带按客户端隔离的策略状态（如 RLDX-1 的 memory/RTC），同一个 ``vla_server`` 服务多个客户端时，不同客户端的策略状态会互相污染，必须按 session 隔离。接入分三块：

- **facade 侧**：构造 ``BaseVLAFacade`` 子类时传 ``enable_sessions=True`` 和 ``session_timeout_s``，并实现 ``_on_session_drop``——session 结束（客户端调用 ``session.close`` RPC 或空闲超时）时在这里清理该客户端的策略状态。需要显式重置时，额外提供 ``reset_session`` RPC（只清策略状态，不销毁 session）。``serve`` 必须传 ``session_sweep_s`` （> 0），让后台线程定期回收过期 session。

- **client 侧**：model client 内部的 ``RpcClient`` 以 ``enable_sessions=True`` 构造，连接时自动向 server 注册 session。 ``session_id`` 由 facade 从连接派生并注入 server 端 handler，客户端 **不传**，也不应在 ``predict`` 的 ``options`` 里伪造 ``session_ids``。

- **primitives 侧**：任务开始前调用 ``reset_session`` 清空上一回合残留的策略状态，保证连续多次运行之间状态不串。

添加 WAM
----------

1. 在 ``rpent/robots/components/`` 下创建模型包。``server.py`` 继承 ``BaseWAMFacade`` 并实现 ``predict_native``，必须提供 ``(encode, decode)`` adapter 函数对。
2. 使用 ``WAMAdapterSpec`` 在模型的 ``ADAPTERS`` 中注册支持的平台。模型元数据工厂声明 checkpoint 身份、相机与状态要求、原生控制约定及可执行 chunk 长度。encode 将物理观测转换为模型输入，decode 将预测还原为声明的平台动作格式。
3. 观测和控制契约匹配时，复用平台的 ``wam_client.py``。``BaseWAMClient`` 在初始化时调用一次 ``wam.get_model_meta``，将结果保存在 ``model_meta``；平台在执行预测前检查控制兼容性。
4. 在 ``runtime_config.py`` 中补充部署参数，并在 ``am_runtime.WAM_BACKENDS`` 注册后端。在模型包中声明是否使用会话；推理保留会话历史时实现 ``reset_native``。
5. 在平台运行时与工具执行中接入所选后端。回合边界重置模型历史，结束后关闭客户端。每个 toolkit 都必须持有 ``MemoryManager``。

支持的模型与平台组合、协议字段、部署命令和集成验证见 :doc:`wam_backends`。新组合需要兼容的权重与输入输出转换，注册本身不能证明兼容性。未来预测和规划操作留待后续开发。
