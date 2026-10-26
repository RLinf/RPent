添加动作模型
================

接入 VLA 或 WAM 服务，并将预测结果交给平台的原生动作执行器。工具声明与状态采集见 :doc:`add_primitive`，平台发现和运行时接线见 :doc:`add_robot`。

共享架构
--------

模型包负责权重与模型专用预处理；平台负责物理观测、控制方式和执行工具。公共 AM 基类复用 RPC 传输与推理转发。VLA 保留模型专用消息格式，WAM 使用 ``am_contracts.py`` 中的物理观测与原生控制声明。

``am_runtime.add_am_args(parser, platform=...)`` 统一注册当前平台支持的后端与公共部署参数。``AMConfig.spawn_worker()`` 将 ``--am-cuda-device`` 传入子进程的 ``CUDA_VISIBLE_DEVICES``，未设置时使用 ``--cuda-device``，两者均未设置时保留继承的 GPU 环境。平台在自己的 ``robot_spec.py`` 中注册 ``--env-cuda-device``；环境 GPU 不会成为模型 GPU 的默认值。外部 endpoint 的 GPU 由部署服务时的配置决定。

下面以 WAM 执行链路说明平台与模型之间的职责边界：

.. code-block:: text

   CLI / Dashboard
     -> robots/<platform>/robot_spec.py: configuration and service startup
     -> toolkit.py: register platform tools
     -> wam_act: observe, predict, execute, record
     -> observation.py + wam_client.py: physical observations and control identity
     -> BaseWAMClient -> existing HTTP/socket RPC -> BaseWAMFacade
     -> <model>/adapter/encode.py -> predict_native -> adapter/decode.py
     -> action chunk -> platform env_client.chunk_step -> simulator

``wam_act`` 在 LIBERO 的 ``tools.py`` 和 RoboTwin 的 ``primitives.py`` 中通过 ``@tool`` 声明，返回 ``ToolResult(data=...)``。平台已有 toolkit 负责工具注册、取消、状态采集和录制。

.. list-table:: 文件职责
   :header-rows: 1

   * - 位置
     - 职责
   * - ``robots/<platform>/observation.py``
     - 提取具名物理传感器与状态，不做模型归一化。
   * - ``robots/<platform>/control.py``
     - 定义原生动作的字段顺序、含义和执行模式。
   * - ``robots/<platform>/wam_client.py``
     - 选择兼容控制方式，给物理观测附加执行身份。
   * - ``rpent/robots/components/am_contracts.py``
     - 分节声明共享物理观测、原生控制，以及 WAM 专用的能力、请求和结果字典。
   * - ``am_client_base.py``
     - 定义公共 ``BaseAMClient`` 及 VLA/WAM 子类，共享推理转发和超时处理，各自保留协议参数与检查。
   * - ``am_facade_base.py``
     - 通过 ``BaseAMFacade`` 共享推理路由注册和 RPC 会话初始化；子类分别负责 VLA/WAM 行为。
   * - ``am_runtime.py``
     - 统一注册公共部署参数，将 VLA/WAM 配置解析为 ``AMConfig``，按指定 Python、工作目录和 GPU 启动 worker。
   * - ``rpent/robots/components/pi05_vla_client.py``
     - 为 Pi0.5 适配平台观测和模型输出；其他 VLA 接入提供各自的模型客户端。
   * - ``<model>/runtime_config.py``
     - 补充 WAM 模型专属参数、默认值和配置要求。
   * - ``<model>/server.py``
     - 加载 WAM 模型，实现原生推理。
   * - ``<model>/adapter/{__init__,encode,decode}.py``
     - 注册成对转换，构造模型输入并还原平台动作。

``BaseVLAClient`` 和 ``BaseWAMClient`` 均继承 ``BaseAMClient``，三个类定义在同一个 ``am_client_base.py`` 中，各自保留原有推理参数和返回格式。WAM 能力检查及 reset/close 留在 ``BaseWAMClient``；VLA 生命周期方法仍由支持这些操作的具体集成提供。

``BaseVLAFacade`` 和 ``BaseWAMFacade`` 同样在 ``am_facade_base.py`` 内继承 ``BaseAMFacade``。VLA 保留模型原生消息和扩展路由，WAM 负责能力检查、配对 adapter 和 reset 钩子。``AMObservation`` 与 ``AMControlSpec`` 描述两类模型可共享的平台物理数据；``WAMRequest`` 在物理观测上增加执行身份，现有 VLA 通信格式保持不变。

物理观测可由 VLA 和 WAM 复用，模型编码仍由各自 adapter 实现。例如 Cosmos 将 LIBERO 的 xyzw 四元数拼入 9 维状态，Fast-WAM 则转换成轴角后组成 8 维状态。RoboTwin 分开提供控制器关节目标与实测关节位置；Fast-WAM 使用对应原生 ``joint_action.vector`` 的目标值。

平台 ``robot_spec.py`` 通过回调连接各层启动逻辑。WAM worker 的调用链如下：

.. code-block:: text

   robot_spec._init_runtime()
     -> am_runtime.select_am(args, platform): resolve AMConfig and model options
     -> robots.runtime.try_spawn_server(spawn_fn)
          -> am_runtime.start_wam_service(config, output_dir)
               -> AMConfig.spawn_worker() -> ProcessDaemon.start()
               -> make_rpc_client()
               -> return daemon + RPC client
          -> register owned daemon
     -> robots.runtime.try_wait_server(post_fn)
          -> wait for readiness
          -> create platform WAM client and check control compatibility
          -> report ready to Dashboard

VLA 启动函数也使用 ``AMConfig.spawn_worker()``。平台保留各自的服务命令和传输方式，包括 LingBot 的原生 WebSocket 连接与 RoboCasa 的带会话 RPC。``ProcessDaemon`` 在启动失败时关闭日志；公共 ``robots/runtime.py`` 在资源交接后负责等待就绪、组件状态和进程组清理。连接外部 endpoint 时不返回自有 daemon。

``AMConfig`` 保存解析后的后端、平台、endpoint、checkpoint、Python、工作目录、GPU 和额外的 worker 参数。默认值解析不会修改原始 Namespace，避免 Dashboard 再次使用时把默认值误认为显式参数。WAM 的 ``runtime_config.py`` 补充模型专属默认值和 worker 参数；VLA 平台启动函数保留各自的原生命令构造。

添加 VLA
--------

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

已有服务的部署方式与 endpoint 示例见 :doc:`../guides/advanced_deployment`。有状态模型还需要实现下面的按客户端隔离与重置逻辑。

按客户端隔离策略状态
~~~~~~~~~~~~~~~~~~~~

大多数 VLA 后端是无状态的：``predict`` 只做推理，不保存各客户端的中间状态，``session_id`` 可以忽略。但有些模型带按客户端隔离的策略状态（如 RLDX-1 的 memory/RTC），同一个 ``vla_server`` 服务多个客户端时，不同客户端的策略状态会互相污染，必须按 session 隔离。接入分三块：

- **facade 侧**：构造 ``BaseVLAFacade`` 子类时传 ``enable_sessions=True`` 和 ``session_timeout_s``，并实现 ``_on_session_drop``——session 结束（客户端调用 ``session.close`` RPC 或空闲超时）时在这里清理该客户端的策略状态。需要显式重置时，额外提供 ``reset_session`` RPC（只清策略状态，不销毁 session）。``serve`` 必须传 ``session_sweep_s`` （> 0），让后台线程定期回收过期 session。

- **client 侧**：model client 内部的 ``RpcClient`` 以 ``enable_sessions=True`` 构造，连接时自动向 server 注册 session。 ``session_id`` 由 facade 从连接派生并注入 server 端 handler，客户端 **不传**，也不应在 ``predict`` 的 ``options`` 里伪造 ``session_ids``。

- **primitives 侧**：任务开始前调用 ``reset_session`` 清空上一回合残留的策略状态，保证连续多次运行之间状态不串。

添加 WAM
--------

Cosmos Policy 支持 LIBERO、LIBERO-Pro；Fast-WAM 支持 LIBERO、LIBERO-Pro 和 RoboTwin。每个模型与平台组合都需要匹配的 checkpoint。

Cosmos 后端使用 NVIDIA 发布的 ``Cosmos-Policy-LIBERO-Predict2-2B`` 权重及其官方 LIBERO 推理接口。RLinf 的 Cosmos3 接入使用另一种模型和 SGLang ``/v1/actions/generations`` 服务，需要独立的后端适配与兼容权重。

RPent ``373e900`` 的历史纯策略评测在 LIBERO Spatial 上取得 **30/30（100%）** 原生任务成功：覆盖全部 10 个任务，每任务使用初始状态 0/1/2，每回合最多 220 个动作。实验使用 A100 40GB、5 步去噪、策略 seed 1，每次预测 16 个动作，不生成未来状态或价值，也没有 planner 或记忆指导；评测程序在成功时停止执行 chunk 中的剩余动作。这是旧实现的小规模评测，不代表当前 AM 重构或其他套件的成功率。下文的有限步数 E2E 检查单独验证接入链路。

1. 在 ``rpent/robots/components/`` 下创建模型包。``server.py`` 继承 ``BaseWAMFacade`` 并实现 ``predict_native``，必须提供 ``(encode, decode)`` adapter 函数对。
2. 使用 ``am_facade_base.py`` 中的 ``WAMAdapterSpec`` 在模型的 ``ADAPTERS`` 中注册支持的平台。模型元数据工厂声明 checkpoint 身份、相机与状态要求、原生控制约定及可执行 chunk 长度。encode 将物理观测转换为模型输入，decode 将预测还原为声明的平台动作格式。
3. 观测和控制契约匹配时，复用平台的 ``wam_client.py``。``BaseWAMClient`` 在初始化时调用一次 ``wam.get_model_meta``，将结果保存在 ``model_meta``；平台在执行预测前检查控制兼容性。
4. 在 ``runtime_config.py`` 中补充部署参数，并在 ``am_runtime.WAM_BACKENDS`` 注册后端。在模型包中声明是否使用会话；推理保留会话历史时实现 ``reset_native``。
5. 在平台运行时与工具执行中接入所选后端。回合边界重置模型历史，结束后关闭客户端。每个 toolkit 都必须持有 ``MemoryManager``。

.. _wam-protocol:

通信与生命周期
~~~~~~~~~~~~~~

RPC 路由为 ``wam.get_model_meta``、``wam.predict`` 和 ``wam.reset``。请求包含 ``images``、``state``、``instruction``、``embodiment`` 和 ``action_space``。模型元数据包含 ``control`` 控制字典、必需相机与状态维度、可执行 ``chunk_size``、后端与 checkpoint 身份及会话设置。预测结果包含 ``actions``，可选返回 ``future_observation``、``value`` 和 ``metadata``。

``BaseWAMClient`` 在构造时查询一次模型元数据并保存在 ``model_meta``，后续预测直接使用该快照。

消息直接使用已有传输支持的字典和 NumPy 数组，没有对象到通信格式的转换，也没有协议版本字段。client 和 worker 应从同一份代码部署。连接时，平台将服务的控制字典与本地支持的定义直接比较，并缓存选择；更换服务或控制方式时创建新 client。服务端检查收到的观测，客户端在执行前检查动作形状、chunk 长度及有限数值。decoder 保留夹爪二值化、动作前缀选择等转换之前的必要检查，避免转换隐藏无效输出。

平台 client 用 ``predict()`` 返回动作，用 ``predict_result()`` 返回完整字典。回合复位同时清空环境与模型历史。RoboTwin 复用环境首次初始化的 reset，并在构造 toolkit 时清空模型历史。连接 client 不会复位已有回合；直接使用 client 时，在首次预测前调用 ``reset()``，结束后调用 ``close()``。带会话的 worker 实现 ``reset_native``；当前两个模型包均只声明一次 ``USES_SESSIONS = False``，供启动逻辑和能力声明共用。

模型依赖安装在单独准备的环境中，RPent 的公共安装不会安装 Cosmos Policy 或 Fast-WAM 的依赖。自己启动的 worker 复用已有 daemon 生命周期；外部服务在 RPent 停止自己管理的服务后仍保持运行。

添加平台与检查兼容性
~~~~~~~~~~~~~~~~~~~~

增加平台时，提供物理观测、原生控制定义、平台 client 及工具和运行时接线。每个模型仍需匹配的 checkpoint 和配对转换；注册本身不能证明权重兼容。RoboTwin 已定义 EEF16 控制方式，但当前 Fast-WAM adapter 使用 qpos14。Cosmos Policy 尚无 RoboTwin adapter。

``future_observation`` 和 ``value`` 为模型预测预留字段，独立的预测与规划 RPC 以及规划器使用预测结果尚未实现。每个 toolkit 必须持有 ``MemoryManager``，并注册记忆文件工具。WAM 默认使用平台记忆根目录下以后端命名的本地子目录，可通过 ``--memory-dir`` 指定目录，语料允许为空。WAM 尚未配置兼容的 HF 经验库，当前评测不支持 HF 记忆、Flash 和探索模式。

.. _action-model-deployment:

动作模型部署与任务运行
----------------------

VLA 与 WAM 共用六个部署参数：

.. list-table:: 公共动作模型参数
   :header-rows: 1

   * - 参数
     - 作用
   * - ``--am-backend``
     - 选择当前平台支持的一个动作模型。
   * - ``--am-endpoint``
     - 连接已经部署的模型服务。
   * - ``--am-checkpoint``
     - 指定本地 worker 使用的权重。
   * - ``--am-python``
     - 指定模型环境的 Python；VLA 默认使用当前解释器。
   * - ``--am-root``
     - 指定 worker 工作目录；默认使用 RPent 仓库目录，Cosmos 则需要已准备好的源码目录。
   * - ``--am-cuda-device``
     - 指定本地模型的可见 GPU；未设置时使用 ``--cuda-device``，再回退到继承的环境。

Franka 和 YAM 连接外部模型服务，帮助中不展示本地部署参数。使用 ``--am-endpoint`` 时，显式传入公共或模型专属的本地部署参数会报错。未选中模型和外部模型的环境变量默认值不会生效。显存允许时，模型可与其他服务共用 GPU。

独立部署模型服务
~~~~~~~~~~~~~~~~

反复运行任务时，先用下面的命令启动一次 worker，再在每次 RPent 运行时通过 ``--am-endpoint`` 连接。RPent 退出后模型服务继续运行，使用结束后需单独停止。也可以使用后面的单次运行示例，让 RPent 自动启动和停止 worker。

先准备官方模型环境、checkpoint 和辅助文件，并安装 RPent 运行时依赖；``PYTHONPATH`` 只让该环境能导入当前代码，不会安装依赖。仿真器保留在 RPent 环境中。在已准备好的 Cosmos 源码目录启动：

.. code-block:: bash

   PYTHONPATH=/path/to/RPent /path/to/cosmos/.venv/bin/python \
     -m rpent.robots.components.cosmos_policy.server \
     --checkpoint /path/to/cosmos-libero --cuda-device 0 --port 8116

checkpoint 目录包含 ``Cosmos-Policy-LIBERO-Predict2-2B.pt``、``libero_dataset_statistics.json`` 和 ``libero_t5_embeddings.pkl``。可用 ``--dataset-stats``、``--text-embeddings`` 覆盖辅助路径。默认使用 5 步动作去噪、seed 1，每次预测 16 个动作，不生成未来状态和价值。``--cached-instructions-only`` 拒绝未缓存指令，否则由官方运行时处理未缓存文字。``--predict-future`` 可开启可选输出，但 RPent 尚不利用这些输出规划。

在已准备好的 Fast-WAM 环境中启动：

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/path/to/RPent \
     /path/to/fast-wam/venv/bin/python \
     -m rpent.robots.components.fast_wam.server \
     --platform libero --checkpoint /path/to/fast-wam-libero.pt \
     --config /path/to/worker.yaml --dataset-stats /path/to/dataset_stats.json \
     --port 8117

LIBERO worker 的模型环境需要安装 RLinf ``release/v0.4`` 的 Fast-WAM 接入，以及其 ``requirements/install.sh`` 指定的 Fast-WAM 版本（目前为 ``45d8e1458921d83f8ad6cf9ce993d371208dabd0``）。在 RLinf 仓库执行 ``bash requirements/install.sh embodied --model fastwam`` 配套安装；其他 Fast-WAM 版本的 VAE 接口可能不兼容。worker 调用 RLinf 的 ``get_model`` 和 ``predict_action_batch``，图像缩放与拼接、状态归一化、推理及 LIBERO 动作还原均由 RLinf 负责。RPent 只将物理观测映射到 RLinf 输入字段，并从返回动作中移除单环境 batch 维度。RLinf 的该策略尚不支持 RoboTwin 的三相机/qpos14 接口，因此 RoboTwin 保留官方 Fast-WAM 推理路径。

RoboTwin 使用 ``--platform robotwin``，并更换为对应 checkpoint 和配置。YAML 提供解析好的 Hydra ``model``、``processor``，后者来自官方配置的 ``data.train.processor``，还需包含 ``action_horizon``、``execute_steps``、``video_size`` 和 ``concat``。LIBERO 的 ``inference_options`` 配置 RLinf 策略的去噪步数、seed、sigma shift、text CFG、negative prompt、随机数设备和 tiling，不支持 ``compile_action_infer=True``。RoboTwin 将选项传给官方 ``model.infer_action``。选项不能覆盖编码后的观测或 ``action_horizon``。处理未缓存指令时需在模型配置中启用文本编码器。``execute_steps`` 为正数且不超过 ``action_horizon``，是对外声明的可执行 ``chunk_size``。``binarize_gripper`` 仅适用于 LIBERO。未来视频生成、时序动作集成和自动 Hydra 配置组合尚未实现。

连接已有服务运行任务
~~~~~~~~~~~~~~~~~~~~

准备好平台资产和 planner 凭据后，在 RPent 仿真器环境中运行下面任一示例（前置配置见 :doc:`../simulators/libero`）：

.. code-block:: bash

   # Cosmos worker already running on port 8116.
   rpent --robot libero --suite libero_spatial --task 0 \
     --am-backend cosmos-policy --am-endpoint http://127.0.0.1:8116 \
     --memory-profile local --planner api --model openai:gpt-5.5

   # Fast-WAM worker already running on port 8117.
   rpent --robot libero --suite libero_spatial --task 0 \
     --am-backend fast-wam --am-endpoint http://127.0.0.1:8117 \
     --memory-profile local --planner api --model openai:gpt-5.5

之后修改 task 或 suite 即可继续使用兼容的 worker。模型路径和推理选项在部署服务时配置，无需在每次任务命令中重复填写。planner 模型可换成你已配置的供应商支持的模型。RoboTwin 的任务命令见 :doc:`../simulators/robotwin`。

由 RPent 在每次运行时启动服务
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

下面的示例会在每次运行时启动新的 worker，并在退出时停止它。模型环境和文件与独立部署相同；RPent 不会自动下载权重或安装模型依赖。将示例路径替换为实际部署路径：

.. code-block:: bash

   rpent --robot libero --suite libero_spatial --task 0 \
     --am-backend pi05 --am-checkpoint /path/to/pi05-libero \
     --am-python /path/to/pi05-env/bin/python \
     --planner api --model openai:gpt-5.5

   rpent --robot libero --suite libero_spatial --task 0 \
     --am-backend cosmos-policy --am-checkpoint /path/to/cosmos-libero \
     --am-python /path/to/cosmos/.venv/bin/python --am-root /path/to/cosmos \
     --memory-profile local --planner api --model openai:gpt-5.5

   rpent --robot libero --suite libero_spatial --task 0 \
     --am-backend fast-wam --am-checkpoint /path/to/fast-wam-libero.pt \
     --am-python /path/to/fast-wam/venv/bin/python \
     --fast-wam-config /path/to/worker.yaml \
     --fast-wam-dataset-stats /path/to/dataset_stats.json \
     --memory-profile local --planner api --model openai:gpt-5.5

``--am-endpoint`` 与 ``--am-checkpoint`` 互斥。Cosmos 需要 ``--am-root``，也可用 ``COSMOS_POLICY_PYTHON`` 和 ``COSMOS_POLICY_ROOT`` 分别提供 Python 路径和工作目录的默认值。Fast-WAM 必须同时提供 ``--fast-wam-config`` 和 ``--fast-wam-dataset-stats``。``--am-cuda-device`` 与 ``--env-cuda-device`` 独立；未指定模型 GPU 时先使用共享的 ``--cuda-device``，两者均未指定时继承 CUDA 可见设备设置。显存允许时，各服务可指定相同设备，共用一张 GPU。

查看可用参数
~~~~~~~~~~~~

帮助按后端选择、连接服务、本地部署、所选模型分组展示参数。在 ``--help`` 前指定模型即可查看对应选项：

.. code-block:: bash

   rpent --robot libero --am-backend cosmos-policy --help
   rpent --robot robotwin --am-backend fast-wam --help

省略 ``--am-backend`` 时展示平台默认模型。公共参数统一使用 ``--am-*``，模型专属参数保留 ``--cosmos-*``、``--fast-wam-*``、``--lingbot-*`` 和 ``--pi05-*``。显式传入其他模型的参数会报错。独立 worker 的 ``--help`` 列出它自己的参数，使用 ``--config``、``--predict-future`` 等不带模型前缀的名称。

旧命令通过隐藏别名继续兼容：

.. list-table:: 模型参数名称
   :header-rows: 1

   * - 旧参数
     - 推荐参数
   * - ``--vla-backend``、``--wam-backend``
     - ``--am-backend``
   * - RoboDojo 的 ``--policy-backend rlinf``
     - ``--am-backend pi05``
   * - ``--vla-endpoint``、``--wam-endpoint``
     - ``--am-endpoint``
   * - ``--vla-model-path``、``--wam-checkpoint``
     - ``--am-checkpoint``
   * - ``--pi05-python``、``--wam-python``
     - ``--am-python``
   * - ``--wam-root``
     - ``--am-root``
   * - ``--vla-cuda-device``、``--wam-cuda-device``
     - ``--am-cuda-device``
   * - ``--vla-repo-id``
     - ``--pi05-repo-id``
   * - ``--wam-text-embeddings``
     - ``--cosmos-text-embeddings``
   * - ``--wam-predict-future``
     - ``--cosmos-predict-future``
   * - ``--wam-cached-instructions-only``
     - ``--cosmos-cached-instructions-only``
   * - ``--wam-config``
     - ``--fast-wam-config``
   * - ``--wam-dataset-stats``
     - ``--fast-wam-dataset-stats``

测试
----

WAM 沿用 VLA/SAM 的测试组织。``test_am.py`` 覆盖公共启动、VLA 转发、WAM 边界和部署配置；``test_wam_adapters.py`` 在 CPU 上检查相机与状态编码、原生动作还原。平台工具行为保留在各平台的单测中。``test_wam_loopback.py`` 复用已有 HTTP/socket fixture，检查会话隔离、复位和清理。

真实模型测试放在各平台的 ``test_components.py`` 和 ``test_policy_chain.py``。LIBERO 对 Cosmos Policy 和 Fast-WAM 参数化复用同一组测试；RoboTwin 使用自己的 Fast-WAM checkpoint。启动 worker 后，在对应仿真器环境运行：

.. code-block:: bash

   RPENT_COSMOS_ENDPOINT=http://127.0.0.1:8116 \
     pytest tests/e2e_tests/libero/test_components.py \
       tests/e2e_tests/libero/test_policy_chain.py -k 'wam or horizon' -v

   RPENT_FAST_WAM_ENDPOINT=http://127.0.0.1:8117 \
     pytest tests/e2e_tests/libero/test_components.py \
       tests/e2e_tests/libero/test_policy_chain.py -k 'wam or horizon' -v

   RPENT_FAST_WAM_ROBOTWIN_ENDPOINT=http://127.0.0.1:8117 \
     pytest tests/e2e_tests/robotwin/test_components.py \
       tests/e2e_tests/robotwin/test_policy_chain.py -k wam -v

没有设置 endpoint 的模型用例会跳过。LIBERO 默认使用 ``libero_spatial``、task 0、seed 0；测试 LIBERO-Pro 时设置 ``RPENT_WAM_SUITE=libero_spatial_task`` 并使用 Pro 资产。组件测试记录 checkpoint、单次推理耗时和原生成功标志。CLI 测试用脚本化 planner 执行有限次数的 ``wam_act``，检查状态产物和 ``finish``。测试使用不含任务指导的本地记忆 fixture，不使用 Qwen，也不估计 benchmark 成功率或记忆收益。这些显式启用的测试独立于标准 GPU 环境准备脚本。

这些测试通过说明接入链路可用，不代表 benchmark 成功率。环境准备和完整测试流程见 ``tests/README.md``。
