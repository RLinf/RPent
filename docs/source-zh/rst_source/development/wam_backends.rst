WAM 后端
========

RPent 将平台观测和动作执行与模型推理解耦。Cosmos Policy 支持 LIBERO、LIBERO-Pro；Fast-WAM 支持 LIBERO、LIBERO-Pro 和 RoboTwin。每个组合都需要匹配的 checkpoint。部署示例见 :doc:`../simulators/libero` 和 :doc:`../simulators/robotwin`。

架构与职责
----------

本地模型服务的 ``--vla-cuda-device``、``--wam-cuda-device`` 统一由 ``am_runtime.add_am_device_arg()`` 注册。``am_device_env()`` 只读取对应模型的参数，未设置时保留继承的 GPU 环境。平台在自己的 ``robot_spec.py`` 中注册 ``--env-cuda-device``，并在启动模型 worker 时传入公共层生成的模型设置；环境 GPU 不会成为模型 GPU 的默认值。外部 endpoint 的 GPU 仍由部署服务时的配置决定。

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
     - 共享 RPent RPC 服务连接、进程启动和失败清理；另设 WAM 后端选择与部署配置部分。
   * - ``<model>/runtime_config.py``
     - 补充模型专属参数、默认值和配置要求。
   * - ``<model>/server.py``
     - 加载模型，实现原生推理。
   * - ``<model>/adapter/{__init__,encode,decode}.py``
     - 注册成对转换，构造模型输入并还原平台动作。

``BaseVLAClient`` 和 ``BaseWAMClient`` 均继承 ``BaseAMClient``，三个类定义在同一个 ``am_client_base.py`` 中，各自保留原有推理参数和返回格式。WAM 能力检查及 reset/close 留在 ``BaseWAMClient``；VLA 生命周期方法仍由支持这些操作的具体集成提供。

``BaseVLAFacade`` 和 ``BaseWAMFacade`` 同样在 ``am_facade_base.py`` 内继承 ``BaseAMFacade``。VLA 保留模型原生消息和扩展路由，WAM 负责能力检查、配对 adapter 和 reset 钩子。``AMObservation`` 与 ``AMControlSpec`` 描述两类模型可共享的平台物理数据；``WAMRequest`` 在物理观测上增加执行身份，现有 VLA 通信格式保持不变。

物理观测可由 VLA 和 WAM 复用，模型编码仍由各自 adapter 实现。例如 Cosmos 将 LIBERO 的 xyzw 四元数拼入 9 维状态，Fast-WAM 则转换成轴角后组成 8 维状态。RoboTwin 分开提供控制器关节目标与实测关节位置；Fast-WAM 使用对应原生 ``joint_action.vector`` 的目标值。

平台 ``robot_spec.py`` 通过回调连接各层启动逻辑：

.. code-block:: text

   robot_spec._init_runtime()
     -> robots.runtime.try_spawn_server(spawn_fn)
          -> WAMConfig.start_service()
               -> <model>.runtime_config.worker_arguments()
               -> am_runtime.start_am_service(daemon, endpoint)
                    -> start worker and return daemon + RPC client
          -> register owned daemon
     -> robots.runtime.try_wait_server(post_fn)
          -> wait for readiness
          -> create platform WAM client and check control compatibility
          -> report ready to Dashboard

``am_runtime.start_am_service()`` 连接外部端点或启动已配置的 daemon，保留会话设置，并在启动失败时清理资源。LIBERO、RoboCasa、RoboDojo、Franka、dual-Franka 和 YAM 中使用 RPent RPC 的 VLA 路径与 WAM 共用此函数；命令构造仍由平台或模型配置负责。LingBot 保留原生 WebSocket 连接路径。公共 ``robots/runtime.py`` 在资源交接后负责等待就绪、组件状态和进程组清理。

``am_runtime.py`` 的 WAM 部分负责 ``--platform``、``--checkpoint``、Python/GPU 选择和部署模式检查；模型 ``runtime_config.py`` 只补充自己的额外配置。VLA 启动参数保持不变。

.. _wam-protocol:

通信与生命周期
--------------

RPC 路由为 ``wam.capabilities``、``wam.predict`` 和 ``wam.reset``。请求包含 ``images``、``state``、``instruction``、``embodiment`` 和 ``action_space``。能力声明包含 ``control`` 控制字典、必需相机与状态维度、可执行 ``chunk_size``、后端与 checkpoint 身份及会话设置。预测结果包含 ``actions``，可选返回 ``future_observation``、``value`` 和 ``metadata``。

消息直接使用已有传输支持的字典和 NumPy 数组，没有对象到通信格式的转换，也没有协议版本字段。client 和 worker 应从同一份代码部署。连接时，平台将服务的控制字典与本地支持的定义直接比较，并缓存选择；更换服务或控制方式时创建新 client。服务端检查收到的观测，客户端在执行前检查动作形状、chunk 长度及有限数值。decoder 保留夹爪二值化、动作前缀选择等转换之前的必要检查，避免转换隐藏无效输出。

平台 client 用 ``predict()`` 返回动作，用 ``predict_result()`` 返回完整字典。回合复位同时清空环境与模型历史。RoboTwin 复用环境首次初始化的 reset，并在构造 toolkit 时清空模型历史。连接 client 不会复位已有回合；直接使用 client 时，在首次预测前调用 ``reset()``，结束后调用 ``close()``。带会话的 worker 实现 ``reset_native``；当前两个模型包均只声明一次 ``USES_SESSIONS = False``，供启动逻辑和能力声明共用。

模型依赖安装在单独准备的环境中，RPent 的公共安装不会安装 Cosmos Policy 或 Fast-WAM 的依赖。自己启动的 worker 复用已有 daemon 生命周期；外部服务在 RPent 停止自己管理的服务后仍保持运行。

扩展方式
--------

增加模型时，新建包含 ``runtime_config.py``、``server.py`` 及配对 ``adapter/encode.py``、``adapter/decode.py`` 的包。在 ``am_runtime.WAM_BACKENDS`` 注册后端，在模型的 ``ADAPTERS`` 中注册支持的平台。``WAMAdapterSpec`` 位于 ``am_facade_base.py``，组合 encode、decode 和能力构造函数。已有平台的物理观测与控制定义满足要求时，可直接复用平台 client。

增加平台时，提供物理观测、原生控制定义、平台 client 及工具和运行时接线。每个模型仍需匹配的 checkpoint 和配对转换；注册本身不能证明权重兼容。RoboTwin 已定义 EEF16 控制方式，但当前 Fast-WAM adapter 使用 qpos14。Cosmos Policy 尚无 RoboTwin adapter。

``future_observation`` 和 ``value`` 为模型预测预留字段，独立的预测与规划 RPC 以及规划器使用预测结果尚未实现。每个 toolkit 必须持有 ``MemoryManager``，并注册记忆文件工具。WAM 默认使用平台记忆根目录下以后端命名的本地子目录，可通过 ``--memory-dir`` 指定目录，语料允许为空。WAM 尚未配置兼容的 HF 经验库，当前评测不支持 HF 记忆、Flash 和探索模式。

离线测试覆盖 adapter、控制匹配、无效观测与动作，以及真实本机 HTTP/socket 会话生命周期。显式启用的 GPU 测试覆盖实模推理和有限步数的仿真器、toolkit 执行，命令见 ``tests/README.md``。这些测试通过说明接入链路可用，不代表 benchmark 成功率。

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

部署模型服务
------------

反复运行任务时，先用下面的命令启动一次 worker，再在每次 RPent 运行时通过 ``--wam-endpoint`` 连接。RPent 退出后模型服务继续运行，使用结束后需单独停止。也可以使用后面的单次运行示例，让 RPent 自动启动和停止 worker。

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

RoboTwin 使用 ``--platform robotwin``，并更换为对应 checkpoint 和配置。YAML 提供解析好的 Hydra ``model``、``processor``，后者来自官方配置的 ``data.train.processor``，还需包含 ``action_horizon``、``execute_steps``、``video_size`` 和 ``concat``。可选 ``inference_options`` 传给 ``model.infer_action``，不能覆盖编码后的观测或 ``action_horizon``。处理未缓存指令时需在模型配置中启用文本编码器。``execute_steps`` 为正数且不超过 ``action_horizon``，是对外声明的可执行 ``chunk_size``。``binarize_gripper`` 仅适用于 LIBERO。未来视频生成、时序动作集成和自动 Hydra 配置组合尚未实现。

连接已有服务运行任务
--------------------

准备好平台资产和 planner 凭据后，在 RPent 仿真器环境中运行下面任一示例（前置配置见 :doc:`../simulators/libero`）：

.. code-block:: bash

   # Cosmos worker already running on port 8116.
   rpent --robot libero --suite libero_spatial --task 0 \
     --wam-backend cosmos-policy --wam-endpoint http://127.0.0.1:8116 \
     --memory-profile local --planner api --model openai:gpt-5.5

   # Fast-WAM worker already running on port 8117.
   rpent --robot libero --suite libero_spatial --task 0 \
     --wam-backend fast-wam --wam-endpoint http://127.0.0.1:8117 \
     --memory-profile local --planner api --model openai:gpt-5.5

之后修改 task 或 suite 即可继续使用兼容的 worker。模型路径和推理选项在部署服务时配置，无需在每次任务命令中重复填写。planner 模型可换成你已配置的供应商支持的模型。RoboTwin 的任务命令见 :doc:`../simulators/robotwin`。

由 RPent 在每次运行时启动服务
----------------------------

下面两种方式会在每次运行时启动新的 worker，并在退出时停止它。模型环境和文件与独立部署相同；RPent 不会自动下载权重或安装模型依赖。将示例路径替换为实际部署路径：

.. code-block:: bash

   rpent --robot libero --suite libero_spatial --task 0 \
     --wam-backend cosmos-policy --wam-checkpoint /path/to/cosmos-libero \
     --wam-python /path/to/cosmos/.venv/bin/python --wam-root /path/to/cosmos \
     --memory-profile local --planner api --model openai:gpt-5.5

   rpent --robot libero --suite libero_spatial --task 0 \
     --wam-backend fast-wam --wam-checkpoint /path/to/fast-wam-libero.pt \
     --wam-python /path/to/fast-wam/venv/bin/python \
     --fast-wam-config /path/to/worker.yaml \
     --fast-wam-dataset-stats /path/to/dataset_stats.json \
     --memory-profile local --planner api --model openai:gpt-5.5

``--wam-endpoint`` 与 ``--wam-checkpoint`` 互斥。Cosmos 需要 ``--wam-root``，也可用 ``COSMOS_POLICY_PYTHON`` 和 ``COSMOS_POLICY_ROOT`` 分别提供 Python 路径和工作目录的默认值。Fast-WAM 必须同时提供 ``--fast-wam-config`` 和 ``--fast-wam-dataset-stats``。``--wam-cuda-device`` 与 ``--env-cuda-device`` 独立；未指定模型 GPU 时，worker 继承 CUDA 可见设备设置，环境 GPU 不会成为模型 GPU 的默认值。

查看可用参数
------------

``rpent --robot libero --help`` 和 ``rpent --robot robotwin --help`` 按后端选择、连接服务、本地部署、具体模型分组显示 WAM 参数。公共参数使用 ``--wam-*``，Cosmos 专用参数使用 ``--cosmos-*``，Fast-WAM 专用参数使用 ``--fast-wam-*``。RPent 命令中的模型专用参数仅用于本地部署。独立 worker 的 ``--help`` 列出它自己的参数，使用 ``--config``、``--predict-future`` 等不带模型前缀的名称。

旧命令通过隐藏别名继续兼容：

.. list-table:: 模型参数名称
   :header-rows: 1

   * - 旧参数
     - 推荐参数
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
