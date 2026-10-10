WAM 后端
========

RPent 将平台观测和动作执行与模型推理解耦。Cosmos Policy 支持 LIBERO、LIBERO-Pro；Fast-WAM 支持 LIBERO、LIBERO-Pro 和 RoboTwin。每个组合都需要匹配的 checkpoint。部署示例见 :doc:`../simulators/libero` 和 :doc:`../simulators/robotwin`。

架构与职责
----------

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
   * - ``rpent/robots/components/wam_rpc_protocol.py``
     - 用 ``TypedDict`` 声明请求、能力和预测结果的字典字段。
   * - ``wam_control_spec.py``
     - 声明平台 client 与模型 adapter 共用的控制字典。
   * - ``wam_client_base.py`` / ``wam_facade_base.py``
     - 复用 RPent RPC，检查边界数据，管理 reset 与会话钩子。
   * - ``wam_runtime.py``
     - 注册后端，连接外部服务或启动自己管理的模型进程。
   * - ``<model>/runtime.py`` / ``server.py``
     - 管理启动参数、模型加载和原生推理。
   * - ``<model>/adapter/{__init__,encode,decode}.py``
     - 注册成对转换，构造模型输入并还原平台动作。

物理观测可由 VLA 和 WAM 复用，模型编码仍由各自 adapter 实现。例如 Cosmos 将 LIBERO 的 xyzw 四元数拼入 9 维状态，Fast-WAM 则转换成轴角后组成 8 维状态。RoboTwin 分开提供控制器关节目标与实测关节位置；Fast-WAM 使用对应原生 ``joint_action.vector`` 的目标值。

.. _wam-protocol:

通信与生命周期
--------------

RPC 路由为 ``wam.capabilities``、``wam.predict`` 和 ``wam.reset``。请求包含 ``images``、``state``、``instruction``、``embodiment`` 和 ``action_space``。能力声明包含 ``control`` 控制字典、必需相机与状态维度、可执行 ``chunk_size``、后端与 checkpoint 身份及会话设置。预测结果包含 ``actions``，可选返回 ``future_observation``、``value`` 和 ``metadata``。

消息直接使用已有传输支持的字典和 NumPy 数组，没有对象到通信格式的转换，也没有协议版本字段。client 和 worker 应从同一份代码部署。连接时，平台将服务的控制字典与本地支持的定义直接比较，并缓存选择；更换服务或控制方式时创建新 client。服务端检查收到的观测，客户端在执行前检查动作形状、chunk 长度及有限数值。decoder 保留夹爪二值化、动作前缀选择等转换之前的必要检查，避免转换隐藏无效输出。

平台 client 用 ``predict()`` 返回动作，用 ``predict_result()`` 返回完整字典。回合复位同时清空环境与模型历史。RoboTwin 复用环境首次初始化的 reset，并在构造 toolkit 时清空模型历史。连接 client 不会复位已有回合；直接使用 client 时，在首次预测前调用 ``reset()``，结束后调用 ``close()``。带会话的 worker 实现 ``reset_native``；当前两个模型包均只声明一次 ``USES_SESSIONS = False``，供启动逻辑和能力声明共用。

模型依赖安装在单独准备的环境中，RPent 的公共安装不会安装 Cosmos Policy 或 Fast-WAM 的依赖。自己启动的 worker 复用已有 daemon 生命周期；外部服务在 RPent 停止自己管理的服务后仍保持运行。

扩展方式
--------

增加模型时，新建包含 ``runtime.py``、``server.py`` 及配对 ``adapter/encode.py``、``adapter/decode.py`` 的包。在 ``wam_runtime.BACKENDS`` 注册后端，在模型的 ``ADAPTERS`` 中注册支持的平台。``WAMAdapterSpec`` 位于 ``wam_facade_base.py``，组合 encode、decode 和能力构造函数。已有平台的物理观测与控制定义满足要求时，可直接复用平台 client。

增加平台时，提供物理观测、原生控制定义、平台 client 及工具和运行时接线。每个模型仍需匹配的 checkpoint 和配对转换；注册本身不能证明权重兼容。RoboTwin 已定义 EEF16 控制方式，但当前 Fast-WAM adapter 使用 qpos14。Cosmos Policy 尚无 RoboTwin adapter。

``future_observation`` 和 ``value`` 为模型预测预留字段。独立的预测与规划 RPC、规划器使用预测结果、WAM 记忆接入尚未实现。当前 WAM 评测不支持 Flash 和探索模式。传入 ``memory=None`` 时不注册记忆文件工具，仍保留 ``finish`` 和平台工具。

离线测试覆盖 adapter、控制匹配、无效观测与动作，以及真实本机 HTTP/socket 会话生命周期。显式启用的 GPU 测试覆盖实模推理和有限步数的仿真器、toolkit 执行，命令见 ``tests/README.md``。这些测试通过说明接入链路可用，不代表 benchmark 成功率。

部署模型服务
------------

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

让 RPent 管理 worker 时，将平台命令中的 ``--wam-endpoint`` 替换为 ``--wam-checkpoint`` 和 ``--wam-python``。Cosmos 还需 ``--wam-root``，``COSMOS_POLICY_PYTHON`` 与 ``COSMOS_POLICY_ROOT`` 可提供默认值。Fast-WAM 还需 ``--wam-config`` 和 ``--wam-dataset-stats``。endpoint 和 checkpoint 两种模式互斥。外部服务的模型参数在其启动时配置，各 worker 的 ``--help`` 可列出启动选项。
