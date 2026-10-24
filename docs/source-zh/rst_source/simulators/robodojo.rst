RoboDojo
========

通过 RPent 在 RoboDojo 中执行双臂桌面操作任务。仿真器基于 Isaac Sim / IsaacLab，采用 ARX-X5 双机械臂平台。策略动作可选择 VPP2（默认）或 Awomo05。VPP2 模式下可使用原生运动原语和 SAM3 分割；Awomo 混合模式通过对应工具组启用这些能力。

概览
----

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 动作模型

      VPP2（默认）、Awomo05 和原生动作原语

   .. grid-item-card:: 规划器

      ``api``、``claude_code``、``codex``；Awomo 协作模式支持 ``api`` 或 ``codex``。

   .. grid-item-card:: 任务

      瓶子入桶、笔筒整理、碗堆叠等桌面操作。

   .. grid-item-card:: 硬件

      Linux、NVIDIA GPU；Python 3.11；Isaac Sim 5.1。

任务
~~~~

`RoboDojo 仿真任务目录 <https://robodojo-benchmark.com/doc/sim-tasks/>`_ 将 42 个基准任务分为五类能力：

.. list-table::
   :header-rows: 1

   * - 类别
     - 任务数
     - 考察内容
   * - 泛化
     - 12
     - 不同物体和布局，包括随机变体。
   * - 记忆
     - 6
     - 状态追踪和顺序记忆。
   * - 精细操作
     - 8
     - 空间约束严格的操作。
   * - 长程任务
     - 8
     - 需要完成多个子目标的多步骤任务。
   * - 开放任务
     - 8
     - 开放式或依赖语言、图像指令的任务。

观测与动作
~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - 项目
     - 说明
   * - 观测
     - 头部和左右腕部相机的 RGB-D 图像、相机标定，以及双臂关节状态、末端执行器位姿和夹爪开合状态。深度与标定用于几何查询，Awomo 策略执行本身不要求深度。
   * - 动作
     - VPP2 模式下可使用 ``vpp2_act`` 和原生运动原语（``move_to``、``move_delta``、``rotate_wrist``、``set_gripper``、``release``、``hold``）。Awomo05 模式下可使用 ``run_policy``；启用 ``--vla-agent`` 后可添加所选运动和感知工具。``finish`` 不能覆盖原生任务判定。
   * - 奖励与成功判定
     - 任务是否成功由 RoboDojo 原生环境判定，结果记录在 ``result.json`` 的 ``success`` 字段中。
   * - 任务指令
     - 来自当前场景的 RoboDojo 任务描述，通过 ``view_env_state`` 读取。Awomo 在整回合中使用官方指令；VPP2 接受子任务指令。

安装与资源准备
--------------

NVIDIA 驱动版本建议参考 `Isaac Sim 5.1 官方系统要求 <https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html>`_。\
宿主机还需具备构建 cuRobo 所需的 CUDA toolkit/NVCC 和编译工具链。从 RPent 根目录创建独立环境并安装依赖：

.. code-block:: bash

   uv venv --python 3.11 .venv-robodojo
   source .venv-robodojo/bin/activate
   uv pip install -e ".[robodojo-sim,sam3]" --extra-index-url https://pypi.nvidia.com \
     --override requirements/robodojo-override.txt

.. note::

   Blackwell GPU 需要 CUDA 12.8 版 PyTorch。本项目固定的 PyTorch 2.7.0 也提供 CUDA 12.6 构建，\
   但该构建不支持 Blackwell。使用这类 GPU 时，首次安装及后续重新安装都使用以下命令，\
   通过 ``--torch-backend=cu128`` 选择 CUDA 12.8 构建：

   .. code-block:: bash

      uv pip install -e ".[robodojo-sim,sam3]" --extra-index-url https://pypi.nvidia.com \
        --override requirements/robodojo-override.txt \
        --torch-backend=cu128

   如果环境中已安装 ``torch==2.7.0+cu126``，uv 会认为它满足版本要求，继续保留原有安装。\
   此时需在上面的命令末尾再添加 ``--reinstall-package torch --reinstall-package torchvision``，\
   强制替换为 CUDA 12.8 构建。

当前固定的 IsaacLab 版本需要 editable 安装，才能找到扩展配置。克隆到一个可写目录，并在同一环境中执行：

.. code-block:: bash

   git clone --no-checkout https://github.com/yuechen0614/IsaacLab.git /path/to/IsaacLab
   git -C /path/to/IsaacLab checkout afca7b09d60d8beb9c1cb28b43066499940b969b
   uv pip install --no-deps \
     -e /path/to/IsaacLab/source/isaaclab \
     -e /path/to/IsaacLab/source/isaaclab_assets \
     -e /path/to/IsaacLab/source/isaaclab_tasks

安装后保留该源码目录，运行时仍会使用它。

下载仿真资源
------------

从 `RoboDojo 数据集 <https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo>`_ 下载机器人、物体、材质和布局文件，\
并设置资源目录：

.. code-block:: bash

   hf download RoboDojo-Benchmark/RoboDojo --repo-type dataset \
     --include 'Assets/**' --local-dir /path/to/robodojo
   export ROBODOJO_ASSETS_ROOT=/path/to/robodojo

下载后，目录中应有 ``Assets/Robots``、``Assets/Object``、``Assets/Material`` 和 ``Assets/Eval_Layout``。

运行以下命令，将 ``Assets/Robots`` 下的 ``*_tmp.yml`` 模板生成为同目录下的 ``*.yml`` 配置，并填入本机资源的绝对路径。\
仿真器使用的 x5 配置为 ``Assets/Robots/x5/curobo.yml``：

.. code-block:: bash

   python - <<'PY'
   import os
   import runpy
   from robodojo_runtime import source_root

   script = source_root() / "utils/update_embodiment_config_path.py"
   os.chdir(os.environ["ROBODOJO_ASSETS_ROOT"])
   runpy.run_path(str(script), run_name="__main__")
   PY
   test -s "$ROBODOJO_ASSETS_ROOT/Assets/Robots/x5/curobo.yml"

首次下载后需要执行一次，之后运行任务时无需重复执行。若移动了资源目录，配置中保存的路径就会失效，需要重新运行此命令。\
重新生成会覆盖已有的 ``*.yml`` 配置。

场景引用的部分 NVIDIA 素材默认在线加载。离线运行或相关资源持续加载失败时，参见文末的 :ref:`robodojo-local-nvidia-assets`。

VLA 配置
--------

通过 ``--vla-backend vpp2|awomo05`` 选择动作模型；省略时使用 ``vpp2``。可按下文配置 VPP2，或转到 :ref:`robodojo-awomo-setup`。Awomo 不需要 VPP2 权重，也不使用 ``VPP2_MODEL_PATH`` 或 ``--vla-model-path``。

VPP2 配置
~~~~~~~~~

准备官方 `VPP2 RoboDojo 实现 <https://github.com/roboterax/video-prediction-policy-2/tree/72c17acf022494ff63617be86079bb92e92f5bf8>`_ 和 `Haodong082399/VPP2 <https://huggingface.co/Haodong082399/VPP2>`_ 发布的配对 100k 权重。资源目录可以自行选择；以下以 ``/path/to/vpp2`` 为例，使用时替换为自己的目录：

.. code-block:: bash

   export VPP2_MODEL_PATH=/path/to/vpp2
   git clone --no-checkout https://github.com/roboterax/video-prediction-policy-2.git \
     "$VPP2_MODEL_PATH/policy"
   git -C "$VPP2_MODEL_PATH/policy" checkout 72c17acf022494ff63617be86079bb92e92f5bf8
   hf download Haodong082399/VPP2 --local-dir "$VPP2_MODEL_PATH" \
     --include 'checkpoints/joint2b_s100000/**' \
     --include 'checkpoints/Wan2.1-I2V-14B-480P/**'

资源根目录采用以下布局，子目录可以通过软链接复用已有资源：

.. code-block:: text

   vpp2/
   ├── policy/src/vpp2/policy.py
   └── checkpoints/
       ├── joint2b_s100000/
       │   ├── action.pt
       │   ├── video.pt
       │   └── dataset_stats.json
       └── Wan2.1-I2V-14B-480P/

按固定版本的 `推理环境说明 <https://github.com/roboterax/video-prediction-policy-2/blob/72c17acf022494ff63617be86079bb92e92f5bf8/docs/robodojo.md#policy-environment>`_ 安装模型依赖。这些依赖与 ``.[robodojo-sim]`` 分开安装。模型依赖与 RPent 环境兼容时，设置 ``VPP2_MODEL_PATH`` 即可自动启动本地服务，``--vla-model-path`` 可覆盖该路径。本地服务使用 RPent 当前的 Python 解释器，在单张 GPU 上加载完整模型。

若使用独立模型环境，在该环境中安装策略依赖和 RPent 核心包，然后启动服务：

.. code-block:: bash

   python -m robots.robodojo.vpp2_server --model-path /path/to/vpp2 \
     --host 127.0.0.1 --port 9201

在 RPent／仿真环境的任务命令中添加 ``--vla-endpoint socket://127.0.0.1:9201``。RPent 负责启动和关闭自己创建的模型进程；独立启动的服务由其启动者管理。远端部署见 :doc:`../guides/advanced_deployment`。

.. _robodojo-awomo-setup:

Awomo05 配置
~~~~~~~~~~~~

环境和资源准备以 `XPolicyLab 官方 Awomo05 指南 <https://github.com/XPolicyLab/XPolicyLab/blob/562c14cf08b9e6e411e7d76f192740afced77160/policy/Awomo05/README.md>`_ 为准。模型使用独立于 Isaac Sim 的环境。RPent 桥接服务（bridge）只需 RPent 和 XPolicyLab 的 WebSocket 依赖，可以运行在 RPent 仿真环境中，无需在该环境安装 Awomo 模型依赖。

以下流程由你启动模型，再由 RPent 启动桥接服务。将示例目录替换为本地绝对路径。先准备上游源码中的策略代码和机器人配置；RPent 仍使用前文安装的仿真器：

.. code-block:: bash

   git clone https://github.com/RoboDojo-Benchmark/RoboDojo.git /path/to/RoboDojo
   git -C /path/to/RoboDojo checkout 2184bf8844ea9d205382c4aefa3a694311418251
   git -C /path/to/RoboDojo submodule update --init XPolicyLab
   git -C /path/to/RoboDojo/XPolicyLab fetch origin 562c14cf08b9e6e411e7d76f192740afced77160
   git -C /path/to/RoboDojo/XPolicyLab checkout 562c14cf08b9e6e411e7d76f192740afced77160
   uv venv --seed --python 3.11 /path/to/awomo-env
   source /path/to/awomo-env/bin/activate
   cd /path/to/RoboDojo/XPolicyLab
   bash policy/Awomo05/install.sh

保留父级 RoboDojo 源码中的 ``env_cfg/arx_x5.yml`` 和 XPolicyLab 的机器人配置。将 `Awomo 权重与配套统计文件 <https://huggingface.co/Auwomo/Awomo-0.5-Robodojo>`_ 一起下载，并遵循仓库的访问与许可要求。官方下载脚本会校验 ``SHA256SUMS``，同时下载 `FLUX.2-klein-base-4B <https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B>`_、`FLUX.2-dev 自编码器 <https://huggingface.co/black-forest-labs/FLUX.2-dev>`_ 和 `Qwen3-VL-4B-Instruct <https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct>`_：

.. code-block:: bash

   # 在 Awomo 环境中，从 XPolicyLab 根目录运行。
   export AWOMO05_WEIGHTS=/path/to/awomo-weights
   hf auth login
   bash policy/Awomo05/download_weights.sh
   export AWOMO05_FLUX2_BASE="$AWOMO05_WEIGHTS/flux2-klein-base-4b/flux-2-klein-base-4b.safetensors"
   export AWOMO05_FLUX2_AE="$AWOMO05_WEIGHTS/flux2-dev/ae.safetensors"
   export AWOMO05_QWEN_VL="$AWOMO05_WEIGHTS/Qwen3-VL-4B-Instruct"

在同一终端中，通过 RPent 提供的入口启动带元数据接口的模型服务。将 XPolicyLab 目录及其父目录一起加入 ``PYTHONPATH``，供官方模型和服务端导入模块。示例为模型保留 GPU 0，并在 :ref:`robodojo-awomo-run` 中为仿真保留 GPU 1；请按实际空闲设备调整：

.. code-block:: bash

   export PYTHONPATH="/path/to/RoboDojo:/path/to/RoboDojo/XPolicyLab${PYTHONPATH:+:$PYTHONPATH}"
   CUDA_VISIBLE_DEVICES=0 python /path/to/RPent/robots/robodojo/awomo05_policy_server.py \
     --config_path /path/to/RoboDojo/XPolicyLab/policy/Awomo05/deploy.yml \
     --host 127.0.0.1 --port 9005 --overrides \
     policy_name=Awomo05 bench_name=RoboDojo task_name=general_pickup \
     env_cfg_type=arx_x5 action_type=joint seed=0 ckpt=model.pt \
     input_color_order=bgr replan_steps=8

等待模型终端依次出现 ``Awomo05 ready`` 和 ``websocket policy server listening``，分别表示模型初始化完成和开始监听。保持该终端运行。上述入口提供 ``get_model_meta``，供桥接服务校验兼容性。保留 BGR 配置、25 Hz 历史、原生关节布局和官方任务指令；RPent 不重排动作，也不额外变换图像。启动失败时，先按 :ref:`robodojo-awomo-troubleshooting` 排查，再启动 RPent。

SAM3 配置
---------

SAM3 用于物体分割。VPP2 默认启动该服务；Awomo 仅在通过 ``--vla-agent --agent-tools ... sam3`` 选择 SAM3 工具组时需要它。先在 `facebook/sam3 模型页面 <https://huggingface.co/facebook/sam3>`_ 申请访问权限，\
再下载 ``sam3.pt`` 并设置路径：

.. code-block:: bash

   hf download facebook/sam3 sam3.pt --local-dir /path/to/sam3
   export SAM3_CHECKPOINT_PATH=/path/to/sam3/sam3.pt

也可以从 `ModelScope <https://modelscope.cn/models/facebook/sam3>`_ 下载同一文件，\
再将 ``SAM3_CHECKPOINT_PATH`` 指向本地文件。

运行一个任务
------------

先按 :doc:`../guides/configure_planner` 选择并配置要使用的 planner 和模型。以下以 Codex 为例，\
在运行任务的同一终端检查连接：

.. code-block:: bash

   rpent-check-llm --planner codex --model <planner-model> --json

将 ``<planner-model>`` 替换为所选模型名，配置好 VPP2 和 SAM3 后，在 seed 组 0、layout 0 上运行瓶子入桶任务：

.. code-block:: bash

   # 使用当前环境中安装的 RLinf 集成。
   export RPENT_RLINF_ROOT="$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"

   export ROBODOJO_PLACEMENT_SETTLE_STEPS=1000  # 推荐：让物体充分稳定；不设置时默认等待 20 步
   export OMNI_KIT_ACCEPT_EULA=YES  # 推荐用于首次非交互启动，提前接受 Kit 许可；已接受时可省略
   # Isaac Sim 启动需数十秒，首次运行还会编译 shader
   rpent --robot robodojo --vla-backend vpp2 \
     --task put_bottles_into_dustbin --seed 0 --layout 0 \
     --planner codex --model <planner-model> --reasoning-effort medium \
     --max-turns 40 --planner-timeout-s 1500 \
     --output-dir /path/to/run-output

.. _robodojo-awomo-run:

使用 Awomo05 运行
~~~~~~~~~~~~~~~~~

完成 :ref:`robodojo-awomo-setup` 并确认模型已开始监听后，打开第二个终端，进入前文创建的 RPent 仿真环境。在该环境安装 XPolicyLab 通信包；这不会安装 Awomo 模型：

.. code-block:: bash

   cd /path/to/RPent
   source .venv-robodojo/bin/activate
   uv pip install -e /path/to/RoboDojo/XPolicyLab
   export ROBODOJO_ASSETS_ROOT=/path/to/robodojo
   export RPENT_RLINF_ROOT="$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
   export ROBODOJO_PLACEMENT_SETTLE_STEPS=1000
   export OMNI_KIT_ACCEPT_EULA=YES
   rpent-check-llm --planner codex --model <planner-model> --json
   CUDA_VISIBLE_DEVICES=1 rpent --robot robodojo --vla-backend awomo05 \
     --task general_pickup --seed 0 --layout 0 \
     --policy-server-url ws://127.0.0.1:9005 \
     --xpolicylab-root /path/to/RoboDojo/XPolicyLab \
     --policy-python /path/to/RPent/.venv-robodojo/bin/python \
     --planner codex --model <planner-model> --reasoning-effort medium \
     --max-turns 40 --planner-timeout-s 1500 \
     --output-dir /path/to/awomo-run-output

将 ``<planner-model>`` 替换为已配置的规划器模型名。该命令启动仿真环境和 CPU 桥接服务，连接已运行的 Awomo 模型，不需要 VPP2 或 SAM3 权重。RPent 会等待桥接服务就绪并校验元数据，再开始执行。若启动中断，检查输出目录中的 ``awomo05_bridge.log`` 和 ``robodojo_env_server.log``。规划器可调用 ``run_policy``、读取已保存状态并调用 ``finish``。需要混合控制时，添加 ``--vla-agent --agent-tools policy motion geometry``；工具与规划器限制见 :ref:`robodojo-awomo-collaboration`。

.. _robodojo-existing-rpc:

连接已有 RPC 服务（替代方式）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``--vla-endpoint`` 表示 RPent RPC 地址，``--policy-server-url`` 表示 Awomo 上游 WebSocket 地址，两者互斥。若希望独立管理桥接服务，在其环境中安装 RPent 和 XPolicyLab 通信包，设置与上文相同的 ``PYTHONPATH``，然后启动：

.. code-block:: bash

   python -m robots.robodojo.awomo05_bridge --task general_pickup \
     --policy-server-url ws://127.0.0.1:9005 --host 127.0.0.1 --port 9205

在任务命令中，将 ``--policy-server-url``、``--xpolicylab-root`` 和 ``--policy-python`` 替换为 ``--vla-endpoint http://127.0.0.1:9205``。兼容的外部服务也可使用 ``socket://host:port``；上面的桥接命令提供 HTTP 服务。RPC 协议必须与 ``--vla-backend`` 一致：VPP2 使用 VPP2 协议，Awomo 使用会话与元数据契约。

RPent 在任务结束时释放策略回合，并在运行环境关闭时停止自己启动的桥接服务。连接的模型与 RPC 服务仍由其管理者停止。启动失败时，RPent 关闭待用连接并停止自己已启动的组件。

每个上游模型只使用一个桥接服务，并仅通过该桥接服务执行回合。官方服务在连接间共享模型历史；会话所有权只能阻止同一桥接服务内的并发回合，无法协调多个桥接服务或直连模型的其他客户端。需要并行运行时，为每个桥接服务分别启动模型服务。

查看结果
--------

终端会显示服务启动信息、规划器输出和工具调用。运行结束后，查看 ``--output-dir`` 指定目录中的文件：

- ``result.json``：环境的任务成功判定及运行错误状态。
- ``states.json``：工具命令、执行结果及同帧观测记录。
- 各相机逐步保存的 RGB；启用几何工具时，还保存可用的米制深度、标定和世界坐标三维图。
- ``videos/``：头部、左腕和右腕相机的视频。
- ``robodojo_env_server.log``：仿真启动与错误日志。VPP2 本地启动的 SAM3 和模型服务分别写入 ``sam3_server.log``、``robodojo_vpp2_server.log``。自动启动的 Awomo 桥接服务写入 ``awomo05_bridge.log``；独立模型服务的日志在模型终端中。Awomo 只有启动 SAM3 时才会产生对应日志。
- ``run.log`` 和 ``transcript_*.json``：主进程日志、规划器对话和错误。

先检查 ``environment_result_available=true`` 和 ``agent_error_present=false``，再看 ``success``。该字段是 RoboDojo 的原生判定；工具调用完成、回合结束或规划器调用 ``finish`` 均不代表任务成功。Awomo 的 ``finish.agent_assessment`` 单独记录规划器自述。

``states.json`` 保存完整 Awomo 执行结果，包括 ``executed_steps``、``pending_actions``、``stop_reason``，以及带有同次推理元信息的 ``inferences``。Planner 返回使用 ``inference_count`` 摘要，不重复携带完整推理列表。``step``／``step_idx`` 选择已保存观测记录，``executed_steps`` 是单次调用确认的动作数，``observation_step`` 相对于当前模型会话计数，三者不能混用。

未指定输出目录时，结果保存在 ``logs/<timestamp>_robodojo_<task>_s<seed>_l<layout>/``。并发运行应使用不同目录。\
实时查看方式见 :doc:`../guides/dashboard`。

常用参数
--------

CLI 和 Dashboard 使用相同默认值：VPP2 在本地启动仿真、SAM3 和 VPP2；Awomo 启动仿真并连接模型，仅在工具组选择 SAM3 时添加该组件。``--sim-python`` 指定仿真解释器；``--policy-python`` 指定 Awomo 桥接服务的解释器，不负责启动模型进程。

使用 ``--task`` 选择任务。固定场景通过 ``--seed`` 选择官方布局组，通过 ``--layout`` 选择组内布局，两者默认均为 ``0``。例如，``--task stack_blocks --seed 0 --layout 2`` 使用第 0 组的第 2 个布局。需要随机生成场景时，添加 ``--random``，此时 ``--layout`` 不生效。

- ``--cuda-device``：为本地启动的服务选择 GPU；省略时沿用 ``CUDA_VISIBLE_DEVICES``。
- ``--vla-backend``：选择 ``vpp2`` （默认）或 ``awomo05``。
- ``--vla-model-path``：仅用于 VPP2，覆盖 ``VPP2_MODEL_PATH``，指向包含 ``policy/`` 和 ``checkpoints/`` 的资源根目录。
- ``--env-endpoint``、``--sam3-endpoint``、``--vla-endpoint``：连接已有 RPent RPC 服务；VLA 协议必须匹配所选后端。
- ``--policy-server-url`` 和 ``--xpolicylab-root``：按 :ref:`robodojo-awomo-run` 配置本地启动的 Awomo 桥接服务。
- ``--vla-agent``：启用 Awomo 混合控制；``--agent-tools`` 选择 ``policy``、``motion``、``geometry`` 或 ``sam3`` 工具组。
- ``--source-root``：使用本地 RoboDojo 源码替代包内源码。

子服务继承当前终端的环境变量；CLI 不会自动读取 ``config/runtime.env``。\
完整参数可运行 ``rpent --robot robodojo --help`` 查看。

工具
----

VPP2 模式下可使用 ``vpp2_act`` 及运动和感知工具。Awomo 模式下使用 ``run_policy(max_steps)`` 执行官方指令；运动和感知工具需要启用对应的 ``--vla-agent`` 工具组。规划器不能重置回合；探索模式和 Flash 重放不可用。

``completed`` 表示本次工具请求是否完成，不代表任务成功。提前停止时检查 ``stop_reason``；``executed_actions`` 与 ``executed_steps`` 均记录已确认执行的动作数。``execution_uncertain=true`` 表示存在可能已执行但无法确认的动作，这些动作不计入确认数。系统不会自动重试，调用方也不应直接重试，以免重复执行。

与有状态策略协作时，动作原语接管会使缓存的剩余策略动作失效，后续策略根据新观测重新推理。同步失效或策略历史无效时，读取观测或恢复执行不能修复状态，需要成功重置并恢复同步；无法重置时需开始新回合。

``vpp2_act(instruction, chunks=1)`` 执行指定子任务。每个动作块预测 32 个双臂 EE16 绝对目标（每臂 xyz、wxyz 和夹爪），通过原生 EE 控制执行前 24 步，再重新推理。VPP2 使用初始 RGB 帧和间隔为 25 个动作的 8 个历史采样。策略动作和动作原语都会更新历史，切换子任务时保留历史；读取图像和 ``render`` 不推进动作历史。

``view_env_state`` 读取保存的图像和状态，``render`` 刷新观测。``sample_world_xyz(view, pixels, step=None, neighborhood=1)`` 查询 [行,列] 像素，默认取 3×3 邻域。``query_world_map`` 查询半开区间 [r0,c0,r1,c1]，默认最多返回 256 个点。view 使用 ``head``、``left_wrist`` 或 ``right_wrist``，必须与选点图像的视角及步号一致。查询读取保存的同帧深度，不暗中刷新；无效深度被排除，返回的是可见表面坐标，不是物体中心。

``segment`` 和 ``back_project`` 也接受 ``step``。分割返回质心 ``centroid_rc`` 和掩码内像素 ``point_rc``；凹形掩码的质心可能在物体外，反投影应优先使用 ``point_rc`` 及返回的 ``step_idx``。这两个工具保留相机名 ``cam_head``、``cam_left_wrist``、``cam_right_wrist``。

调用 ``segment(text_prompt=...)`` 时，描述可见的颜色、形状和位置。工具返回红色掩码叠加图；使用定位点前先检查叠加图，高分不代表选中了正确物体。分割失败时可修改描述，或在同一历史帧中手动选取可见像素查询。叠加图和几何结果保存在对应帧中。使用 ``back_project`` 获取选定掩码内像素的世界坐标。

运动 Primitive 使用原生 cuRobo 位姿规划并执行关节目标。规划器的障碍物模型包含简化桌面，未完整表示场景物体和另一只机械臂；选择路点时应为它们留出空间。

``move_to(arm, xyz, quat=None, gripper=None, substeps=25)`` 使用世界坐标、米及 wxyz 四元数；省略姿态或夹爪参数则保持对应状态／目标。``substeps`` 限制路径抽样数，0 保留完整路径，1 仅取最后路点，不代表速度或物理步数。``rotate_wrist`` 绕世界 Z 轴相对旋转，单位为度。``final_dist_m`` 是实测位置误差，不是到达保证。

``move_delta`` 为 ``arm="left"`` 或 ``"right"`` 指定的机械臂规划末端相对目标。``dxyz`` 为三维位移，单位为米，默认 ``[0, 0, 0]``；``angle_deg`` 为绕三维轴向量 ``axis`` 的转角，单位为度，二者默认分别为 ``0`` 和 ``[0, 0, 1]``。向量分量与转角须为有限数值；非零转角要求非零轴向量，无需预先归一化。旋转以动作开始时的末端原点为中心。``frame="world"`` （默认）使用世界坐标系表示位移和轴方向，``frame="tool"`` 使用动作开始时的工具坐标系。

``substeps=25`` 须为非负整数，路径采样含义与 ``move_to`` 相同。工具不保证直线运动或多圈旋转轨迹。

``gripper=None`` 保持已有夹爪控制目标；显式目标的范围和开度含义与 ``set_gripper`` 相同。位移和转角均为零且未指定夹爪目标时，不执行动作；显式指定夹爪目标时，即使位移和转角为零也会规划并执行。

``set_gripper(arm, val, steps=10)`` 使用 0=闭、1=开，``release`` 默认开到 1。中间值表示归一化开度目标，而非物理宽度。反馈读取实测关节状态，夹爪运动期间保持其他关节目标不变。

``hold(steps)`` 要求 ``steps`` 为正整数，保持双臂和夹爪已有控制目标，推进相应数量的原生动作步。仿真继续推进，不保证实测位姿或物体静止。回合终止、动作预算耗尽或取消都可能使其提前停止。

.. _awomo05-与-agent-协作:
.. _robodojo-awomo-collaboration:

Awomo 执行模式
--------------

不加 ``--vla-agent`` 时，规划器通过 ``view_env_state`` 观察，选择有界 ``run_policy(max_steps)`` 调用，最后调用 ``finish``，不能插入运动原语。

启用 ``--vla-agent`` 后，规划器根据当前观测，自主选择已启用的策略、运动和感知工具，决定调用顺序和预算。两种模式都向 Awomo 传入官方完整任务指令，不改写模型子任务。

混合模式始终提供 ``view_env_state``、``render``、``finish`` 和下文的记忆读取工具；``--agent-tools`` 默认选择 ``policy motion geometry``：

.. list-table:: 工具组与观测要求
   :header-rows: 1

   * - 工具组
     - 工具
     - 所需数据或服务
   * - ``policy``
     - ``run_policy``
     - Awomo 服务、官方完整指令、原生关节状态及三路 RGB
   * - ``motion``
     - ``move_to``、``move_delta``、``rotate_wrist``、``set_gripper``、``release``、``hold``
     - 原生运动状态与规划能力，不要求深度
   * - ``geometry``
     - ``back_project``、``sample_world_xyz``、``query_world_map``
     - 查询视图已保存的同帧深度、有效标定，深度尺寸须与 RGB 一致
   * - ``sam3``
     - ``segment``
     - SAM3 服务和选定记录中的 RGB，不要求深度

即使未启用 ``policy`` 工具组，Awomo 会话仍需要模型服务同步历史。头部和双腕 RGB 始终必需；只有启用的几何查询需要对应深度与标定。缺少几何数据时查询报错，已提供但格式无效的数据会导致捕获失败。同一记录的 RGB、深度和标定来自同次观测。

``run_policy(max_steps)`` 限制单次调用的原生动作数，不是动作块数。预算截断后可续接未消费动作；运动原语实际执行动作后才会清空旧动作块，并保留模型历史。拒绝或零动作调用不算接管，只读查询不推进动作或模型历史。

所有控制共享 ``--max-episode-steps``。规划过程由 ``--max-turns`` 和 ``--planner-timeout-s`` 按所选 :doc:`规划器 <../guides/configure_planner>` 的语义约束，规划轮次不等于工具调用次数。中断与恢复见 :ref:`robodojo-awomo-troubleshooting`。

任务记忆
--------

Awomo 混合模式沿用现有 :doc:`记忆配置 <../guides/memory>` 读取已有经验；``list_dir`` 和 ``read_text_file`` 仅允许读取配置的经验目录，不能写入记忆。任务源码、评估器和隐藏资产不属于任务访问接口。当前指令与新观测优先于历史经验；探索和 Flash 回放仍不支持。Codex 工具来源控制见 :ref:`Planner 指南 <planner-codex>`。

.. _robodojo-awomo-troubleshooting:

Awomo 启动与恢复排查
--------------------

权重和推理错误查看模型终端；WebSocket 或元数据错误查看 ``awomo05_bridge.log``；场景启动错误查看 ``robodojo_env_server.log``。重新运行前，检查对应条件：

- 缺少权重或模块导入失败：对照官方安装指南，检查 ``AWOMO05_WEIGHTS``、三个基础模型路径、``dataset_stats.json`` 和 ``PYTHONPATH`` 中的两个目录。模型应在独立环境中运行。
- 缺少 ``get_model_meta`` 或元数据不兼容：使用 RPent 的模型启动入口，保持 ``arx_x5``、绝对 ``joint`` 动作、BGR 输入和 25 Hz 历史。普通 VPP2 或旧策略端点不是 Awomo 端点。
- 连接被拒绝：等待模型的监听日志，确认桥接服务能访问指定主机和端口，并区分 WebSocket 与 RPent RPC 地址。端口监听本身不能证明模型兼容或推理成功。
- 重置时报 ``All scene Unstable Error``：检查前文的 ``ROBODOJO_PLACEMENT_SETTLE_STEPS=1000`` 设置和场景资源。
- 已确认的取消会暂停控制，可通过现有会话控制恢复；缓存由 RPent 处理，不重置模型历史。
- 推理、执行或观测交付结果不确定时，不要重试或直接恢复。反馈仍保留已确认进度，但后续控制会被阻止。检查已保存记录和错误后开始新回合；规划器没有 reset 工具。显式重置环境会重新建立模型历史。


.. _robodojo-awomo-verification:

有界验证
~~~~~~~~

使用已配置 XPolicyLab 通信依赖的 RPent 仿真环境，并保持模型服务运行。测试使用当前解释器启动桥接服务。通过 ``CUDA_VISIBLE_DEVICES`` 保留仿真 GPU，并沿用首次运行时的资产、RLinf 和稳定步数环境变量：

.. code-block:: bash

   uv pip install -e ".[test]"
   RPENT_RUN_AWOMO05_INTEGRATION=1 \
   RPENT_AWOMO05_POLICY_SERVER_URL=ws://127.0.0.1:9005 \
   RPENT_XPOLICYLAB_ROOT=/path/to/RoboDojo/XPolicyLab \
   RPENT_E2E_OUTPUT_DIR=/path/to/new-output \
     pytest tests/e2e_tests/robodojo/test_awomo_bounded.py -v -s

这些检查使用真实推理和脚本规划器验证集成链路，不评估在线 LLM 的规划质量或任务成功率。原生任务成功结果应单独报告。详细覆盖范围与跳过条件见 ``tests/README.md``。

.. _robodojo-local-nvidia-assets:

NVIDIA 场景素材的本地准备（可选）
------------------------------------------------------------

以下两类用户需要预先下载 NVIDIA 场景素材：

- 需要在无法联网的机器上运行仿真的用户。
- 运行时无法稳定访问 NVIDIA 资源地址，日志中相关文件反复下载失败的用户。

RoboDojo 场景还会引用 NVIDIA 提供的三维模型、材质和贴图。这些文件与前面下载的 RoboDojo 资源分开提供，默认在加载场景时通过网络获取。\
场景能正常加载时，可跳过本节，无需下载本地资产包，也不用设置 ``ROBODOJO_USD_ASSET_PREFIX``。

以下步骤用于准备 ``Assets/Isaac/5.0`` 路径对应的本地资源。完整资产包的三个分卷合计约 150 GB，并非单个 RoboDojo 场景所需的下载量；\
合并和解压还需额外空间。离线机器可先在能联网的机器上下载，再将分卷复制到 ``/path/to/nvidia-assets``。

.. code-block:: bash

   mkdir -p /path/to/nvidia-assets

   # 下载三个分卷，已有下载可断点续传
   wget -c -P /path/to/nvidia-assets \
     https://download.isaacsim.omniverse.nvidia.com/isaac-sim-assets-complete-5.0.0.zip.001 \
     https://download.isaacsim.omniverse.nvidia.com/isaac-sim-assets-complete-5.0.0.zip.002 \
     https://download.isaacsim.omniverse.nvidia.com/isaac-sim-assets-complete-5.0.0.zip.003

   # 合并并解压，保留 Assets/Isaac/5.0 目录结构
   cat /path/to/nvidia-assets/isaac-sim-assets-complete-5.0.0.zip.001 \
       /path/to/nvidia-assets/isaac-sim-assets-complete-5.0.0.zip.002 \
       /path/to/nvidia-assets/isaac-sim-assets-complete-5.0.0.zip.003 \
     > /path/to/nvidia-assets/isaac-sim-assets-complete-5.0.0.zip
   unzip /path/to/nvidia-assets/isaac-sim-assets-complete-5.0.0.zip -d /path/to/nvidia-assets

   test -d /path/to/nvidia-assets/Assets/Isaac/5.0/Isaac
   test -d /path/to/nvidia-assets/Assets/Isaac/5.0/NVIDIA

在运行 RPent 的终端中设置以下变量，再运行任务：

.. code-block:: bash

   export ROBODOJO_USD_ASSET_PREFIX=/path/to/nvidia-assets

这会将 RoboDojo 匹配的 5.0 在线资源地址改为本地路径。它并不代表所有 Isaac Sim 资源均可离线使用；\
若仍有其他在线资源加载失败（例如 ``Assets/Isaac/5.1`` 路径），需另外准备对应版本的资产并配置本地加载，\
见 `NVIDIA 本地资产说明 <https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_faq.html#isaac-sim-setup-assets-content-pack>`_。
