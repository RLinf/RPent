机器人 Episode
==============

用同一套 episode 在 LIBERO、RoboTwin、RoboCasa 或 RoboDojo 中运行模型。模型可调用 ``view_env``、``segment``、``back_project`` 和 ``execute_actions``。仿真器测量状态、执行动作并判定任务成功。只有 ``view_env`` 向模型返回图像。

安装仿真环境
------------

准备配套版本的 RPent 和 RLinf。将两个仓库放在同一父目录。在仿真主机上，从 RPent 根目录执行对应的安装命令。脚本创建独立虚拟环境，并使用同一份 RLinf checkout 的仿真适配器。

.. code-block:: bash

   export PROJECT_DIR="$(dirname "$PWD")"
   export RLINF_ROOT="$PROJECT_DIR/RLinf"
   bash tools/rl/install_simulator.sh libero   "$PROJECT_DIR/envs/libero"   "$RLINF_ROOT"
   bash tools/rl/install_simulator.sh robotwin "$PROJECT_DIR/envs/robotwin" "$RLINF_ROOT"
   bash tools/rl/install_simulator.sh robocasa "$PROJECT_DIR/envs/robocasa" "$RLINF_ROOT"
   bash tools/rl/install_simulator.sh robodojo "$PROJECT_DIR/envs/robodojo" "$RLINF_ROOT"

.. list-table::
   :header-rows: 1

   * - 仿真器
     - 主机上的依赖和资产
   * - LIBERO
     - Python 3.12、EGL 和 MuJoCo。使用下方的 LIBERO-PRO 资产命令。
   * - RoboTwin
     - Python 3.11、NVIDIA Vulkan 和与 Torch 匹配的 CUDA 工具链，用于构建 cuRobo。使用下方的 RoboTwin 资产命令。
   * - RoboCasa
     - Python 3.11 和 EGL。在环境中运行 ``robocasa-download-assets --assets-path /work/data/robocasa --no-macros -y``。见 :doc:`../simulators/robocasa`。
   * - RoboDojo
     - Python 3.11、Isaac Sim 5.1 和 RTX GPU。将 RoboDojo 资产和评测布局放在 ``ROBODOJO_ASSETS_ROOT``。见 :doc:`../simulators/robodojo`。

下载资产和 SAM3
---------------

在运行仿真的主机上执行对应的下载命令。LIBERO-PRO 安装包已提供 BDDL 和原生初态。下载器从 `RLinf/LIBERO-PRO-assets <https://huggingface.co/datasets/RLinf/LIBERO-PRO-assets>`_ 准备场景资产，LIBERO-PRO 在 ``$PROJECT_DIR/data/libero-config`` 下生成 ``config.yaml``。

.. code-block:: bash

   mkdir -p "$PROJECT_DIR/data/libero-config"
   LIBERO_CONFIG_PATH="$PROJECT_DIR/data/libero-config" \
     "$PROJECT_DIR/envs/libero/bin/liberopro-download-assets" --skip-existing

RoboTwin 安装包附带的下载器将资产快照保存到 ``$PROJECT_DIR/data/robotwin``。

.. code-block:: bash

   "$PROJECT_DIR/envs/robotwin/bin/robotwin-download-assets" \
     --output "$PROJECT_DIR/data/robotwin"

先在 `SAM3 权重页面 <https://huggingface.co/facebook/sam3>`_ 申请访问权限。在 SAM3 主机上，用 uv 运行 Hugging Face CLI 下载 ``sam3.pt``。

.. code-block:: bash

   uv venv --python 3.12 "$PROJECT_DIR/envs/sam3"
   export SAM3_PYTHON="$PROJECT_DIR/envs/sam3/bin/python"
   uv pip install --python "$SAM3_PYTHON" -e '.[sam3]'
   uvx --from huggingface_hub hf auth login
   uvx --from huggingface_hub hf download facebook/sam3 sam3.pt --local-dir "$PROJECT_DIR/models/sam3"

单独采集轨迹时，在分配的 GPU 上启动 SAM3。训练时，下方的节点启动命令会按 YAML 启动 SAM3。

.. code-block:: bash

   export SAM3_CHECKPOINT_PATH="$PROJECT_DIR/models/sam3/sam3.pt"
   "$SAM3_PYTHON" -m rpent.robots.components.sam3_server \
     --host 127.0.0.1 --port 28424 --cuda-device 1

运行一条轨迹
------------

模型服务可以部署到独立 GPU。从 RPent 根目录启动本地 SGLang 模型服务，再在仿真环境中运行 LIBERO。跨主机时，先为回环监听的实例设置受限端口转发，再填写仿真节点可达的地址。

.. code-block:: bash

   export PROJECT_DIR="$(dirname "$PWD")"
   MODEL_PATH="$PROJECT_DIR/models/Qwen3.8-27B" SERVE_GPUS=2,3 \
     bash tools/rl/serve.sh

另开终端，进入 RPent 根目录，激活 LIBERO 环境并设置资产路径。

.. code-block:: bash

   export PROJECT_DIR="$(dirname "$PWD")"
   source "$PROJECT_DIR/envs/libero/bin/activate"
   export LIBERO_TYPE=pro
   export LIBERO_CONFIG_PATH="$PROJECT_DIR/data/libero-config"
   export RPENT_MODEL=openai-chat:qwen3.8-27b
   export RPENT_BASE_URL=http://127.0.0.1:30080/v1
   export RPENT_SAM3_ENDPOINT=http://127.0.0.1:28424
   export RPENT_OUTPUT_DIR="$PROJECT_DIR/runs/libero-first"
   bash tools/rl/run.sh --simulator libero --suite libero_spatial --task 0 --seed 0

RoboTwin、RoboCasa 和 RoboDojo 使用 ``--simulator`` 和 ``--task-name``。RoboDojo 可用 ``--layout`` 指定评测布局。输出目录保存图像、实测状态、工具回执和仿真器判定的任务结果。批量采集读取 JSON 任务清单，逐条封存并支持续跑。例如将以下内容保存到 ``$PROJECT_DIR/data/jobs.json``。

.. code-block:: json

   [{"environment":"libero","suite":"libero_spatial","task":0,"seed":0}]

.. code-block:: bash

   python tools/rl/run_batch.py run "$PROJECT_DIR/runs/libero-batch" \
     --jobs "$PROJECT_DIR/data/jobs.json" --gpus 0 --per-gpu 1

使用相同清单、配置和输出目录重跑命令即可跳过已封存的轨迹。

提供训练工具服务
----------------

各节点使用同一份 YAML。复制 ``tools/rl/deployment-local.yaml`` 或 ``tools/rl/deployment-384.yaml``，将示例中的 ``/work`` 换成项目目录，再填写节点可达地址。每个节点从 RPent 根目录执行自己的角色。

节点命令在前台运行。单机示例使用 SAM3 环境，只运行仿真的节点可使用一个已安装的仿真环境。

.. code-block:: bash

   "$PROJECT_DIR/envs/sam3/bin/python" -m robots.rl.services \
     --config "$PROJECT_DIR/deployment-local.yaml" --node local

单机示例的 ``local`` 使用 GPU 0 运行 SAM3，GPU 1 提供 LIBERO 和 RoboTwin 各 4 个槽位。384 示例的角色为 ``sim``、``model0`` 至 ``model3``、``twin``。每台各跑一条同样的命令，换成对应节点名。``model0`` 提供 LIBERO、RoboTwin 两个路由地址，填入 RLinf 的现场配置。

``backends.<simulator>.gpu_slots`` 描述该节点的实际 GPU 槽位，``endpoint`` 是路由节点访问该后端的地址。SAM3 的 ``replicas`` 描述每卡进程数，``max_pending`` 控制感知排队容量。跨网段时使用受限转发，把转发地址填入对应 endpoint。进程日志和仿真轨迹存入该节点配置的 ``output_dir``，每次启动使用新的服务目录。停止节点命令会回收它拥有的服务及环境进程。

训练主机与仿真端使用相同的 RPent 提交。模型和推理进程由 RLinf 管理。从配套 RLinf 仓库的 ``docs/source-zh/rst_source/examples/agentic/rpent_episode_sync.rst`` 查看任务清单生成和训练命令。

坐标与动作
----------

模型接口用厘米表示位置和长度，用厘米每秒表示线速度。姿态为固定轴 XYZ 的 roll、pitch、yaw，单位为弧度。单臂取实测安装座为原点，双臂取两个安装座的中点。X 向前，Y 向机器人左侧，Z 向上。感知工具在该坐标系内回报可见表面的几何，原始深度和仿真点云仍用米。

每批最多三个顺序动作，每项控制一个部件。RoboCasa 的手臂目标相对于该项开始时实测的坐标系，包括前一项刚移动底盘或升降的情况。观察和感知回执记录拍摄及结果所用的坐标版本，动作输入无需填写坐标系版本。
