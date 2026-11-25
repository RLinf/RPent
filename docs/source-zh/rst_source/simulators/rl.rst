机器人 Episode
==============

用同一套 episode 在 LIBERO、RoboTwin、RoboCasa 或 RoboDojo 中运行模型。模型可调用 ``view_env``、``segment``、``back_project`` 和 ``execute_actions``。仿真器测量状态、执行动作并判定任务成功。只有 ``view_env`` 向模型返回图像。

安装仿真环境
------------

准备配套版本的 RPent 和 RLinf。在运行仿真的主机上执行对应的安装命令。安装脚本创建独立虚拟环境，并引用该 RLinf checkout 中的仿真适配器。

.. code-block:: bash

   export RLINF_ROOT=/work/RLinf
   bash tools/rl/install_simulator.sh libero   .venv-libero   "$RLINF_ROOT"
   bash tools/rl/install_simulator.sh robotwin .venv-robotwin "$RLINF_ROOT"
   bash tools/rl/install_simulator.sh robocasa .venv-robocasa "$RLINF_ROOT"
   bash tools/rl/install_simulator.sh robodojo .venv-robodojo "$RLINF_ROOT"

.. list-table::
   :header-rows: 1

   * - 仿真器
     - 主机上的依赖和资产
   * - LIBERO
     - Python 3.12、EGL 和 MuJoCo。设置 ``LIBERO_TYPE=pro`` 与 ``LIBERO_CONFIG_PATH``，配置中给出资产、BDDL 和原生初态文件的位置。见 :doc:`libero`。
   * - RoboTwin
     - Python 3.11、NVIDIA Vulkan 和与 Torch 匹配的 CUDA 工具链，用于构建 cuRobo。在环境中运行 ``robotwin-download-assets``，将快照根目录设为 ``ROBOTWIN_ASSETS_PATH``。见 :doc:`robotwin`。
   * - RoboCasa
     - Python 3.11 和 EGL。在环境中运行 ``robocasa-download-assets --assets-path /work/data/robocasa --no-macros -y``。见 :doc:`robocasa`。
   * - RoboDojo
     - Python 3.11、Isaac Sim 5.1 和 RTX GPU。将 RoboDojo 资产和评测布局放在 ``ROBODOJO_ASSETS_ROOT``。见 :doc:`robodojo`。

准备兼容 OpenAI API 的模型服务和 SAM3 服务。在负责 SAM3 的 Python 环境中安装 ``sam3`` extra，按 :doc:`libero` 中的 SAM3 配置下载权重，并设置 ``SAM3_CHECKPOINT_PATH``。用分配给 SAM3 的 GPU 启动服务。

.. code-block:: bash

   uv venv --python 3.12 /work/envs/sam3
   export SAM3_PYTHON=/work/envs/sam3/bin/python
   uv pip install --python "$SAM3_PYTHON" -e '.[sam3]'
   export SAM3_CHECKPOINT_PATH=/work/models/sam3/sam3.pt
   "$SAM3_PYTHON" -m rpent.robots.components.sam3_server \
     --host 127.0.0.1 --port 28424 --cuda-device 1

运行一条轨迹
------------

模型服务可以部署到独立 GPU。以下示例启动一份本地 SGLang 模型服务，然后在仿真环境中运行 LIBERO。跨主机时，先为回环监听的实例设置受限端口转发，再填写仿真节点可达的地址。

.. code-block:: bash

   MODEL_PATH=/work/models/Qwen3.8-27B SERVE_GPUS=2,3 \
     bash tools/rl/serve.sh

另开终端，激活 LIBERO 环境并设置资产路径。

.. code-block:: bash

   source .venv-libero/bin/activate
   export LIBERO_TYPE=pro
   export LIBERO_CONFIG_PATH=/work/data/libero-config
   export RPENT_MODEL=openai-chat:qwen3.8-27b
   export RPENT_BASE_URL=http://127.0.0.1:30080/v1
   export RPENT_SAM3_ENDPOINT=http://127.0.0.1:28424
   export RPENT_OUTPUT_DIR=/work/runs/libero-first
   bash tools/rl/run.sh --simulator libero --suite libero_spatial --task 0 --seed 0

RoboTwin、RoboCasa 和 RoboDojo 使用 ``--simulator`` 和 ``--task-name``。RoboDojo 可用 ``--layout`` 指定评测布局。输出目录保存图像、实测状态、工具回执和仿真器判定的任务结果。批量采集读取 JSON 任务清单，逐条封存并支持续跑。例如将以下内容保存到 ``/work/data/jobs.json``。

.. code-block:: json

   [{"environment":"libero","suite":"libero_spatial","task":0,"seed":0}]

.. code-block:: bash

   python tools/rl/run_batch.py run /work/runs/libero-batch \
     --jobs /work/data/jobs.json --gpus 0 --per-gpu 1

使用相同清单、配置和输出目录重跑命令即可跳过已封存的轨迹。

提供训练工具服务
----------------

各节点使用同一份 YAML。复制 ``tools/rl/deployment-local.yaml`` 或 ``tools/rl/deployment-384.yaml``，填写模型、仿真解释器、资产和节点可达地址。每个节点从 RPent 根目录执行自己的角色。

用已安装 RPent 和 PyYAML 的 Python 环境运行节点命令。该命令在前台运行。

.. code-block:: bash

   python -m robots.rl.services --config /work/deployment.yaml --node local

单机示例的 ``local`` 使用 GPU 0 运行 SAM3，GPU 1 提供 LIBERO 和 RoboTwin 各 4 个槽位。384 示例的角色为 ``sim``、``model0`` 至 ``model3``、``twin``。每台各跑一条同样的命令，换成对应节点名。``model0`` 提供 LIBERO、RoboTwin 两个路由地址，填入 RLinf 的现场配置。

``backends.<simulator>.gpu_slots`` 描述该节点的实际 GPU 槽位，``endpoint`` 是路由节点访问该后端的地址。SAM3 的 ``replicas`` 描述每卡进程数，``max_pending`` 控制感知排队容量。跨网段时使用受限转发，把转发地址填入对应 endpoint。进程日志和仿真轨迹存入该节点配置的 ``output_dir``，每次启动使用新的服务目录。停止节点命令会回收它拥有的服务及环境进程。

训练主机与仿真端使用相同 RPent 提交版本。模型和推理进程由 RLinf 管理。参照 `RLinf 混合训练指南 <https://github.com/zhexuanxu/RLinf/blob/codex/rpent-mixed-grpo/docs/source-zh/rst_source/examples/agentic/rpent_episode_sync.rst>`_ 启动 Ray 和训练。

坐标与动作
----------

模型接口用厘米表示位置、长度和线速度。姿态为固定轴 XYZ 的 roll、pitch、yaw，单位为弧度。单臂取实测安装座为原点，双臂取两个安装座的中点。X 向前，Y 向机器人左侧，Z 向上。感知工具在该坐标系内回报可见表面的几何，原始深度和仿真点云仍用米。

每批最多三个顺序动作，每项控制一个部件。RoboCasa 的手臂目标相对于该项开始时实测的坐标系，包括前一项刚移动底盘或升降的情况。观察和感知回执记录拍摄及结果所用的坐标版本，动作输入只写目标。
