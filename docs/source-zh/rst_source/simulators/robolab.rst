RoboLab
=======

通过 RPent 的观测和动作工具，调用 RLinf 的 RoboLab 环境运行 Isaac Lab 任务，每次运行连接一个 DROID 关节位置控制环境。任务成功由 RoboLab 原生评估器判定。

概览
----

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 动作模型

      DROID 关节位置控制的 Pi0.5，或不使用模型直接执行原生动作。

   .. grid-item-card:: 规划器

      ``api``、``claude_code``、``codex``。

   .. grid-item-card:: 任务

      RoboLab 基准任务类，例如 ``BananaInBowlTask``。

   .. grid-item-card:: 硬件

      Linux、NVIDIA GPU、Python 3.11、Isaac Sim 5.0 / Isaac Lab 2.2.0。

任务
~~~~

.. list-table::
   :header-rows: 1

   * - 选项
     - 作用
   * - ``--task BananaInBowlTask``
     - 从已安装的 RoboLab 中选择一个基准任务，创建一个仿真环境。
   * - ``--seed 0``
     - 将种子传入原生环境配置。

任务类由已安装的 RoboLab 版本提供。RPent 不包含其任务定义或场景资产。

观测与动作
~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - 项目
     - 说明
   * - 观测
     - 外部与腕部 RGB 图像、八个 DROID 关节/夹爪状态值和任务指令。策略使用 224 像素图像；保存给规划器的图像保持相机原始分辨率。
   * - 动作
     - ``pi05_act`` 执行一次 Pi0.5 预测中的 1–15 个动作。``execute_action`` 执行一个八维 DROID 关节位置动作。
   * - 奖励与成功判定
     - RoboLab 的原生环境结果写入 ``view_env_state`` 的 ``state.success``；规划器的 ``finish`` 状态与其分开。
   * - 任务指令
     - 当前环境的任务指令，也会传给 Pi0.5。

安装与资源准备
--------------

准备 Linux、兼容的 NVIDIA 驱动、Python 3.11、``uv``、Git、Git LFS 和 ``ffmpeg``。Isaac Sim 还需要 EGL、GLU、OpenGL 和 Vulkan 运行库。在 Ubuntu 22.04 上，可安装缺少的系统包：

.. code-block:: bash

   sudo apt-get update
   sudo apt-get install git-lfs ffmpeg libegl1 libglu1-mesa libopengl0 libvulkan1

其他发行版请使用相应的系统包管理器；没有安装权限时，请管理员提供这些运行库。在 RPent 仓库根目录，按照 :doc:`../get_started/quickstart` 的方式创建策略环境并安装可选依赖：

.. code-block:: bash

   uv venv --python 3.11 .venv-robolab-policy
   source .venv-robolab-policy/bin/activate
   uv pip install -e ".[robolab]"

按照 `RLinf RoboLab 指南 <https://rlinf.readthedocs.io/zh-cn/latest/rst_source/evaluations/guides/robolab.html>`_，使用 RLinf ``main`` 分支的安装器创建仿真环境。在 RPent 仓库根目录运行以下命令，并保留 RLinf 源码目录。``--install-rlinf`` 会将 RLinf 安装到仿真环境，使 RPent 在 RLinf 仓库之外也能导入 ``RoboLabEnv``。

.. code-block:: bash

   export RPENT_ROOT="$PWD"
   export RLINF_SOURCE="$HOME/.cache/rpent/RLinf"
   git lfs install
   mkdir -p "$(dirname "$RLINF_SOURCE")"
   git clone --branch main https://github.com/RLinf/RLinf.git "$RLINF_SOURCE"
   (
     cd "$RLINF_SOURCE"
     bash requirements/install.sh embodied --model openpi --env robolab \
       --venv "$RPENT_ROOT/.venv-robolab-sim" --install-rlinf
   )
   git -C "$RPENT_ROOT/.venv-robolab-sim/robolab" lfs pull

安装器负责选择 RoboLab 版本，安装 Isaac Sim 5.0、Isaac Lab 2.2.0 和 OpenPI，并处理 Isaac/Kit 的包文件。默认情况下，RoboLab 及其 USD 资产位于 ``.venv-robolab-sim/robolab``。若系统依赖已安装，可添加 ``--no-root`` 跳过系统包安装；完整系统前提见 RLinf 指南。

复用已有 RLinf 安装时，请确保源码已同步到 ``main``，且仿真环境已安装 RLinf，再将所有运行命令中的 ``--sim-python`` 指向该环境的 ``bin/python``。如果设置了 ``ROBOLAB_PATH``，安装器会使用该源码目录；相应地，在该目录执行 ``git -C "$ROBOLAB_PATH" lfs pull``。

完整 RPent 包仅安装在策略环境中。仿真进程通过启动器设置的 ``PYTHONPATH`` 从本仓库加载环境服务和共享 RPC 模块。将规划器依赖装入仿真环境，会与 Isaac Sim 固定的版本冲突。仅在策略环境安装 RLinf 不够：仿真解释器还要运行 RLinf 的 RoboLab 适配器及其 Isaac 子进程。安装完成后检查两个环境及适配器导入：

.. code-block:: bash

   uv pip check --python .venv-robolab-policy/bin/python
   uv pip check --python .venv-robolab-sim/bin/python
   .venv-robolab-sim/bin/python -c \
     "from rlinf.envs.sim.robolab.robolab_env import RoboLabEnv; print(RoboLabEnv.__module__)"

模型权重
~~~~~~~~

在已激活的策略环境中，从 Hugging Face 下载公开的 `RLinf-Pi05-RoboLab-droid-jointpos 权重 <https://huggingface.co/RLinf/RLinf-Pi05-RoboLab-droid-jointpos>`_：

.. code-block:: bash

   export PI05_CHECKPOINT_PATH="$HOME/.cache/rpent/checkpoints/RLinf-Pi05-RoboLab-droid-jointpos"
   hf download RLinf/RLinf-Pi05-RoboLab-droid-jointpos \
     --local-dir "$PI05_CHECKPOINT_PATH"

目录包含 ``model.safetensors``、``config.json`` 和 ``assets/droid/norm_stats.json``，请保持目录结构完整。策略使用 RLinf 的 ``pi05_droid_polaris`` 配置：两路图像、10 步去噪、15 步预测长度和八维环境动作。加载器接收 PyTorch 权重，不能直接使用 JAX checkpoint。在新终端中重新设置 ``PI05_CHECKPOINT_PATH``，或通过 ``--am-checkpoint`` 指定该目录。

运行一个任务
------------

按照 :doc:`../guides/configure_planner` 配置规划器。示例使用当前用户的 Codex 登录和已配置的模型，此登录方式无需 API Key。在 RPent 仓库根目录运行，并保留上文设置的 checkpoint 路径。将 ``--cuda-device`` 设为显存充足的物理 GPU 编号；示例使用 GPU 0。

.. code-block:: bash

   source .venv-robolab-policy/bin/activate
   rpent-check-llm --planner codex
   rpent --robot robolab --task BananaInBowlTask --seed 0 \
     --sim-python "$PWD/.venv-robolab-sim/bin/python" \
     --cuda-device 0 --planner codex \
     --output-dir ./logs/first-robolab

动作模型部署使用共享的 ``--am-checkpoint``、``--am-python``、``--am-root`` 和 ``--am-cuda-device`` 参数。复用已有模型服务时，传入 ``--am-endpoint``，不要同时传入本地部署参数。

使用 ``--env-cuda-device`` 可单独覆盖仿真环境的 ``--cuda-device``，与动作模型独立选卡。显式选择服务 GPU 后，该服务使用自身 CUDA 掩码内的 ``cuda:0``；否则，``--device`` 在继承的 CUDA 命名空间中选择 Isaac 设备，默认为 ``cuda:0``。在调度器或容器中，省略服务 GPU 参数可继承分配的掩码，包括 GPU UUID。RLinf 会将数字形式的 CUDA 掩码映射为 Isaac 和 Kit 使用的 GPU 编号，并保留非数字掩码。使用容器或 UUID 分配 GPU 时，请通过 Isaac 日志确认实际渲染设备。使用 ``--env-endpoint`` 可连接已有的同任务、同 seed 环境服务，其 GPU 无需与本地设备设置相同。

不加载模型权重、仅使用环境动作时，加入 ``--policy-backend none``。此时可使用 ``execute_action`` 和 ``view_env_state``，但没有 ``pi05_act``。

要使用 Dashboard，将 ``--task BananaInBowlTask --seed 0`` 改为 ``--dashboard``，再通过 ``/rpent-task BananaInBowlTask 0`` 创建任务。Dashboard 在多个任务间共用 Pi0.5，并为每个任务创建新环境；传入 ``--policy-backend none`` 时，仅启动环境并提供原生动作，不显示 VLA 组件或 ``pi05_act``。

查看结果
--------

``logs/first-robolab/states.json`` 保存 RPent 步骤记录；``main.png`` 和 ``wrist.png`` 目录保存观测图像。``result.json`` 单独记录原生任务成功状态；RoboLab 尚未返回终止结果时，``success`` 为 ``null``。``env_server.log`` 保存环境日志，启动本地模型时还会生成 ``vla_server.log``。RoboLab 原生输出位于 ``logs/first-robolab/native/``。

查看最后一次 ``view_env_state``：只有 RoboLab 报告成功时，``state.success`` 才为 ``true``。原生任务尚未结束时，该值为 ``null``；达到步数上限不代表成功。

任务记忆
--------

目前没有公开提供的 RoboLab 任务记忆库。RPent 会创建本地 RoboLab 记忆目录，全新安装时该目录为空，默认不下载记忆库。传入 ``--memory-profile local --memory-dir /path/to/memory`` 可复用自己的任务笔记。也可通过 ``--memory-repo owner/repo@revision``，将指定 Hugging Face 数据集的 ``robolab/`` 子目录下载到 RPent 记忆缓存；不要与 ``--memory-profile local`` 或 ``--memory-dir`` 同时使用。RoboLab 暂不支持 RPent 探索模式。

评估任务子集
------------

配置好模型权重与规划器后，在 RPent 仓库根目录执行以下命令，串行评估六个任务，每个任务使用环境 seed 0、1、2。开始前请选择可用 GPU。每次执行都会创建新的输出目录，每个回合使用独立的空白本地记忆：

.. code-block:: bash

   (
     set -e
     mkdir -p ./logs
     eval_root="$(mktemp -d "$PWD/logs/robolab-subset-XXXXXX")"
     for task in BananaInBowlTask BowlStackingLeftOnRightTask \
       BowlStackingRightOnLeftTask BananaThenRubiksCubeTask \
       ReorientRedMugTask StackYellowOnRedTask; do
       for seed in 0 1 2; do
         episode_dir="$eval_root/${task}_s${seed}"
         rpent --robot robolab --task "$task" --seed "$seed" \
           --sim-python "$PWD/.venv-robolab-sim/bin/python" \
           --cuda-device 0 --planner codex --reasoning-effort low \
           --max-turns 400 --planner-timeout-s 3600 \
           --memory-profile local --memory-dir "$episode_dir/memory" \
           --output-dir "$episode_dir"
       done
     done
   )

每回合最多 400 个规划轮次、3600 秒规划预算，仿真启动时间不计入该预算。环境 seed 不会固定策略采样随机数和规划器决策；规划器可缩短动作块或直接发送动作。这是子集评估，不代表完整基准结果。

逐回合检查 ``result.json`` 和日志，仅将 ``success`` 为 ``true`` 的回合计为原生成功，分别报告 ``false`` 和 ``null``。以计划的全部 18 回合为分母，明确列出缺失结果、运行错误和中断。循环在进程非零退出时停止，不自动重试或生成汇总报告。需要重试时，为对应任务和 seed 指定新的输出与记忆目录，保留原先的失败尝试。比较结果时，还应记录代码与资源版本、依赖版本、规划模型和重试规则。

问题排查
--------

若环境启动失败，查看 ``env_server.log``，确认 ``--sim-python`` 指向通过 RLinf 安装器创建的仿真环境，能够导入 ``rlinf.envs.sim.robolab.robolab_env``，并以 editable 模式安装了带完整 ``assets/`` 的 RoboLab。若策略启动失败，查看 ``vla_server.log`` 和 checkpoint 路径。

遇到 Vulkan 初始化错误或缺少 ``libGLU.so.1`` 时，检查 NVIDIA 驱动和上文列出的系统图形库，并查看 Isaac 日志。渲染和物理仿真使用选定的物理 GPU，并禁用多 GPU 渲染；Isaac 仍可能在其他 GPU 创建少量 CUDA 上下文。在共享机器上使用 ``nvidia-smi`` 检查实际显存占用。

下载 wheel 超时时，可使用相同的 ``--venv`` 路径和缓存重新运行 RLinf 安装器。依赖冲突应分别查看两个环境的 ``uv pip check`` 输出。进程正常退出本身不能证明原生任务成功。
