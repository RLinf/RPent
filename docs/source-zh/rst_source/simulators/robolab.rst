RoboLab
=======

通过 RPent 的观测和动作工具运行 Isaac Lab RoboLab 任务，每次运行连接一个 DROID 关节位置控制环境。任务成功由 RoboLab 原生评估器判定。本地六任务评估属于子集实验，不代表完整 RoboLab 基准结果。

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

准备 Linux、兼容的 NVIDIA 驱动、Python 3.11、``uv``、Git、Git LFS 和 ``ffmpeg``，然后在 RPent 仓库根目录运行：

.. code-block:: bash

   bash script/install_robolab.sh

脚本创建 ``.venv-robolab-policy`` 和 ``.venv-robolab-sim``，安装 RPent 的 ``robolab`` 可选依赖，并将 RoboLab 固定版本 ``b6cfcf6`` 及 LFS 资产下载到仓库旁的 ``../RoboLab-runtime``。可通过第一个参数指定其他外部目录；已有 checkout 必须符合固定版本。模型权重需要单独准备。

安装器会检查两个环境及图形运行库。在 Ubuntu 22.04 x86_64 上，缺少的 EGL、GLU、OpenGL 和 Vulkan 库通过 ``curl``、``dpkg-deb`` 和 ``sha256sum`` 安装到 ``.venv-robolab-sim/system-libs``，无需 sudo，也不修改 NVIDIA 驱动。其他发行版需要通过系统包管理器安装缺少的图形库。下载并发和超时可通过 ``UV_*`` 环境变量调整，具体见 ``bash script/install_robolab.sh --help``。

手动安装
~~~~~~~~

手动安装时，沿用其他仿真环境的 ``uv venv`` 和可选依赖安装方式：

.. code-block:: bash

   uv venv --python 3.11 .venv-robolab-policy
   source .venv-robolab-policy/bin/activate
   uv pip install -e ".[robolab]"

RoboLab 从源码包相邻的 ``assets/`` 读取 USD 资产。因此另建 Isaac 环境，并在
RPent 仓库之外安装带资产的 RoboLab 源码，并安装 ``openpi-client``。
``git lfs pull`` 下载任务所需资产。以下命令使用 Isaac Sim 5.0：

.. code-block:: bash

   export ROBOLAB_SOURCE="$HOME/.cache/rpent/RoboLab"
   git lfs install
   mkdir -p "$(dirname "$ROBOLAB_SOURCE")"
   GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/NVlabs/RoboLab.git "$ROBOLAB_SOURCE"
   git -C "$ROBOLAB_SOURCE" checkout b6cfcf6
   git -C "$ROBOLAB_SOURCE" lfs pull
   export RPENT_ROOT="$PWD"
   uv venv --python 3.11 .venv-robolab-sim
   (
     cd "$ROBOLAB_SOURCE"
     UV_PROJECT_ENVIRONMENT="$RPENT_ROOT/.venv-robolab-sim" \
       uv sync --python 3.11 --extra isaac50 --no-dev --locked
     uv pip install --python "$RPENT_ROOT/.venv-robolab-sim/bin/python" \
       -e ".[isaac50]" openpi-client
   )

完成手动安装后，检查图形库和依赖：

.. code-block:: bash

   bash script/install_robolab_graphics.sh
   uv pip check --python .venv-robolab-policy/bin/python
   uv pip check --python .venv-robolab-sim/bin/python

完整 RPent 包仅安装在策略环境中。仿真进程通过启动器设置的 ``PYTHONPATH``，
从本仓库加载环境服务和共享 RPC 模块。将 RPent 的规划器依赖装入 Isaac 环境，
会与 Isaac Sim 固定的依赖版本冲突。

模型权重
~~~~~~~~

仿真进程从已安装的 ``robolab`` 包及其相邻的 ``assets/`` 读取资源，不读取
RPent 的 ``ref-repo/`` 或 ``data/``。另行设置 Pi0.5 检查点路径，路径可以在仓库外：

.. code-block:: bash

   export PI05_CHECKPOINT_PATH=/absolute/path/to/pi05_droid_jointpos_torch

检查点需要匹配 DROID 关节位置 Pi0.5 模型：预测长度为 15，环境动作维度为 8。
策略环境使用 RLinf 的 ``pi05_droid_polaris`` 模型配置。请提供兼容的 PyTorch 权重目录，包含 ``model.safetensors`` 和 ``assets/droid/norm_stats.json``。安装脚本不会下载或转换模型权重，JAX checkpoint 不能直接传入此加载器。运行不要求仓库中预先存在本地数据或权重目录；通过 ``--pi05-checkpoint`` 或 ``PI05_CHECKPOINT_PATH`` 指定自己的路径。

运行一个任务
------------

激活策略环境并检查本地 Codex 登录。随依赖安装的 CLI 使用当前用户登录，此登录方式无需 API Key。其他规划器及认证方式见 :doc:`../guides/configure_planner`。

.. code-block:: bash

   source .venv-robolab-policy/bin/activate
   export CODEX_BIN="$(python -c 'from codex_cli_bin import bundled_codex_path; print(bundled_codex_path())')"
   "$CODEX_BIN" login
   "$CODEX_BIN" login status
   rpent-check-llm --planner codex --timeout-s 90 --json

连接检查成功后运行任务。将 ``--cuda-device`` 设置为显存充足的物理 GPU 编号，示例使用 GPU 0。规划器默认读取 Codex 配置中的模型，也可以通过 ``--model`` 指定。


.. code-block:: bash

   rpent --robot robolab --task BananaInBowlTask --seed 0 \
     --sim-python "$PWD/.venv-robolab-sim/bin/python" \
     --cuda-device 0 --planner codex \
     --output-dir ./logs/first-robolab

快捷脚本会在启动任务前检查当前 Codex 登录：

.. code-block:: bash

   bash script/run_robolab.sh "$PI05_CHECKPOINT_PATH" BananaInBowlTask 0 ./logs/first-robolab

运行快捷脚本的终端应配置好仿真需要的 GPU 可见性；上面的 CLI 命令可直接指定物理 GPU。脚本使用现有 Codex 配置和登录。

只检查环境而不加载检查点时，加入 ``--policy-backend none``。此时可使用
``execute_action`` 和 ``view_env_state``，但没有 ``pi05_act``。

要使用 Dashboard，将 ``--task BananaInBowlTask --seed 0`` 改为
``--dashboard``，再通过 ``/rpent-task BananaInBowlTask 0`` 创建任务。
Dashboard 在多个任务间共用 Pi0.5，并为每个任务创建新环境；该模式需要 Pi0.5。

查看结果
--------

``logs/first-robolab/states.json`` 保存 RPent 步骤记录；``main.png`` 和
``wrist.png`` 目录保存观测图像。``result.json`` 单独记录原生任务成功状态；
RoboLab 尚未返回终止结果时，``success`` 为 ``null``。
``env_server.log`` 与 ``vla_server.log``
分别保存组件日志。RoboLab 原生输出位于 ``logs/first-robolab/native/``。

查看最后一次 ``view_env_state``：只有 RoboLab 报告成功时，``state.success``
才为 ``true``。原生任务尚未结束时，该值为 ``null``；达到步数上限不代表成功。

任务记忆
--------

RoboLab 默认使用本地记忆目录。传入
``--memory-profile local --memory-dir /path/to/memory`` 可复用自己的任务笔记。
运行时不会下载 RoboLab 专属记忆库。

评估任务子集
------------

评估脚本串行运行六个任务，各使用环境 seed 0、1、2：``BananaInBowlTask``、``BowlStackingLeftOnRightTask``、``BowlStackingRightOnLeftTask``、``BananaThenRubiksCubeTask``、``ReorientRedMugTask`` 和 ``StackYellowOnRedTask``。模型权重路径必须显式传入，结果默认保存到 ``logs/robolab_eval/unaligned_6tasks_3seeds/``。先预览运行命令：

.. code-block:: bash

   bash script/eval_robolab_subset.sh \
     --checkpoint "$PI05_CHECKPOINT_PATH" --gpu 0 \
     --output-dir ./logs/robolab-subset --dry-run

去掉 ``--dry-run`` 后开始执行，也可通过 ``--tasks`` 和 ``--seeds`` 选择更小的子集。默认每回合最多 400 个规划轮次、3600 秒规划预算，推理强度为 ``low``；仿真启动时间不计入规划预算。每次尝试使用空白的 RPent 本地记忆，Codex 会话存储遵循用户配置。环境 seed 不会固定策略采样随机数。策略使用上游 RLinf 实现、10 步去噪、两路图像，每次预测最多执行 15 个动作；规划器可缩短动作块或直接发送动作。

``manifest.json`` 保存配置、依赖版本和源码哈希。``summary.json`` 与 ``summary.csv`` 分别统计原生成功、失败、提前结束、预算耗尽、运行错误和中断。成功率以已结束的评估回合为分母，其中包括提前结束和预算耗尽；运行错误与中断单独报告。重复相同命令会跳过已结束回合，对错误或中断使用新的 attempt 目录重试。遇到运行错误时脚本停止；配置或源码改变后必须使用新输出目录，避免混合结果。

问题排查
--------

若环境启动失败，查看 ``env_server.log``，确认 ``--sim-python`` 指向的
Isaac 环境以 editable 模式安装了带 ``assets/`` 的 RoboLab。若策略启动失败，
查看 ``vla_server.log`` 和检查点路径。仿真与策略依赖冲突时，保持两个 Python
环境分离。

遇到 Vulkan 初始化错误或缺少 ``libGLU.so.1`` 时，重新执行 ``bash script/install_robolab_graphics.sh`` 并查看 Isaac 日志；启动器会自动加入本地图形库路径。渲染和物理仿真使用选定的物理 GPU，并禁用多 GPU 渲染；Isaac 仍可能在其他 GPU 创建少量 CUDA 上下文。在共享机器上使用 ``nvidia-smi`` 检查实际显存占用。

下载 wheel 超时时，可对相同外部目录和虚拟环境重新执行安装脚本，已有下载缓存会保留。依赖冲突应分别查看两个环境的 ``uv pip check`` 输出。Codex 网络错误记录在规划器日志中，可能自动重试；进程正常退出本身不能证明原生任务成功。
