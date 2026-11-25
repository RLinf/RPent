RoboLab
=======

通过 RPent 的观测与动作工具，在 Isaac Lab 中运行 RoboLab 基准任务。

概览
----

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 动作模型

      DROID 关节位置控制的 Pi0.5，或不使用模型直接执行原生动作。

   .. grid-item-card:: 规划器

      ``api``、``claude_code``、``codex``、``gemini_cli``。

   .. grid-item-card:: 任务

      RoboLab-120，共 120 个任务，由已安装的 RoboLab 版本提供。

   .. grid-item-card:: 硬件

      Ubuntu 22.04 及以上、带可用 Vulkan 驱动的 NVIDIA GPU（推荐 48 GB 以上显存）；Python 3.11；Isaac Sim 5.0 / Isaac Lab 2.2.0。

任务
~~~~

RoboLab-120 是 RoboLab 的基准套件，包含 120 个任务，涵盖抓放、堆叠、重排与工具使用。任务类位于源码的 ``robolab/tasks/benchmark/`` 下，用 ``ls <检出目录>/robolab/tasks/benchmark`` 列出，类名就是 ``--task`` 的取值。`RoboLab 任务目录 <https://github.com/NVlabs/RoboLab/blob/main/robolab/tasks/README.md>`_ 给出每个任务的指令、难度与回合时长。

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - 选项
     - 作用
   * - 套件与回合上限
     - RoboLab-120 共 120 个任务；回合上限由任务自身决定，RPent 不截断，各任务在 20 到 300 秒之间。

观测与动作
~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - 项目
     - 说明
   * - 观测
     - DROID 本体（固定底座 Franka 臂、Robotiq 2F-85 夹爪、腕部相机、高增益关节控制）的外部与腕部 RGB 图像、八个关节与夹爪状态值、任务指令。策略接收 224×224 图像；保存给规划器的图像保持相机原始分辨率。
   * - 动作
     - ``pi05_act`` 执行一次 Pi0.5 预测中的 1–15 个动作，默认 5 个。``execute_action`` 发送一个八维关节位置动作：七个臂关节加夹爪。
   * - 奖励与成功判定
     - RoboLab 的原生结果写入 ``view_env_state`` 的 ``state.success``；规划器的 ``finish`` 状态与其分开。读取方法见“查看结果”。
   * - 任务指令
     - 当前环境的任务指令，也会传给 Pi0.5。

安装与资源准备
--------------

验证过的配置使用两个 Python 3.11 环境：RPent 与 Pi0.5 在策略环境，Isaac Sim 5.0 和 Isaac Lab 2.2.0 在仿真环境。已有可用的 RoboLab 仿真环境时，把 ``SIM_PYTHON`` 设为它的解释器，跳到“模型权重”。

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - 项目
     - 要求
   * - 操作系统
     - Ubuntu 22.04 及以上。
   * - GPU 与驱动
     - 带可用 Vulkan 驱动的 NVIDIA GPU。上游 RoboLab 要求 RTX，推荐 48 GB 以上显存；RLinf 实测使用 A800 80 GB 显卡与 570.172.08 驱动。
   * - 磁盘
     - 约 8 GB，其中资产约 7 GB，不含 Isaac Sim 和两个 Python 环境。
   * - 软件
     - Python 3.11、``uv``、Git LFS、ffmpeg，以及 EGL、GLU、OpenGL、Vulkan 运行库。

在 Ubuntu 上安装系统软件包和 `uv <https://docs.astral.sh/uv/getting-started/installation/>`_：

.. code-block:: bash

   sudo apt update
   sudo apt install -y git-lfs ffmpeg \
     libegl1 libglu1-mesa libgl1 libvulkan1 vulkan-tools
   git lfs install
   curl -LsSf https://astral.sh/uv/install.sh | sh

NVIDIA 专有驱动按 Isaac Sim 5.0 的系统要求单独安装，见 `RLinf RoboLab 指南 <https://rlinf.readthedocs.io/zh-cn/latest/rst_source/evaluations/guides/robolab.html>`_。

安装 RPent 和策略
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   uv venv --python 3.11 .venv-robolab-policy
   source .venv-robolab-policy/bin/activate
   uv pip install -e ".[robolab]"

``robolab`` extra 提供 RPent、RLinf 的 Pi0.5 加载器和策略服务使用的 OpenPI 客户端；Isaac Sim 与 RoboLab 由下文的仿真环境提供。

安装仿真环境
~~~~~~~~~~~~

RLinf 安装器会克隆 RoboLab 并切换到固定版本，把 Isaac Sim 5.0、Isaac Lab 2.2.0 和以 editable 模式安装的 RoboLab 及其场景资产装入新的 ``.venv``。把 ``ROBOLAB_PATH`` 指向已有的 RoboLab 目录可跳过克隆和切换版本。

.. code-block:: bash

   git clone https://github.com/RLinf/RLinf.git "$HOME/RLinf"
   cd "$HOME/RLinf"
   bash requirements/install.sh embodied --env robolab --install-rlinf
   ROBOLAB_DIR="${ROBOLAB_PATH:-$PWD/.venv/robolab}"
   git -C "$ROBOLAB_DIR" lfs pull
   export SIM_PYTHON="$PWD/.venv/bin/python"
   "$SIM_PYTHON" -c 'from rlinf.envs.sim.robolab.robolab_env import RoboLabEnv'
   cd -

模型权重
--------

在已激活的策略环境中，从 Hugging Face 下载公开的 `RLinf-Pi05-RoboLab-droid-jointpos 权重 <https://huggingface.co/RLinf/RLinf-Pi05-RoboLab-droid-jointpos>`_：

.. code-block:: bash

   export PI05_CHECKPOINT_PATH="$HOME/.cache/rpent/checkpoints/RLinf-Pi05-RoboLab-droid-jointpos"
   hf download RLinf/RLinf-Pi05-RoboLab-droid-jointpos \
     --local-dir "$PI05_CHECKPOINT_PATH"

该权重是 RoboLab 的 ``pi05_droid_jointpos`` checkpoint，由 RLinf 转换为 PyTorch 并使用 ``pi05_droid_polaris`` 配置。加载器不能直接使用 JAX checkpoint。在新终端中重新设置 ``PI05_CHECKPOINT_PATH``，或通过 ``--am-checkpoint`` 指定该目录。

运行一个任务
------------

规划器配置见 :doc:`../guides/configure_planner`，全部参数见 :doc:`../guides/cli`。示例使用当前用户的 Codex 登录，无需 API Key；此时 ``SIM_PYTHON`` 与 ``PI05_CHECKPOINT_PATH`` 已设置：

.. code-block:: bash

   source .venv-robolab-policy/bin/activate
   rpent-check-llm --planner codex
   rpent --robot robolab --task BananaInBowlTask --seed 0 \
     --sim-python "$SIM_PYTHON" \
     --cuda-device 0 --planner codex \
     --output-dir ./logs/first-robolab

``rpent-check-llm`` 在规划器应答时以 0 退出，否则显示 ``FAILED`` 状态。运行时 RPent 把仿真器和 Pi0.5 服务作为子进程启动，再把任务交给规划器；用 ``--env-endpoint`` 或 ``--am-endpoint`` 指向已在运行的服务时，对应组件不在本地启动。首次启动较慢：Isaac Sim 需要几分钟构建渲染上下文。RPent 等待仿真器最多 900 秒、等待模型服务最多 1800 秒，超时后报错并指出未连上的组件。

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - 参数
     - 作用
   * - ``--task``
     - RoboLab 任务类；加 ``--dashboard`` 时可以不填，用法见 :doc:`../guides/dashboard`。
   * - ``--seed``
     - 原生环境的随机种子，默认 ``0``。更换种子是否改变场景布局取决于任务。
   * - ``--sim-python``
     - 仿真环境的 Python，不要指向策略环境。
   * - ``--output-dir``
     - 日志与结果目录，默认 ``<仓库>/logs/<时间戳>_robolab_<task>_s<seed>``。
   * - ``--cuda-device`` / ``--env-cuda-device``
     - 模型服务与仿真器的 GPU；后者单独指定仿真器，如 ``--cuda-device 0 --env-cuda-device 1``。
   * - ``--env-endpoint``
     - 已在运行的仿真服务 URL；其任务与随机种子必须和 ``--task``、``--seed`` 一致。

动作模型部署使用共享的 ``--am-checkpoint``、``--am-python``、``--am-root`` 和 ``--am-cuda-device`` 参数，复用已有模型服务时传入 ``--am-endpoint``；不加载模型权重时加入 ``--policy-backend none``，仍可使用 ``execute_action`` 和 ``view_env_state``，但没有 ``pi05_act``。

查看结果
--------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 文件
     - 内容
   * - ``result.json``
     - 每次运行一条记录，包含 ``task``、``seed``、``success``、``elapsed_s``、规划器后端与模型，以及两个标志：``environment_result_available`` 在回合结束时没有原生终止结果时为 ``false``，``agent_error_present`` 在规划器报错时为 ``true``。
   * - ``states.json``、``main.png``、``wrist.png``、``native/``
     - RPent 步骤记录（含每次工具调用及结果）、保存给规划器的图像，以及 RoboLab 原生的回合输出。
   * - ``env_server.log``、``vla_server.log``、``run.log``、``transcript_*.json``
     - 服务与规划器日志；``vla_server.log`` 只在 RPent 启动本地模型时生成。

查看最后一次 ``view_env_state`` 返回的 ``state.success``。RoboLab 判定成功时为 ``true``，失败时为 ``false``，尚无原生判定时为 ``null``。运行摘要将该值写入 ``result.json``。

任务记忆
--------

RPent 在 RPent 仓库下创建空的 ``memory/robolab/`` 并默认从这里读取记忆；RoboLab 没有公开的记忆库。要复用自己的记录，传入 ``--memory-profile local --memory-dir /path/to/memory``。``--memory-repo owner/repo@revision`` 会改为下载指定 Hugging Face 数据集中 ``robolab/`` 子目录，不能与上述本地选项同时使用。RoboLab 不支持探索模式；记忆来源与文件结构见 :doc:`记忆指南 <../guides/memory>`。

常见问题
--------

若环境启动失败，查看 ``env_server.log``；``--sim-python`` 必须指向 RLinf 安装的仿真环境，且能导入 ``rlinf.envs.sim.robolab.robolab_env``。若策略启动失败，查看 ``vla_server.log``；常见原因是当前终端没有设置 ``PI05_CHECKPOINT_PATH``。

遇到 Vulkan 错误或缺少 ``libGLU.so.1`` 时，检查驱动及安装与资源准备表格中的图形库；在调度器或容器中省略 ``--cuda-device 0``，让服务继承分配的 GPU。下载 wheel 超时时，用相同的 ``--venv`` 路径和缓存重新运行安装器，系统软件包已装好时加 ``--no-root``；依赖冲突分别查看两个环境的 ``uv pip check`` 输出。

.. note::

   本页于 2026-11-24 对照 RPent ``3ed1dbc`` 与 RLinf 安装器（RoboLab ``b6cfcf6``）核对；A800 80 GB / 驱动 570.172.08 是 RLinf 实测的配置，不是最低要求。
