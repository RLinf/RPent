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

在 Linux 主机上准备两个 Python 3.11 环境。RPent 和 Pi0.5 使用策略环境，Isaac Sim 5.0 和 Isaac Lab 2.2.0 使用仿真环境。两边依赖的 PyTorch 版本不同。

安装 RPent 和策略
~~~~~~~~~~~~~~~~~~~~~~

从 RPent 仓库根目录执行：

.. code-block:: bash

   uv venv --python 3.11 .venv-robolab-policy
   source .venv-robolab-policy/bin/activate
   uv pip install -e ".[robolab]"

复用已有仿真环境
~~~~~~~~~~~~~~~~~~~~~~~~

如果已有按 RLinf 流程安装、能运行 RoboLab 任务的环境，将下面的路径换成该环境的 Python。后续运行命令通过 ``--sim-python`` 使用它。设置后可直接跳到“模型权重”。

.. code-block:: bash

   export SIM_PYTHON=/path/to/robolab-env/bin/python

首次安装仿真环境
~~~~~~~~~~~~~~~~~~~~~~~~

首次安装需要兼容的 NVIDIA 驱动、Git LFS、ffmpeg 和 EGL、GLU、OpenGL、Vulkan 运行库。按 `RLinf RoboLab 指南 <https://rlinf.readthedocs.io/zh-cn/latest/rst_source/evaluations/guides/robolab.html>`_ 准备系统依赖，然后从 RPent 根目录运行：

.. code-block:: bash

   git lfs install
   git clone https://github.com/RLinf/RLinf.git "$HOME/RLinf"
   cd "$HOME/RLinf"
   bash requirements/install.sh embodied --env robolab --install-rlinf
   git -C .venv/robolab lfs pull
   export SIM_PYTHON="$PWD/.venv/bin/python"
   cd -

示例将 RLinf 放在主目录下，安装器默认在该源码目录内创建 ``.venv``。请保留源码目录。若已设置 ``ROBOLAB_PATH``，在该目录拉取 LFS 资产。系统依赖已装好时，可给安装器加 ``--no-root``。检查仿真环境能否导入适配器：

.. code-block:: bash

   "$SIM_PYTHON" -c 'from rlinf.envs.sim.robolab.robolab_env import RoboLabEnv'

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

按照 :doc:`../guides/configure_planner` 配置规划器。示例使用当前用户的 Codex 登录和已配置的模型，此登录方式无需 API Key。在设置好 ``SIM_PYTHON`` 和 ``PI05_CHECKPOINT_PATH`` 的终端中，从 RPent 仓库根目录运行：

.. code-block:: bash

   source .venv-robolab-policy/bin/activate
   rpent-check-llm --planner codex
   rpent --robot robolab --task BananaInBowlTask --seed 0 \
     --sim-python "$SIM_PYTHON" \
     --cuda-device 0 --planner codex \
     --output-dir ./logs/first-robolab

动作模型部署使用共享的 ``--am-checkpoint``、``--am-python``、``--am-root`` 和 ``--am-cuda-device`` 参数。复用已有模型服务时，传入 ``--am-endpoint``，不要同时传入本地部署参数。

示例中仿真和模型共用 GPU 0。若要让模型用 GPU 0、仿真用 GPU 1，使用 ``--cuda-device 0 --env-cuda-device 1``。连接已有仿真服务时，使用 ``--env-endpoint`` 并指定与服务匹配的任务和 seed。

不加载模型权重、仅使用环境动作时，加入 ``--policy-backend none``。此时可使用 ``execute_action`` 和 ``view_env_state``，但没有 ``pi05_act``。

要使用 Dashboard，将 ``--task BananaInBowlTask --seed 0`` 改为 ``--dashboard``，再通过 ``/rpent-task BananaInBowlTask 0`` 创建任务。Dashboard 在多个任务间共用 Pi0.5，并为每个任务创建新环境；传入 ``--policy-backend none`` 时，仅启动环境并提供原生动作，不显示 VLA 组件或 ``pi05_act``。

查看结果
--------

``logs/first-robolab/states.json`` 保存 RPent 步骤记录；``main.png`` 和 ``wrist.png`` 目录保存观测图像。``result.json`` 单独记录原生任务成功状态；RoboLab 尚未返回终止结果时，``success`` 为 ``null``。``env_server.log`` 保存环境日志，启动本地模型时还会生成 ``vla_server.log``。RoboLab 原生输出位于 ``logs/first-robolab/native/``。

查看最后一次 ``view_env_state``：只有 RoboLab 报告成功时，``state.success`` 才为 ``true``。原生任务尚未结束时，该值为 ``null``；达到步数上限不代表成功。

任务记忆
--------

目前没有公开提供的 RoboLab 任务记忆库。RPent 会创建本地 RoboLab 记忆目录，全新安装时该目录为空，默认不下载记忆库。传入 ``--memory-profile local --memory-dir /path/to/memory`` 可复用自己的任务笔记。也可通过 ``--memory-repo owner/repo@revision``，将指定 Hugging Face 数据集的 ``robolab/`` 子目录下载到 RPent 记忆缓存；不要与 ``--memory-profile local`` 或 ``--memory-dir`` 同时使用。RoboLab 暂不支持 RPent 探索模式。

问题排查
--------

若环境启动失败，查看 ``env_server.log``，确认 ``--sim-python`` 指向通过 RLinf 安装器创建的仿真环境，能够导入 ``rlinf.envs.sim.robolab.robolab_env``，并以 editable 模式安装了带完整 ``assets/`` 的 RoboLab。若策略启动失败，查看 ``vla_server.log`` 和 checkpoint 路径。

遇到 Vulkan 初始化错误或缺少 ``libGLU.so.1`` 时，检查 NVIDIA 驱动及图形运行库。在调度器或容器中，先省略示例中的 ``--cuda-device 0``，让服务继承分配的 GPU。若设备编号无效或渲染到了其他卡，请核对 Isaac 日志中的设备与容器可见的设备。

下载 wheel 超时时，可使用相同的 ``--venv`` 路径和缓存重新运行 RLinf 安装器。依赖冲突应分别查看两个环境的 ``uv pip check`` 输出。进程正常退出本身不能证明原生任务成功。
