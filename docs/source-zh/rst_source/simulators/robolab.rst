RoboLab
=======

通过 RPent 的观测和动作工具运行 Isaac Lab RoboLab 任务。每次运行连接一个
DROID 关节位置控制环境。任务成功由 RoboLab 原生评估器判定；目前尚无 RPent
在完整 RoboLab 基准上的成功率结果。

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

在 RPent 仓库根目录创建策略环境，沿用其他仿真环境的 ``uv venv`` 和可选依赖安装方式：

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
   git clone https://github.com/NVlabs/RoboLab.git "$ROBOLAB_SOURCE"
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

完整 RPent 包仅安装在策略环境中。仿真进程通过启动器设置的 ``PYTHONPATH``，
从本仓库加载环境服务和共享 RPC 模块。将 RPent 的规划器依赖装入 Isaac 环境，
会与 Isaac Sim 固定的依赖版本冲突。

仿真进程从已安装的 ``robolab`` 包及其相邻的 ``assets/`` 读取资源，不读取
RPent 的 ``ref-repo/`` 或 ``data/``。另行设置 Pi0.5 检查点路径，路径可以在仓库外：

.. code-block:: bash

   export PI05_CHECKPOINT_PATH=/absolute/path/to/pi05_droid_jointpos_torch

检查点需要匹配 DROID 关节位置 Pi0.5 模型：预测长度为 15，环境动作维度为 8。
策略环境使用 RLinf 的 ``pi05_droid_polaris`` 模型配置。

运行一个任务
------------

激活策略环境并配置规划器后，运行：

.. code-block:: bash

   rpent --robot robolab --task BananaInBowlTask --seed 0 \
     --sim-python "$PWD/.venv-robolab-sim/bin/python" \
     --planner claude_code --model claude-opus-4-8 \
     --output-dir ./logs/first-robolab

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

问题排查
--------

若环境启动失败，查看 ``env_server.log``，确认 ``--sim-python`` 指向的
Isaac 环境以 editable 模式安装了带 ``assets/`` 的 RoboLab。若策略启动失败，
查看 ``vla_server.log`` 和检查点路径。仿真与策略依赖冲突时，保持两个 Python
环境分离。
