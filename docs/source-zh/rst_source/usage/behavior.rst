BEHAVIOR
========

`BEHAVIOR-1K <https://behavior.stanford.edu/>`_ 基于 OmniGibson 提供长程家庭任务。
RPent 当前把 ``turning_on_radio`` 和 ``picking_up_trash`` 作为标准 sibling
robot plugin 接入，源码位于 ``robots/behavior``。

接入合同与 LIBERO、RoboCasa、RoboTwin 相同：``get_robot_spec()`` 提供
CLI/config/runtime hooks，``get_toolkit()`` 提供公开工具。BEHAVIOR 生命周期逻辑
留在 ``robots/behavior`` 内。公共 ``--explore`` loop 仍只属于 LIBERO；
BEHAVIOR 使用 ``--behavior-mode explore`` 和自己的外层 harness。

安装状态
--------

BEHAVIOR 以源码 editable 方式运行，并使用两个相互独立的 Python 3.10 环境：

- **RPent venv**：运行 CLI、planner、Dashboard 和 MemoryManager；
- **BEHAVIOR venv**：运行 RLinf、OmniGibson、Isaac Sim 和 Pi0.5。

``.[behavior]`` 只安装 RPent 侧依赖，不包含完整模拟器、资产或 checkpoint；
普通 wheel 不承诺可直接运行 BEHAVIOR。请在源码 checkout 中执行：

.. code-block:: bash

   export RPENT_REPRO_ROOT="$PWD/.behavior-runtime"
   export UV_CACHE_DIR="$RPENT_REPRO_ROOT/uv-cache"
   bash scripts/install_behavior_runtime.sh

安装器会在两个 venv 中保持 RPent editable，克隆已审查的 RLinf revision，调用
官方 RLinf BEHAVIOR 安装器，应用已审查的 CUDA/OpenPI 兼容性 pin，验证关键 import
和 CUDA，并在 ``$RPENT_REPRO_ROOT/manifests`` 写入 freeze 与源码身份。fresh install
应使用新的 ``RPENT_REPRO_ROOT``；脚本不会覆盖 revision 错误或 dirty 的 RLinf
checkout。

仿真资产
--------

接受 BEHAVIOR/OmniGibson 许可后，选择独立数据根，并在 BEHAVIOR venv 中调用三个
官方下载函数：

.. code-block:: bash

   export OMNIGIBSON_DATA_PATH=/path/to/BEHAVIOR-1K-datasets
   export BEHAVIOR_PYTHON="$RPENT_REPRO_ROOT/venvs/behavior/bin/python"
   mkdir -p "$OMNIGIBSON_DATA_PATH"

   "$BEHAVIOR_PYTHON" -c \
     "from omnigibson.utils.asset_utils import download_omnigibson_robot_assets; download_omnigibson_robot_assets()"
   "$BEHAVIOR_PYTHON" -c \
     "from omnigibson.utils.asset_utils import download_behavior_1k_assets; download_behavior_1k_assets(accept_license=True)"
   "$BEHAVIOR_PYTHON" -c \
     "from omnigibson.utils.asset_utils import download_2025_challenge_task_instances; download_2025_challenge_task_instances()"

最终数据根必须包含：

.. code-block:: text

   BEHAVIOR-1K-datasets/
     2025-challenge-task-instances/
     behavior-1k-assets/
       scenes/
     omnigibson-robot-assets/
     omnigibson.key

Pi0.5 checkpoint
----------------

将已审查 checkpoint 下载到源码树之外：

.. code-block:: bash

   export PI05_CHECKPOINT_PATH=/path/to/RLinf-Pi05-BEHAVIOR-1K-PT50-CS32
   "$RPENT_REPRO_ROOT/venvs/rpent/bin/hf" download \
     RLinf/RLinf-Pi05-BEHAVIOR-1K-PT50-CS32 \
     --local-dir "$PI05_CHECKPOINT_PATH"

``scripts/verify_behavior_assets.sh`` 会检查 OmniGibson 必需目录，以及源码中固定的
checkpoint size/SHA binding。#136 的共享 Pi0.5 component 接收 head、left wrist、
right wrist 和 raw R1Pro proprio；原始 RPC 输出为 ``[1, 32, 23]``，公共 client
返回 ``[32, 23]``。

.. code-block:: bash

   scripts/verify_behavior_assets.sh

任务身份
--------

使用 ``--task-name`` 和 ``--public-seed``。public seed 通过
``robots/behavior/task_specs.py`` 固定映射到官方 activity instance。

.. list-table::
   :header-rows: 1
   :widths: 24 42 16 18

   * - 任务
     - 指令
     - Explore seeds
     - Eval seeds
   * - ``turning_on_radio``
     - 打开客厅桌上的收音机。
     - ``0``
     - ``1``-``9``
   * - ``picking_up_trash``
     - 把客厅的三个汽水罐放进厨房垃圾桶。
     - ``0``-``9``
     - ``10``-``19``

运行一次 Eval
-------------

每个 CUDA 子进程必须显式绑定一个物理 GPU：

.. code-block:: bash

   "$RPENT_REPRO_ROOT/venvs/rpent/bin/rpent" --robot behavior \
     --task-name turning_on_radio --public-seed 1 \
     --behavior-mode eval \
     --planner codex --model gpt-5.5 \
     --behavior-repo "$RPENT_REPRO_ROOT/RLinf" \
     --behavior-python "$RPENT_REPRO_ROOT/venvs/behavior/bin/python" \
     --activity-instance-dir \
       "$OMNIGIBSON_DATA_PATH/2025-challenge-task-instances" \
     --policy-checkpoint "$PI05_CHECKPOINT_PATH" \
     --behavior-env-cuda-device 0 \
     --behavior-model-cuda-device 1 \
     --memory-profile local \
     --memory-dir /path/to/behavior-memory

首次加载环境通常需要数分钟。env 与 VLA 是不同进程，每个进程只接收自己显式选择
的 GPU。

官方 MemoryManager
------------------

BEHAVIOR 使用和其他机器人相同的 Markdown/YAML ``MemoryManager`` 格式与公共 memory
工具；不会加载、迁移或静默回退到旧 DINO episode catalog。

- Eval 只构造一个 ``read_only`` MemoryManager；
- Explore 只构造一个 ``inbox_write`` MemoryManager，写入范围限定为
  ``<memory-dir>/_inbox/<recipe-tag>``；
- ``MEMORY.md``、``global/``、``suite/``、``task/``、``_inbox/`` 和
  ``_merged/`` 保持 RPent 标准语义。

缺失或空 corpus 是合法状态，但不会提供任何建议。需要共享已审查 memory 的运行应
显式传入同一个 ``--memory-dir``。

多次 Explore attempt 必须通过 BEHAVIOR 外层 harness 执行。它为每次 attempt 启动
fresh RPent 进程和 episode，让全部 attempt 指向同一个官方 corpus，并在结束后调用
现有 ``MemoryManager.merge_memory()``。planner 不能在单次 invocation 内 reset。

.. code-block:: bash

   python -m robots.behavior.harness explore \
     --attempts 3 \
     --output-dir /path/to/behavior-explore \
     --memory-dir /path/to/behavior-memory \
     -- \
     --task-name picking_up_trash --public-seed 0 \
     --planner codex --model gpt-5.5

如果 review 合同要求先保留 inbox、不立即发布，可传
``--no-auto-merge-memory``。只有 terminal receipt 携带官方成功时，task audit/recipe
pair 才会晋升。

Runtime 与 Dashboard
--------------------

runtime 有三个 component role：

- ``env``：task-scoped 官方 BEHAVIOR/OmniGibson 环境；
- ``vla``：共享 ``rpent/robots/components/pi05_vla_server.py`` 服务；
- ``memory``：task-scoped 官方 MemoryManager。

启动 Dashboard Session：

.. code-block:: bash

   TASK_NAME=turning_on_radio PUBLIC_SEED=1 \
     BEHAVIOR_MEMORY_DIR=/path/to/behavior-memory \
     scripts/run_behavior_dashboard.sh

Dashboard 使用公共 Start Session 流程与 head/left-wrist/right-wrist 相机视图。
BEHAVIOR 不增加 robot-local 手动按钮、手动控制 backend 或
``env.dashboard_*`` RPC。``pi0_nav_pick``、``observe``、``navigate_to``、
``move_to``、``press``、``open``、``close`` 等 planner primitive 是否可用，以
active tool schema 和 backend capability 为准。

主要日志：

.. code-block:: text

   <output-dir>/run.log
   <output-dir>/behavior_vla_server.log
   <output-dir>/tasks/<task-run>/behavior_env_server.log
   <output-dir>/tasks/<task-run>/episode.mp4
   <output-dir>/tasks/<task-run>/terminal_receipt.json

成功与诊断
----------

官方 task success 只等于当前 episode 的
``info["done"]["success"] is True``。reward、``terminated``、``truncated``、
primitive success、截图、视频和进程退出都不能替代它。receipt 只能记录 raw evidence，
不能制造成功。

启动仿真前可运行轻量源码检查：

.. code-block:: bash

   python -m robots.behavior.selfcheck

self-check 验证 plugin discovery、CLI/config、任务映射、memory profile 和公开工具数量；
它不会加载资产、启动 GPU 服务、执行动作或证明 task success。
