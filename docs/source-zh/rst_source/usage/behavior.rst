BEHAVIOR
========

`BEHAVIOR-1K <https://behavior.stanford.edu/>`_ 支持以可选 RPent robot
integration 形式维护，用于长程家庭操作任务。普通 ``rpent`` wheel 仍只
打包 ``rpent*`` 模块；它不包含 ``robots/behavior``、OmniGibson 或 Isaac Sim、
官方 BEHAVIOR 数据、大型 DINOv2 资产、策略 checkpoint、或已记录的 episode
memory。

安装边界
--------

RPent 侧稳定依赖可用以下命令安装：

.. code-block:: bash

   pip install -e ".[behavior]"

``behavior`` extra 覆盖 BEHAVIOR plugin 常用的 RPent 运行时依赖：RLinf、
OpenPI、Pi0.5 与 DINOv2 图像编码所需的 PyTorch/TorchVision、Pillow/ImageIO
视频工具，以及 RPent 的 HTTP/socket RPC 栈。它不会被加入 ``.[full]``，因为
BEHAVIOR 还依赖 pinned source checkout、官方仿真资产和大型运行资源，这些均在
wheel 之外管理。

OmniGibson、Isaac Sim、BEHAVIOR 数据、机器人资产，以及
``OMNI_KIT_ACCEPT_EULA``、BEHAVIOR asset root 等环境变量，按 upstream pinned
安装文档配置。source tree 和资源安装完成后，运行 plugin 自检：

.. code-block:: bash

   python -m robots.behavior.selfcheck

self-check 只验证 RPent 侧插件导入、任务/seed 映射、prompt 合同和公开工具数量；
它不会启动 OmniGibson，也不会验证官方资产、DINO 权重或 policy checkpoint。
这些大型运行资源应通过 pinned upstream 安装检查和有界 smoke 单独验证，且不要
放入 RPent package data。

运行范围
--------

当前 RPent BEHAVIOR runtime 仅覆盖已审查的 Radio 和 Trash 任务面：

- ``turning_on_radio``：操作 radio button。
- ``picking_up_trash``：将 soda can 放入 kitchen trash can。

其他 BEHAVIOR task 可用于开发探索，但在获得独立 task spec、prompt、memory 与
receipt 之前，不属于本文档承诺的 runtime contract。

最小 Eval
---------

从包含 ``robots/behavior`` 的 source checkout 运行评测：

.. code-block:: bash

   hf download RLinf/RLinf-Pi05-BEHAVIOR-1K-PT50-CS32 \
     --local-dir ./checkpoints/RLinf-Pi05-BEHAVIOR-1K-PT50-CS32
   export PI05_CHECKPOINT_PATH=$PWD/checkpoints/RLinf-Pi05-BEHAVIOR-1K-PT50-CS32
   export BEHAVIOR_ENV_GPU=2
   export BEHAVIOR_MODEL_GPU=7

   rpent --robot behavior \
     --task-name turning_on_radio \
     --public-seed 1 \
     --behavior-mode eval \
     --model gpt-5.5 \
     --behavior-env-cuda-device "$BEHAVIOR_ENV_GPU" \
     --behavior-model-cuda-device "$BEHAVIOR_MODEL_GPU" \
     --dino-source-archive /path/to/dinov2-source.tar.gz \
     --dino-weights /path/to/dinov2_vits14_pretrain.pth \
     --behavior-memory-dir /path/to/reviewed-episode-catalog \
     --output-dir /path/to/behavior-eval

Eval 是正式的单次测量路径，必须保留 raw action trace 和最终 artifact。官方任务
成功只看 BEHAVIOR 原始位：``info["done"]["success"]``，即 trace 中记录的
``info_done.success``。planner 进展、primitive success、``task_success``、
workflow sealing、terminal receipt 和公开发布状态都要作为独立结论报告。

独立 Explore harness
--------------------

Explore 是独立的 memory 生成流程。它可以运行多次 attempt、 fresh planner
session 和本地 memory review，但不是 held-out success-rate 测量：

.. code-block:: bash

   export BEHAVIOR_ENV_GPU=2
   export BEHAVIOR_MODEL_GPU=7

   python -m robots.behavior.harness explore \
     --attempts 3 \
     --output-dir /path/to/behavior-explore \
     -- \
     --task-name picking_up_trash \
     --public-seed 0 \
     --model gpt-5.5 \
     --behavior-env-cuda-device "$BEHAVIOR_ENV_GPU" \
     --behavior-model-cuda-device "$BEHAVIOR_MODEL_GPU" \
     --dino-source-archive /path/to/dinov2-source.tar.gz \
     --dino-weights /path/to/dinov2_vits14_pretrain.pth \
     --behavior-memory-dir /path/to/reviewed-episode-catalog

Explore 产物可以进入已审查 recipe、task memory 和 episode memory，但必须把
candidate/development 证据与正式 Eval artifact 分开。

Dashboard
---------

标准 RPent Dashboard launcher 支持 BEHAVIOR：VLA 与 DINO 作为 shared
component 在 TaskRun 之间复用，每个 TaskRun 拥有 fresh env：

.. code-block:: bash

   hf download RLinf/RLinf-Pi05-BEHAVIOR-1K-PT50-CS32 \
     --local-dir ./checkpoints/RLinf-Pi05-BEHAVIOR-1K-PT50-CS32
   export PI05_CHECKPOINT_PATH=$PWD/checkpoints/RLinf-Pi05-BEHAVIOR-1K-PT50-CS32
   export BEHAVIOR_ENV_GPU=2
   export BEHAVIOR_MODEL_GPU=7

   rpent --robot behavior --dashboard \
     --model gpt-5.5 \
     --behavior-env-cuda-device "$BEHAVIOR_ENV_GPU" \
     --behavior-model-cuda-device "$BEHAVIOR_MODEL_GPU" \
     --dino-source-archive /path/to/dinov2-source.tar.gz \
     --dino-weights /path/to/dinov2_vits14_pretrain.pth

在页面里用以下命令启动 TaskRun：

.. code-block:: text

   /rpent-task turning_on_radio 1

底层 BEHAVIOR Dashboard module 也可以直接用于人工控制和调试：

.. code-block:: bash

   export BEHAVIOR_ENV_GPU=2
   export BEHAVIOR_MODEL_GPU=7

   python -m robots.behavior.dashboard \
     --task-name turning_on_radio \
     --public-seed 1 \
     --behavior-env-cuda-device "$BEHAVIOR_ENV_GPU" \
     --behavior-model-cuda-device "$BEHAVIOR_MODEL_GPU" \
     --dino-source-archive /path/to/dinov2-source.tar.gz \
     --dino-weights /path/to/dinov2_vits14_pretrain.pth

Dashboard 用于观察和引导 BEHAVIOR run。它不会改变官方成功定义，也不沿用
LIBERO 的任务成功定义。

env process 使用独立 GPU；VLA 与 DINO 按 LIBERO 的共享 VLA/SAM3 component
模式共用 model GPU。每个本地 CUDA child 仍只收到一个显式物理
``CUDA_VISIBLE_DEVICES`` 值。若三个 component 确实要使用同一物理 GPU，可用
``--cuda-device`` 作为两个 component-specific 参数的共同 fallback。

组件职责
--------

- **env_server**（``robots/behavior/env_server.py``）通过 pinned source
  checkout 持有官方 BEHAVIOR/OmniGibson 进程，并通过 RPent RPC 暴露 reset、
  observation、action、Dashboard control 和 raw success receipt。
- **vla_server**（``robots/behavior/vla_server.py``）持有 Pi0.5 BEHAVIOR
  checkpoint，并按 RPent runtime contract 返回 BEHAVIOR ``[T,23]`` action。
- **dino_server**（``robots/behavior/dino_server.py``）持有 DINOv2 图像
  embedding 服务，用于 episode-memory 检索。
- **toolkit**（``robots/behavior/toolkit.py``）只暴露公开 planner tools，并记录
  public observation、action trace 和 terminal receipt。

Planner 可调用工具
------------------

BEHAVIOR tools 按 task scope 暴露。当前 public surface 包括：

- VLA-backed action：``pi0_nav_pick(instruction, chunks)``。
- Observation 与 geometry：``observe(...)``、``pixel_to_world(...)``。
- Analytic motion 与 gripper action：``move_to(...)``、``move_both_to(...)``、
  ``rotate_wrist(...)``、``close(...)``、``open(...)``、``press(...)``、
  ``navigate_to(...)``。
- Safety 与 receipts：``get_prepared_motion_status(...)``、
  ``save_robot_state_checkpoint(...)``、``finish(status, summary)``。

若某个 runtime component 被刻意关闭，工具面会随之收窄；实际运行以 active
toolkit schema 为准。

VLA 与 DINO 组件
----------------

BEHAVIOR policy checkpoint 已发布到 Hugging Face：
`RLinf/RLinf-Pi05-BEHAVIOR-1K-PT50-CS32 <https://huggingface.co/RLinf/RLinf-Pi05-BEHAVIOR-1K-PT50-CS32>`_。
使用 ``hf download`` 下载该仓库，并将 ``PI05_CHECKPOINT_PATH`` 指向下载后的目录；
不要通过隐藏 task registry 静默替换成任务专用 checkpoint。

DINOv2 视觉检索使用经过审查的本地 DINOv2-S/14 部署，用于图像 embedding 和
episode-memory lookup。DINO source archive 与 weights 是运行时资产，不是
wheel data；它们的 digest 应保存在 resource binding 或单独的部署审计记录中。

Episode memory
--------------

BEHAVIOR memory 是运行时数据，可能包含 global task notes、已审查 recipe、
DINO 索引的 episode memory 和 run receipt。它应保存在 Python package 外部，
并且每次运行都要绑定到实际使用的 memory revision。

Receipt 与 raw success
----------------------

对每次 Eval 或 Explore，检查公开 tool record 与 ``terminal_receipt.json``。
成功结论必须由 official receipt 中的原始 ``info["done"]["success"]`` 证据
支撑；planner status 和本地 primitive completion 都不能替代它。

故障排查
--------

- ``ModuleNotFoundError: robots.behavior`` 表示 BEHAVIOR source plugin 没有在
  ``PYTHONPATH`` 中，或没有以 editable 方式安装。
- OmniGibson 或 Isaac 启动失败应按 upstream pinned install guide 修复，不要把
  仿真器包加入 ``behavior`` extra。
- 缺少 ``PI05_CHECKPOINT_PATH`` 或 digest 不匹配时，应在 VLA 执行前失败。更换
  checkpoint 后重新运行 ``python -m robots.behavior.selfcheck``。
- 视频或 frame 提取失败通常是 ImageIO ffmpeg backend 缺失；在当前环境重新安装
  ``behavior`` extra。
- 如果 run 有过程进展但没有 official success receipt，除非 raw trace 中存在
  ``info_done.success=true``，否则应归类为未成功。

已知 smoke-test 边界
--------------------

短 BEHAVIOR smoke run 只能证明 source checkout、仿真进程、RPC wiring、图像路径和
Pi0.5 call path 可以启动。它不是 held-out evaluation，不代表 benchmark success
rate；没有 raw BEHAVIOR success bit 和 receipt 时，不应报告为官方任务完成。
