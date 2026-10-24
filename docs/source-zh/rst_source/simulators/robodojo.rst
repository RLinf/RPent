RoboDojo
========

通过 RPent 工具控制 RoboDojo 双臂桌面场景。仿真器基于 Isaac Sim / IsaacLab，采用 ARX-X5 双机械臂平台。

.. note::

   解析式工具调用原生 cuRobo 位姿规划，并执行关节目标。规划器默认障碍物模型只有简化桌面，未包含全部物体或另一只机械臂。路径规划及抽样不能保证执行时无碰撞。组件检查和有界动作执行检查不能说明基准任务成功率。

   ``pi0_pick``、``stabilize`` 和 ``place_in_bin`` 不注册。常规启动仅加载 Env 和 SAM3，不需要动作策略权重；Flash 重放仍不可用。

概览
----

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 动作模型

      解析式位姿和夹爪 Primitive

   .. grid-item-card:: 规划器

      ``api``、``claude_code``、``codex``

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
     - 头部和左右腕部相机的 RGB-D 图像、相机标定，以及双臂关节状态、末端执行器位姿和夹爪开合状态。
   * - 动作
     - ``move_to``、``rotate_wrist``、``set_gripper`` 和 ``release``。``finish`` 不能覆盖原生任务判定。
   * - 奖励与成功判定
     - 任务是否成功由 RoboDojo 原生环境判定，结果记录在 ``result.json`` 的 ``success`` 字段中。
   * - 任务指令
     - 来自当前场景的 RoboDojo 任务描述，通过 ``view_env_state`` 读取。

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
     --include 'Assets/**' --local-dir /data/robodojo
   export ROBODOJO_ASSETS_ROOT=/data/robodojo

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

SAM3 配置
---------

SAM3 用于物体分割。先在 `facebook/sam3 模型页面 <https://huggingface.co/facebook/sam3>`_ 申请访问权限，\
再下载 ``sam3.pt`` 并设置路径：

.. code-block:: bash

   hf download facebook/sam3 sam3.pt --local-dir /data/sam3
   export SAM3_CHECKPOINT_PATH=/data/sam3/sam3.pt

也可以从 `ModelScope <https://modelscope.cn/models/facebook/sam3>`_ 下载同一文件，\
再将 ``SAM3_CHECKPOINT_PATH`` 指向本地文件。

运行一个任务
------------

先按 :doc:`../guides/configure_planner` 选择并配置要使用的 planner 和模型。以下以 Codex 为例，\
在运行任务的同一终端检查连接：

.. code-block:: bash

   rpent-check-llm --planner codex --model <planner-model> --json

将 ``<planner-model>`` 替换为所选模型名，配置好 SAM3 后运行瓶子入桶任务。此命令会让 Agent 控制双臂动作：

.. code-block:: bash

   # 临时处理：当前 RPent 会优先导入仓库旁的 rlinf，可能覆盖已安装的 RoboDojo 集成版。
   # 指向当前已激活环境的 site-packages，让子服务使用该环境安装的 RLinf。
   # 后续修正 RPent 的默认导入路径后，可移除此设置。
   export RPENT_RLINF_ROOT="$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"

   export ROBODOJO_PLACEMENT_SETTLE_STEPS=1000  # 推荐：让物体充分稳定；不设置时默认等待 20 步
   export OMNI_KIT_ACCEPT_EULA=YES  # 推荐用于首次非交互启动，提前接受 Kit 许可；已接受时可省略
   # Isaac Sim 启动需数十秒，首次运行还会编译 shader
   rpent --robot robodojo --task put_bottles_into_dustbin --layout 0 \
     --planner codex --model <planner-model> --reasoning-effort medium \
     --max-turns 40 --planner-timeout-s 1500 \
     --output-dir /path/to/run-output

查看结果
--------

终端会显示服务启动信息、planner 输出和工具调用。运行结束后，查看 ``--output-dir`` 指定目录中的文件：

- ``result.json``：环境的任务成功判定及运行错误状态。
- ``states.json``：工具命令、执行结果及同帧观测记录。
- 各相机逐步保存的 RGB、米制深度、标定和世界坐标三维图。
- ``videos/``：头部、左腕和右腕相机的视频。
- ``robodojo_env_server.log`` 和 ``sam3_server.log``：\
  本地启动服务的日志和报错信息。
- ``run.log`` 和 ``transcript_*.json``：主进程日志、planner 对话和错误。

先检查 ``environment_result_available=true`` 和\
``agent_error_present=false``，再看 ``success`` 是否为 ``true``。服务正常退出或 planner\
调用 ``finish``，均不能单独说明任务成功。

未指定输出目录时，结果保存在 ``logs/<timestamp>_robodojo_<task>_s<seed>_l<layout>/``。并发运行应使用不同目录。\
实时查看方式见 :doc:`../guides/dashboard`。

常用参数
--------

默认由 RPent 在当前 Python 环境中启动仿真和 SAM3 服务。仿真器安装在独立环境时，指定 ``--sim-python``。旧 Pi0.5 诊断辅助代码保留在源码中，但 CLI 不启动该服务，也不向规划器开放其工具。

使用 ``--task`` 选择任务。固定场景通过 ``--seed`` 选择官方布局组，通过 ``--layout`` 选择组内布局，两者默认均为 ``0``。例如，``--task stack_blocks --seed 0 --layout 2`` 使用第 0 组的第 2 个布局。需要随机生成场景时，添加 ``--random``，此时 ``--layout`` 不生效。

- ``--cuda-device``：为本地启动的服务选择 GPU；省略时沿用 ``CUDA_VISIBLE_DEVICES``。
- ``--env-endpoint`` 和 ``--sam3-endpoint``：连接已有服务。
- ``--source-root``：使用本地 RoboDojo 源码替代包内源码。

子服务继承当前终端的环境变量；CLI 不会自动读取 ``config/runtime.env``。\
完整参数可运行 ``rpent --robot robodojo --help`` 查看。

工具
----

规划器不能重置回合；探索模式和 Flash 重放不可用。

初始观测和运动反馈包含 ``episode_status.step`` （已用原生动作步数）、``step_limit`` （总步数上限）和 ``remaining_steps`` （剩余步数）。
这些计数对应观测保存时的状态；顶层 ``step`` 是观测记录编号。读取历史观测不会消耗动作步数。

``view_env_state`` 读取保存的图像和状态，``render`` 刷新观测。``sample_world_xyz(view, pixels, step=None, neighborhood=1)`` 查询 [行,列] 像素，默认取 3×3 邻域。``query_world_map`` 查询半开区间 [r0,c0,r1,c1]，默认最多返回 256 个点。view 使用 ``head``、``left_wrist`` 或 ``right_wrist``，必须与选点图像的视角及步号一致。查询读取保存的同帧深度，不暗中刷新；无效深度被排除，返回的是可见表面坐标，不是物体中心。

``segment`` 和 ``back_project`` 也接受 ``step``。分割返回质心 ``centroid_rc`` 和掩码内像素 ``point_rc``；凹形掩码的质心可能在物体外，反投影应优先使用 ``point_rc`` 及返回的 ``step_idx``。这两个工具保留相机名 ``cam_head``、``cam_left_wrist``、``cam_right_wrist``。

调用 ``segment(text_prompt=...)`` 时，描述可见的颜色、形状和位置。工具返回红色掩码叠加图；使用定位点前先检查叠加图，高分不代表选中了正确物体。分割失败时可修改描述，或在同一历史帧中手动选取可见像素查询。叠加图和几何结果保存在对应帧中。与 LIBERO 的接口不同，这里不接受点提示，也不返回掩码区域的世界坐标中位数；选定掩码内像素后需调用 ``back_project``。

``move_to(arm, xyz, quat=None, gripper=None, substeps=25)`` 使用世界坐标、米及 wxyz 四元数；省略姿态或夹爪参数则保持对应状态／目标。``substeps`` 限制路径抽样数，0 保留完整路径，1 仅取最后路点，不代表速度或物理步数。``rotate_wrist`` 绕世界 Z 轴相对旋转，单位为度。``final_dist_m`` 是实测位置误差，不是到达保证。检查 ``completed``、``executed_steps`` 和 ``stop_reason``；超时的动作可能已在服务端执行，不会自动重试。

``set_gripper(arm, val, steps=10)`` 使用 0=闭、1=开，``release`` 默认开到 1。中间值是连续控制目标，不保证对应精确物理宽度；反馈读取实际关节，而非命令字段。仅当实测值因物理超调越界时，将插值起点限制在合法范围，用户非法输入仍被拒绝。其他关节目标保持不变，每次下发目标还会经过底层插值。

.. _robodojo-local-nvidia-assets:

NVIDIA 场景素材的本地准备（可选）
------------------------------------------------------------

以下两类用户需要预先下载 NVIDIA 场景素材：

- 需要在无法联网的机器上运行仿真的用户。
- 运行时无法稳定访问 NVIDIA 资源地址，日志中相关文件反复下载失败的用户。

RoboDojo 场景还会引用 NVIDIA 提供的三维模型、材质和贴图。这些文件与前面下载的 RoboDojo 资源分开提供，默认在加载场景时通过网络获取。\
场景能正常加载时，可跳过本节，无需下载本地资产包，也不用设置 ``ROBODOJO_USD_ASSET_PREFIX``。

以下步骤用于准备 ``Assets/Isaac/5.0`` 路径对应的本地资源。完整资产包的三个分卷合计约 150 GB，并非单个 RoboDojo 场景所需的下载量；\
合并和解压还需额外空间。离线机器可先在能联网的机器上下载，再将分卷复制到 ``/data/nvidia``。

.. code-block:: bash

   mkdir -p /data/nvidia

   # 下载三个分卷，已有下载可断点续传
   wget -c -P /data/nvidia \
     https://download.isaacsim.omniverse.nvidia.com/isaac-sim-assets-complete-5.0.0.zip.001 \
     https://download.isaacsim.omniverse.nvidia.com/isaac-sim-assets-complete-5.0.0.zip.002 \
     https://download.isaacsim.omniverse.nvidia.com/isaac-sim-assets-complete-5.0.0.zip.003

   # 合并并解压，保留 Assets/Isaac/5.0 目录结构
   cat /data/nvidia/isaac-sim-assets-complete-5.0.0.zip.001 \
       /data/nvidia/isaac-sim-assets-complete-5.0.0.zip.002 \
       /data/nvidia/isaac-sim-assets-complete-5.0.0.zip.003 \
     > /data/nvidia/isaac-sim-assets-complete-5.0.0.zip
   unzip /data/nvidia/isaac-sim-assets-complete-5.0.0.zip -d /data/nvidia

   test -d /data/nvidia/Assets/Isaac/5.0/Isaac
   test -d /data/nvidia/Assets/Isaac/5.0/NVIDIA

在运行 RPent 的终端中设置以下变量，再运行任务：

.. code-block:: bash

   export ROBODOJO_USD_ASSET_PREFIX=/data/nvidia

这会将 RoboDojo 匹配的 5.0 在线资源地址改为本地路径。它并不代表所有 Isaac Sim 资源均可离线使用；\
若仍有其他在线资源加载失败（例如 ``Assets/Isaac/5.1`` 路径），需另外准备对应版本的资产并配置本地加载，\
见 `NVIDIA 本地资产说明 <https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_faq.html#isaac-sim-setup-assets-content-pack>`_。
