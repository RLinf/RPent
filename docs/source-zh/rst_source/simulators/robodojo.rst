RoboDojo
========

使用 RPent 在 RoboDojo 中运行双臂桌面操作任务。仿真器基于 Isaac Sim / IsaacLab，采用 ARX-X5 双机械臂平台，默认使用 Pi0.5 生成动作。\
该接入仍为实验性，目前尚无完整任务套件的成功率结果。

概览
----

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 动作模型

      Pi0.5

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

目录还单列了一个 DLC 任务。通过 ``--task`` 选择已安装运行时中的任务，``--layout`` 选择场景布局。\
例如，``stack_bowls_random`` 属于泛化任务，``fill_pen_holder`` 和 ``put_bottles_into_dustbin``\
属于长程任务。这三个名称只是示例，并非完整任务列表。

观测与动作
~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - 项目
     - 说明
   * - 观测
     - 头部和左右腕部相机的 RGB-D 图像、相机标定，以及双臂关节状态、末端执行器位姿和夹爪开合状态。Pi0.5 使用三路 RGB 图像、机器人状态和任务文字。
   * - 动作
     - planner 调用 ``pi0_pick`` 或 ``move_to``、``set_gripper`` 等原语。Pi0.5 输出双臂关节和夹爪动作，共 14 维。
   * - 奖励与成功判定
     - 任务是否成功由 RoboDojo 原生环境判定，结果记录在 ``result.json`` 的 ``success`` 字段中。
   * - 任务指令
     - 来自当前场景的 RoboDojo 任务描述。省略 ``pi0_pick`` 的 ``prompt`` 参数时，使用该指令。

安装与资源准备
--------------

NVIDIA 驱动版本建议参考 `Isaac Sim 5.1 官方系统要求 <https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html>`_。\
宿主机还需具备构建 cuRobo 所需的 CUDA toolkit/NVCC 和编译工具链。从 RPent 根目录创建独立环境并安装依赖：

.. code-block:: bash

   uv venv --python 3.11 .venv-robodojo
   source .venv-robodojo/bin/activate
   uv pip install -e ".[robodojo]" --extra-index-url https://pypi.nvidia.com \
     --override requirements/robodojo-override.txt

.. note::

   Blackwell GPU 需要 CUDA 12.8 版 PyTorch。本项目固定的 PyTorch 2.7.0 也提供 CUDA 12.6 构建，\
   但该构建不支持 Blackwell。使用这类 GPU 时，首次安装及后续重新安装都使用以下命令，\
   通过 ``--torch-backend=cu128`` 选择 CUDA 12.8 构建：

   .. code-block:: bash

      uv pip install -e ".[robodojo]" --extra-index-url https://pypi.nvidia.com \
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

VLA 配置
--------

通过 ``--policy-backend`` 选择 Pi0.5 后端，默认使用 ``rlinf``，也可选择另行准备的 ``xpolicylab``。

RLinf（默认）
~~~~~~~~~~~~~~~~

``--policy-backend rlinf`` 使用安装步骤中提供的 RLinf 运行时，\
通过 ``PI05_CHECKPOINT_PATH`` 指定 PyTorch 权重和配套统计量。

发布权重为 JAX 格式，需要转换成 PyTorch 格式。本例只需下载 ``RoboDojo-sim-arx_x5-joint-0/59999`` 中的完整 ``params/`` 目录和配套的 ``norm_stats.json``，\
合计约 12.4 GB，无需下载训练状态或其他模型。转换后的权重约 7.23 GB，请另为转换过程和缓存预留空间。

.. code-block:: bash

   hf download RoboDojo-Benchmark/RoboDojo --repo-type dataset \
     --include \
       'ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0/59999/params/**' \
       'ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0/59999/assets/arx_x5_sim/norm_stats.json' \
     --local-dir /data/robodojo/pi05_jax
   export PI05_JAX_CHECKPOINT=/data/robodojo/pi05_jax/ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0/59999

.. note::

   建议将 ``--local-dir`` 的目录名设为小写 ``pi05_`` 开头，例如上面的 ``pi05_jax``。\
   当前转换器要求输入路径中包含小写 ``pi05``；发布目录名 ``Pi_05`` 不满足这一要求。

   如果已下载到不含小写 ``pi05`` 的路径，无需重新下载。为包含 ``59999/`` 的模型目录创建一个以 ``pi05_`` 开头的软链接，\
   再将 ``PI05_JAX_CHECKPOINT`` 设为链接下的 ``59999``。例如，原下载目录为 ``/data/robodojo`` 时：

   .. code-block:: bash

      ln -s /data/robodojo/ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0 \
        /data/robodojo/pi05_robodojo_arx_x5
      export PI05_JAX_CHECKPOINT=/data/robodojo/pi05_robodojo_arx_x5/59999

在同一环境中转换权重，并设置转换后的模型目录：

.. code-block:: bash

   export PI05_CHECKPOINT_PATH=/data/robodojo/pi05_robodojo_arx_x5_torch

   python -m rlinf.utils.ckpt_convertor.convert_openpi_jax_to_python \
     --checkpoint-dir "$PI05_JAX_CHECKPOINT" \
     --config-name pi05_aloha \
     --output-path "$PI05_CHECKPOINT_PATH"

``--checkpoint-dir`` 指向包含 ``params/`` 的目录。

添加 ``--inspect_only`` 可只查看参数信息，不执行转换。

模型还需要发布权重配套的归一化统计文件 ``norm_stats.json``。转换器不会将该文件写入\
``$PI05_CHECKPOINT_PATH/RoboDojo/arx_x5/norm_stats.json``，需按下方命令复制：

.. code-block:: bash

   mkdir -p "$PI05_CHECKPOINT_PATH/RoboDojo/arx_x5"
   cp "$PI05_JAX_CHECKPOINT/assets/arx_x5_sim/norm_stats.json" \
     "$PI05_CHECKPOINT_PATH/RoboDojo/arx_x5/norm_stats.json"

XPolicyLab（可选）
~~~~~~~~~~~~~~~~~~~~

推荐使用前面的 RLinf 方式，默认后端为 ``--policy-backend rlinf``，无需额外安装 XPolicyLab。\
RPent 也提供 XPolicyLab 后端，已有该环境或需要使用其 Pi0.5 实现时，可选择 ``--policy-backend xpolicylab``。

XPolicyLab 是另一套加载和运行 Pi0.5 的软件。使用前需单独克隆源码，\
并按其 `Pi0.5 安装说明 <https://github.com/XPolicyLab/XPolicyLab/blob/432f82b1758c5b1202e42a3dfe014546dbc50871/policy/Pi_05/README.md#installation>`_ 准备 Python 环境。\
可复用前面下载的原始 JAX 权重和统计量，但需按 XPolicyLab 的要求配置存放位置。
按下文完成 SAM3 和 planner 配置后，使用单独的 XPolicyLab 源码与 Python 环境运行任务：

.. code-block:: bash

   rpent --robot robodojo --task put_bottles_into_dustbin --layout 0 \
     --planner codex --model <planner-model> \
     --policy-backend xpolicylab --xpolicylab-root /path/to/XPolicyLab \
     --pi05-python /path/to/xpolicylab-env/bin/python \
     --output-dir /path/to/run-output

``--xpolicylab-root`` 指向单独克隆的 XPolicyLab 源码目录。如果 XPolicyLab 的 Pi0.5 推理所需\
Python 包安装在另一个环境中，用 ``--pi05-python`` 指向该环境的 Python 解释器；否则省略此参数。

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

将 ``<planner-model>`` 替换为所选模型名，运行瓶子入桶任务：

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
- ``states.json``：动作记录及动作后的观测。
- ``videos/``：头部、左腕和右腕相机的视频。
- ``robodojo_env_server.log``、``robodojo_vla_server.log``、``sam3_server.log``：\
  各服务的启动和报错信息。
- ``run.log`` 和 ``transcript_*.json``：主进程日志、planner 对话和错误。

先检查 ``environment_result_available=true`` 和\
``agent_error_present=false``，再看 ``success`` 是否为 ``true``。服务正常退出或 planner\
调用 ``finish``，均不能单独说明任务成功。

未指定输出目录时，结果保存在 ``logs/<timestamp>_robodojo_<task>_l<layout>/``。并发运行应使用不同目录。\
实时查看方式见 :doc:`../guides/dashboard`。

常用参数
--------

默认由 RPent 在当前 Python 环境中启动仿真、策略和 SAM3 服务。只有将它们分别安装到不同环境时，\
才需要指定 ``--sim-python`` 和 ``--pi05-python``。

- ``--task`` 和 ``--layout``：选择任务与固定布局；传入 ``--random`` 时忽略 ``--layout``，\
  每次重置重新采样随机场景。
- ``--cuda-device``：为本地启动的服务选择 GPU；省略时沿用 ``CUDA_VISIBLE_DEVICES``。
- ``--env-endpoint``、``--vla-endpoint``、``--sam3-endpoint``：连接已有服务。\
  VLA 服务需与 ``--policy-backend`` 匹配。
- ``--source-root``：使用本地 RoboDojo 源码替代包内源码。

子服务继承当前终端的环境变量；CLI 不会自动读取 ``config/runtime.env``。\
完整参数可运行 ``rpent --robot robodojo --help`` 查看。

任务记忆与 Flash 重放
----------------------------

RoboDojo 暂不支持 Flash Mode 和 ``--explore``。

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
