RoboDojo 后端安装
=================

本页只说明 RPent 集成在上游仓库之上新增的部分。仿真器、CUDA 与策略依赖请遵循
RoboDojo 与 XPolicyLab 官方说明；GPU 运行需要相应资产与 checkpoint，RPent
不会自动下载这些文件。

Python 环境
-----------

该后端会驱动三个解释器，三者必须彼此独立。Isaac Sim 固定了
``websockets==12.0``、``numpy==1.26.0``、``packaging==23.0``、
``filelock==3.13.1`` 与 ``typing_extensions==4.12.2``；而 RPent 环境使用更新的
``websockets`` 和自己的 ``torch`` 构建，Pi_05 环境运行 JAX 与 ``openpi``。
把它们装进同一个解释器会破坏 Isaac Sim 的版本约束。

**一、RPent。** 按仓库说明安装，再补上本后端需要的感知扩展：

.. code-block:: bash

   uv pip install -e ".[sam3]"

**二、RoboDojo 仿真器。** 使用上游安装脚本，它会构建 Isaac Sim 环境与内置的
CuRobo；这是被支持的路径，RPent 不重复实现：

.. code-block:: bash

   cd /path/to/RoboDojo
   bash scripts/install.sh

本后端验证过的组合是 Python 3.11 配 ``isaacsim 5.1.0.0``、
``torch 2.7.0+cu128``、``numpy 1.26.0``、``websockets 12.0``、
``viser 0.1.34``、``tyro 0.9.0`` 与 ``warp-lang 1.11.0``，CuRobo 来自
``third_party/curobo``。把该环境的解释器作为 ``--sim-python`` 传入，不要把
RPent 或 Pi_05 的包装进它。

**三、Pi_05 策略。** 构建 XPolicyLab 部署配置指定的 uv 环境
（``policy_uv_env_path: openpi``）：

.. code-block:: bash

   cd /path/to/RoboDojo/XPolicyLab/policy/Pi_05
   bash install.sh

该脚本需要 ``uv``，会生成 ``openpi/.venv``。RoboDojo 的启动脚本会激活这个环境，
并需要 conda 以及一个能 import YAML 的解释器；默认解释器不满足时请设置
``ROBODOJO_CONDA_ROOT``。把对应的解释器作为 ``--pi05-python`` 传入。RPent
不会被安装进该环境：CLI 会用 RPent 仓库根目录、``--source-root`` 和
``--xpolicylab-root`` 为每个子服务拼出 ``PYTHONPATH``，因此策略环境里不需要存在
RPent 包。

源码与资产
----------

克隆官方仓库及子模块前，先安装 Git LFS：

.. code-block:: bash

   git lfs install
   git clone --recurse-submodules https://github.com/RoboDojo-Benchmark/RoboDojo.git
   cd RoboDojo
   git lfs pull
   git submodule foreach --recursive 'git lfs pull'
   git lfs fsck

对于官方 RoboDojo release 链接的资产和 checkpoint 仓库，同样依次执行克隆、
``git lfs pull`` 和 ``git lfs fsck``，并按该 release 的说明放置文件。
启动仿真器前，确认所需文件是实际数据而非 LFS 指针文本。
如果 release 的 LFS 属性不完整，应向数据集发布方核实；RPent 不提供自定义物化器。

RPent 配置
----------

默认的 ``--policy-backend rlinf`` 需要策略解释器提供
``pi05_robodojo_arx_x5`` 配置及其 openpi 依赖；官方 main 尚未包含该配置。通过 ``PI05_CHECKPOINT_PATH`` 指定兼容的 RLinf checkpoint，并以
``--pi05-python`` 传入该解释器。使用上文的 XPolicyLab 环境时，选择
``--policy-backend xpolicylab``。

通过 ``SAM3_CHECKPOINT_PATH`` 配置 SAM3 checkpoint，并导出摆放稳定步数。
默认值会让物体在 official 模式下不稳定；该变量由 RoboDojo 源码读取，而非
RPent，CLI 会把它传给启动的子服务：

.. code-block:: bash

   export ROBODOJO_PLACEMENT_SETTLE_STEPS=1000

显式传入 RoboDojo 源码目录和 Python 可执行文件：

.. code-block:: bash

   rpent --robot robodojo --task put_bottles_into_dustbin --layout 0 \
     --source-root /path/to/RoboDojo \
     --sim-python /path/to/sim-env/bin/python \
     --pi05-python /path/to/pi05-env/bin/python

``--xpolicylab-root`` 默认使用 ``SOURCE_ROOT/XPolicyLab``；独立克隆时请指定。
两个 Python 参数默认使用当前解释器，因此独立运行时需要显式指定可执行文件路径。
CLI 构造子进程导入路径，不读取工作区的 ``config/runtime.env``，也不修改父进程环境。
子进程仍会继承已有 shell 环境变量。

通过 ``--env-endpoint``、``--vla-endpoint`` 和 ``--sam3-endpoint`` 可连接已有服务。
连接已有服务时，该组件不需要本地源码或 Python 路径。
CLI 默认启动共享的 ``rpent.robots.components.pi05_vla_server --embodiment robodojo``。
选择 ``--policy-backend xpolicylab`` 时，改为启动 ``xpolicylab_vla_server``，
并通过 ``--policy-root`` 指定 ``XPolicyLab/policy/Pi_05``。
连接已有 VLA 服务时也应选择匹配的后端。切换后端不会转换 checkpoint；
RLinf 客户端会将原生观测编码为 openpi wire 格式。

每个自有服务的日志与输出都落在本次运行的输出目录：CLI 以 ``--save-dir``
传给环境服务、以 ``--output-dir`` 传给可选的 XPolicyLab 策略入口，因此并发运行不会互相干扰。

验证安装
--------

跑一次有界的开发模式 episode，确认规划器接管之前各服务已就绪：

.. code-block:: bash

   export ROBODOJO_PLACEMENT_SETTLE_STEPS=1000
   rpent --robot robodojo --task put_bottles_into_dustbin --layout 0 \
     --planner codex --model <planner-model> --max-turns 1 \
     --source-root /path/to/RoboDojo \
     --sim-python /path/to/sim-env/bin/python \
     --pi05-python /path/to/pi05-env/bin/python \
     --output-dir /path/to/run-output

预期现象：

* ``/path/to/run-output`` 下出现 ``robodojo_env_server.log``、
  ``sam3_server.log``、``robodojo_vla_server.log``，XPolicyLab 策略服务启动后还会出现
  ``vla_server.log``。
* 环境服务报告 ready，第一条观测包含 ``cam_head``、``cam_left_wrist`` 与
  ``cam_right_wrist`` 的内参、外参，以及关节与夹爪状态。
* 本次运行在 ``/path/to/run-output/videos`` 下为每路相机写一个 MP4。
* 退出后没有遗留的自有子进程，GPU 回到空闲。

启动阶段某个服务退出是最常见的失败形式，先去运行输出目录读它的日志。
Isaac Sim 启动需要数十秒，首次运行还要编译 shader。
