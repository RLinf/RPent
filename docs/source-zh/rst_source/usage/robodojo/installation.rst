RoboDojo 后端安装
=================

RPent/SAM3、RoboDojo 的 Isaac Sim 环境和 Pi_05 应使用独立 Python 环境。
请遵循 RoboDojo 与 XPolicyLab 官方安装说明，选择兼容的仿真器、CUDA 和策略依赖。
GPU 运行需要相应资产与 checkpoint；RPent 不会自动下载这些文件。

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

在 RPent 环境中安装：

.. code-block:: bash

   uv pip install -e ".[sam3]"

通过 ``SAM3_CHECKPOINT_PATH`` 配置 SAM3 checkpoint，显式传入 RoboDojo
源码目录和 Python 可执行文件：

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
直接启动 ``vla_server.py`` 时，需将 ``ROBODOJO_PI05_POLICY_ROOT`` 设为
``XPolicyLab/policy/Pi_05``。
