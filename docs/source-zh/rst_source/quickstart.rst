快速开始
============

本页从安装开始，带你运行一个 LIBERO-PRO 任务，并检查执行结果。示例使用 Claude Code 规划器、Pi0.5 动作模型和 SAM3 视觉分割。

若要使用其他平台，直接进入 :doc:`usage/robocasa`、:doc:`usage/robotwin` 或 :doc:`真机部署 <usage/real_robots>`，按该平台的要求安装。

准备工作
------------

- Linux、NVIDIA GPU，以及兼容 CUDA 12 的驱动。
- ``git``、``bash`` 和 C/C++ 编译工具。
- `uv <https://docs.astral.sh/uv/getting-started/installation/>`_，用于创建独立的 Python 环境。以下命令使用 Python 3.11。
- 可访问示例模型的 Anthropic API Key，以及下载仿真资源和模型权重的网络。模型调用会产生 API 费用。

1. 安装 RPent
---------------

创建独立的 Python 环境，并安装 LIBERO-PRO 所需依赖。

.. code-block:: bash

   git clone https://github.com/RLinf/RPent.git
   cd RPent
   uv venv --python 3.11
   source .venv/bin/activate
   uv pip install -e ".[libero-pro]"

后续命令都在这个终端和 RPent 仓库目录中执行。重新打开终端时，先激活 ``.venv``，再设置下文的模型路径与 API Key。

2. 下载仿真资源和模型
------------------------------

先下载 LIBERO-PRO 场景资源；``--skip-existing`` 可复用已下载的文件：

.. code-block:: bash

   liberopro-download-assets --skip-existing

再下载 Pi0.5 和 SAM3，设置模型路径：

.. code-block:: bash

   uv pip install "huggingface_hub>=0.34,<1.0" modelscope
   hf download RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT \
     --exclude optimizer.pt \
     --local-dir ./checkpoints/RLinf-Pi05-LIBERO-130-fullshot-SFT
   modelscope download --model facebook/sam3 sam3.pt \
     --local_dir ./checkpoints/sam3

   export PI05_CHECKPOINT_PATH="$PWD/checkpoints/RLinf-Pi05-LIBERO-130-fullshot-SFT"
   export SAM3_CHECKPOINT_PATH="$PWD/checkpoints/sam3/sam3.pt"

Pi0.5 用于执行机器人动作，SAM3 用于定位图像中的物体。SAM3 也可从 Hugging Face 下载，授权和替代下载方法见 :doc:`usage/libero`。

3. 配置规划器并检查连接
---------------------------------

把 ``YOUR_API_KEY`` 替换为自己的 Key：

.. code-block:: bash

   export ANTHROPIC_API_KEY="YOUR_API_KEY"
   rpent-check-llm --planner claude_code --model claude-opus-4-8

使用 Anthropic 官方服务时无需设置 ``ANTHROPIC_BASE_URL``。自定义服务地址和其他规划器的配置见 :doc:`usage/configure_planner`。连接检查通过后再启动任务；它只验证模型服务的认证和连通性。

4. 运行第一个任务
------------------------

运行 ``libero_object_swap`` 任务集中的任务 ``2``，场景随机种子为 ``0``：

.. code-block:: bash

   rpent --robot libero --libero-type pro \
     --suite libero_object_swap --task 2 --seed 0 \
     --planner claude_code --model claude-opus-4-8 \
     --output-dir ./logs/first-libero-pro

终端会依次显示环境、VLA 和 SAM3 服务的启动状态，随后输出规划器对话与工具调用。结束后查看 ``logs/first-libero-pro/``。再次运行时，请换一个输出目录以保留前一次记录。

5. 查看结果
---------------

- ``episode.mp4``：回看机器人操作过程。
- ``transcript_*.json``：查看规划器对话、工具调用和结束状态。
- ``run.log``：查看运行日志和错误。

LIBERO 的任务成功以最终环境状态的顶层 ``terminated`` 为准，可通过 ``view_env_state(step=-1)`` 的工具结果查看。规划器在 ``finish`` 中声明成功，不等于环境判定成功。

默认目录命名、动作序列与逐步观测文件的说明见 :ref:`run-output-files`。

若要实时观看相机和动作记录，按 :doc:`usage/dashboard` 启动 Dashboard。更多任务、探索模式和实验复现见 :doc:`usage/libero`。

遇到问题时
---------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - 现象
     - 处理方法
   * - 模型服务连接失败
     - 检查 Key、模型访问权限和服务地址，重新运行 ``rpent-check-llm``。
   * - 资源下载失败
     - 重新运行下载命令。访问 Hugging Face 较慢时，可为相应命令设置 ``HF_ENDPOINT=https://hf-mirror.com``。
   * - 环境或模型启动失败
     - 先检查两个 checkpoint 路径，再查看 ``env_server.log``、``vla_server.log`` 或 ``sam3_server.log``；显存不足时检查其他 GPU 进程。
   * - 任务运行但未成功
     - 回看视频和最后的环境状态，区分操作失败与服务错误。单次任务结果不代表完整基准成绩。
