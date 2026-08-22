RoboTwin
========

`RoboTwin <https://robotwin-platform.github.io/>`_ 是一个面向双臂机器人操作的
仿真基准，包含多种桌面操作任务和随机化场景。RPent 通过 RLinf 运行 RoboTwin，
并使用 LingBot-VLA 生成机器人动作。

安装
----

请使用已完成完整验证的 Python 3.11。创建虚拟环境并安装 RoboTwin
所需依赖：

.. code-block:: bash

   cd /path/to/RPent
   uv venv --python 3.11
   source .venv/bin/activate
   uv pip install -e ".[robotwin]"

用户不需要运行 RLinf 安装器，也不需要单独克隆 RoboTwin。

国内网络可使用 PyPI 镜像加速：\

.. code-block:: bash

   uv pip install -e ".[robotwin]" \
      --default-index https://mirrors.aliyun.com/pypi/simple \
      --index https://pypi.tuna.tsinghua.edu.cn/simple

.. note::

   ``.[robotwin]`` 使用 SAPIEN 3.0.0b1。其他版本可能改变仿真观测，导致模型
   效果下降。

下载仿真资源
------------

下载 RPent 支持的 RoboTwin 仿真资源，并设置资源目录：

.. code-block:: bash

   robotwin-download-assets --output ~/.robotwin/assets
   export ROBOTWIN_ASSETS_PATH=~/.robotwin/assets
   # 国内用户可以使用下面的命令
   # HF_ENDPOINT=https://hf-mirror.com robotwin-download-assets --output ~/.robotwin/assets

下载工具会先校验已有文件；如果目标目录中的 RoboTwin 资源已经完整，
则不会重复下载。

下载模型
--------

下载 LingBot 模型并设置模型目录：

.. code-block:: bash

   # 国内用户在可以加上 HF_ENDPOINT=https://hf-mirror.com
   hf download RLinf/LingBot-VLA-RoboTwin-EEF-ckpt1500 \
      --revision e727b46cd220b66981ea4d2fd9ba84adc189e2cc \
      --local-dir /path/to/LingBot-VLA-RoboTwin-EEF-ckpt1500
   export LINGBOT_MODEL_PATH=/path/to/LingBot-VLA-RoboTwin-EEF-ckpt1500

模型目录中已经包含 RoboTwin 的默认机器人配置。

运行任务
--------

激活虚拟环境后运行一个任务：

.. code-block:: bash

   # 国内用户在可以加上 HF_ENDPOINT=https://hf-mirror.com,
   # 因为下面的命令运行过程中会下载相关的memory数据
   rpent --env robotwin \
      --task-name beat_block_hammer \
      --seed 100000 \
      --planner codex \
      --model gpt-5.5

修改 ``--task-name`` 可以选择其他任务，修改 ``--seed`` 可以使用其他随机种子。
完整参数请运行 ``rpent --env robotwin --help`` 查看。

查看运行结果
------------

终端会显示服务启动信息、规划器输出和工具调用。默认情况下，运行结果保存在
``logs/<timestamp>_robotwin_<task-name>_s<seed>/``。排查或复核运行结果时，
可以先查看以下文件：

- ``run.log``：RPent 主进程日志。
- ``robotwin_env_server.log`` 和 ``lingbot_vla_server.log``：仿真环境与模型
  服务的启动和报错信息。
- ``transcript_*.json``：规划器对话和最终回复。

任务是否成功以最新工具结果中的 RoboTwin 原生
``TASK_ENV.eval_success`` 为准。``finish`` 只负责结束规划器循环，不会另外
定义一套成功条件。

添加 ``--dashboard`` 可以在浏览器中查看规划器输出以及头部和腕部相机画面。
Dashboard 启动后，访问地址会显示在终端中。

常用参数
--------

- ``--task-config``：选择 RoboTwin 环境配置，支持 ``demo_clean`` 和
  ``demo_randomized``，默认使用 ``demo_randomized``。
- ``--robotwin-assets-path``：覆盖 ``ROBOTWIN_ASSETS_PATH`` 指定的资源目录。
- ``--vla-model-path``：覆盖 ``LINGBOT_MODEL_PATH`` 指定的模型目录。
- ``--cuda-device``：让仿真环境和 VLA 使用同一张 GPU。
- ``--env-cuda-device`` 和 ``--vla-cuda-device``：让仿真环境和 VLA 使用不同
  GPU。这两个参数不能与 ``--cuda-device`` 同时使用。

规划器配置、外部服务和离线参考资料的说明分别见 :doc:`configure_planner`、
:doc:`advanced_deployment` 和 :doc:`../development/memory`。

每次运行前，RPent 会自动从公开数据集 ``RLinf/RPent-memory`` 同步可选的 RoboTwin
经验和任务参考。这些内容包含经过验证的操作方法，可以帮助规划器提高任务表现；
即使无法下载，任务仍可正常启动。
