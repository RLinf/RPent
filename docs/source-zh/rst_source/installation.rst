安装
====

RPent 可以通过一条 ``pip install`` 命令完成安装，并提供多种可选依赖组合。

准备工作
--------

- Linux + NVIDIA GPU。
- 与 CUDA 12.x 兼容的 NVIDIA 驱动。
- Python 3.10–3.12。
- ``git``、``bash``、以及可用的 C 工具链。

此外，还需要：

- 至少一个 LLM 提供商的 API key，例如 Anthropic、OpenAI 或提供 OpenAI
  兼容接口的服务商，供 planner 调用。

1. 用 pip 安装 RPent
--------------------

先克隆 RPent，其中包含 CLI 和运行配置，然后根据需要选择依赖组合：

.. code-block:: bash

   git clone https://github.com/RLinf/RPent rpent && cd rpent
   # 默认推荐：
   pip install -e ".[libero-pro]"  # LIBERO-PRO

.. tip::

   ``openai-codex-cli-bin`` 目前在清华 TUNA PyPI 镜像中不可用。
   如果从默认 PyPI 安装较慢，可以使用阿里云 PyPI 镜像：

   .. code-block:: bash

      pip install -i https://mirrors.aliyun.com/pypi/simple openai-codex-cli-bin

如需使用其他环境配置，可选择：

.. code-block:: bash

   pip install -e ".[robocasa]"    # RoboCasa
   pip install -e ".[robotwin]"    # RoboTwin

``.[libero-pro]`` 是默认推荐的依赖组合。

可选的依赖组合：

.. list-table::
   :header-rows: 1

   * - Extra
     - 安装内容
   * - ``.[libero]``
     - 标准 LIBERO + openpi Pi0.5 VLA + SAM 3.0 + RLinf 运行时
   * - ``.[libero-pro]``
     - LIBERO-PRO + openpi Pi0.5 VLA + SAM 3.0 + RLinf 运行时
   * - ``.[libero-plus]``
     - LIBERO-plus + openpi Pi0.5 VLA + SAM 3.0 + RLinf 运行时
   * - ``.[robocasa]``
     - RoboCasa365 仿真器 + RLDX-1 VLA，详见 :doc:`usage/robocasa`
   * - ``.[robotwin]``
     - RoboTwin 仿真环境和 LingBot 推理所需依赖，详见 :doc:`usage/robotwin`
   * - ``.[rlinf]``
     - 仅 RLinf 运行时
   * - ``.[sam3]``
     - 仅 SAM 3.0

2. 下载运行 LIBERO 所需的仿真资源
------------------------------------------------

通过 pip 安装的 Python 包不包含运行 LIBERO 所需的大型资源文件。请根据上一步安装的
依赖组合，从以下命令中选择一条：

.. code-block:: bash

   libero-download-assets --skip-existing      # .[libero]
   liberopro-download-assets --skip-existing   # .[libero-pro]
   liberoplus-download-assets --skip-existing  # .[libero-plus]

这些资源通常只需下载一次；``--skip-existing`` 会跳过已经存在的文件。

.. tip::

   如果访问 Hugging Face 较慢，可以通过设置 ``HF_ENDPOINT`` 使用镜像下载：

   .. code-block:: bash

      HF_ENDPOINT=https://hf-mirror.com liberopro-download-assets --skip-existing

3. (可选) 真实机器人依赖
------------------------

Franka 与 SO-101 的支持正在逐步接入; 每个机器人的 robot 包未来会以一个
包的形式放在 ``robots/<name>/`` 下, 并附带 ``README.md`` 说明其 SDK /
固件要求。当前进度参见 :doc:`usage/franka` 与 :doc:`usage/so101`。

检查是否安装成功
----------------

最直接的检查方法是完整运行一个 LIBERO 任务，具体步骤见
:doc:`quickstart`。任务成功运行，说明 ``env_server``、``vla_server``、
``sam3_server`` 和 agent 均能正常工作。

如果出错:

- env server 的日志在 ``<output_dir>/env_server.log``。
- VLA server 的日志在 ``<output_dir>/vla_server.log``。
- SAM3 server 的日志在 ``<output_dir>/sam3_server.log``。
- agent 的运行日志在 ``<output_dir>/run.log``。

这四份日志都保存在本次运行的输出目录中，排查失败任务时无需再从其他位置
收集日志。
