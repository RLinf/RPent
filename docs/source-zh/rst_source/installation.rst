安装
====

RPent 可以通过一条 ``pip install`` 命令完成安装。不同的可选依赖组合会从
PyPI 安装对应的 RLinf 运行时、openpi、LIBERO 仿真器和 SAM 3.0 依赖。

准备工作
--------

- Linux + NVIDIA GPU (LIBERO 通过 EGL 渲染)。
- 与显卡匹配的 CUDA 12.x 驱动。
- Python 3.10–3.11。
- ``git``、``bash``、以及能编译 MuJoCo / robosuite 的 C 工具链。

此外，还需要：

- 至少一个 LLM 提供商的 API key，例如 Anthropic、OpenAI 或提供 OpenAI
  兼容接口的服务商，供 planner 调用。
- 一个 VLA checkpoint。使用 LIBERO 和 Pi0.5 时，推荐使用
  `Hugging Face: RLinf-Pi05-LIBERO-130-fullshot-SFT
  <https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT>`_。
- 本地 SAM 3.0 ``sam3.pt`` 文件, 可从 `Hugging Face: facebook/sam3
  <https://huggingface.co/facebook/sam3>`_ 或 `ModelScope: facebook/sam3
  <https://modelscope.cn/models/facebook/sam3>`_ 下载。

1. 用 pip 安装 RPent
--------------------

先克隆 RPent，其中包含 CLI 和运行配置，然后根据需要选择依赖组合：

.. code-block:: bash

   git clone https://github.com/RLinf/RPent rpent && cd rpent
   pip install -e ".[full]"

``.[full]`` 是默认的端到端依赖组合，包括 openpi Pi0.5 VLA、
LIBERO-PRO 仿真器、SAM 3.0 和 RLinf 运行时。

可选的依赖组合：

.. list-table::
   :header-rows: 1

   * - Extra
     - 安装内容
   * - ``.[full]``
     - ``rlinf`` + ``openpi`` + ``libero-pro`` + ``sam3`` —— 默认运行组合
   * - ``.[libero-pro]``
     - 仅基础 LIBERO + LIBERO-PRO 仿真器
   * - ``.[libero-plus]``
     - 基础 LIBERO + LIBERO-plus 仿真器
   * - ``.[libero]``
     - 仅基础 LIBERO
   * - ``.[openpi]``
     - 仅 openpi VLA
   * - ``.[rlinf]``
     - 仅 RLinf 运行时
   * - ``.[sam3]``
     - 仅 SAM 3.0

2. 下载仿真资源（LIBERO）
-------------------------

通过 pip 安装的 Python 包不包含运行 LIBERO 所需的大型资源文件。请根据上一步安装的
依赖组合，从以下命令中选择一条。使用推荐的 ``.[full]`` 时运行第二条：

.. code-block:: bash

   libero-download-assets --skip-existing      # .[libero]
   liberopro-download-assets --skip-existing   # .[libero-pro] / .[full]
   liberoplus-download-assets --skip-existing  # .[libero-plus]

这些资源通常只需下载一次；``--skip-existing`` 会跳过已经存在的文件。

.. tip::

   如果访问 Hugging Face 较慢，可以通过设置 ``HF_ENDPOINT`` 使用镜像下载：

   .. code-block:: bash

      HF_ENDPOINT=https://hf-mirror.com liberopro-download-assets --skip-existing

3. (可选) 真实机器人依赖
------------------------

Franka 与 SO-101 的支持正在逐步接入; 每个机器人的 driver 会以一个包的
形式放在 ``robots/<name>/`` 下, 并附带 ``README.md`` 说明其 SDK / 固件
要求。当前进度参见 :doc:`usage/franka` 与 :doc:`usage/so101`。

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
