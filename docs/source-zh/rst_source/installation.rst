安装
====

RPent 可以通过一条 ``pip install`` 命令完成安装。不同的可选依赖组合会从
PyPI 安装对应的 openpi 和 LIBERO 仿真器包。

先决条件
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

1. 用 pip 安装 RPent
--------------------

先克隆 RPent，其中包含 CLI 和运行配置，然后根据需要选择依赖组合：

.. code-block:: bash

   git clone https://github.com/RLinf/RPent rpent && cd rpent
   pip install -e ".[full]"

``.[full]`` 是默认的端到端依赖组合，包括 openpi Pi0.5 VLA、
LIBERO-PRO 仿真器和 RLinf 运行时。

可选的依赖组合：

.. list-table::
   :header-rows: 1

   * - Extra
     - 安装内容
   * - ``.[full]``
     - ``rlinf`` + ``openpi`` + ``libero-pro`` —— 默认运行组合
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

2. 下载仿真资产
---------------

PyPI 安装包不包含体积较大的仿真资产。安装完成后，需要运行一次对应的下载命令：

.. code-block:: bash

   libero-download-assets --skip-existing      # 基础 LIBERO
   liberopro-download-assets --skip-existing   # LIBERO-PRO —— .[libero-pro] / .[full]
   liberoplus-download-assets --skip-existing  # LIBERO-plus —— .[libero-plus]

.. tip::

   如果访问 Hugging Face 较慢，可以通过设置 ``HF_ENDPOINT`` 使用镜像下载：

   .. code-block:: bash

      HF_ENDPOINT=https://hf-mirror.com liberopro-download-assets --skip-existing

3. (可选) 真实机器人依赖
------------------------

Franka 与 SO-101 的支持正在逐步接入; 每个机器人的 driver 会以一个包的
形式放在 ``robots/<name>/`` 下, 并附带 ``README.md`` 说明其 SDK / 固件
要求。当前进度参见 :doc:`usage/franka` 与 :doc:`usage/so101`。

验证安装
--------

验证安装最直接的方法是完整运行一个 LIBERO 任务，具体步骤见
:doc:`quickstart`。任务成功运行，说明 env server、VLA server 和 planner
均能正常工作。

如果出错:

- env server 的 stdout / stderr 会写到
  ``<output_dir>/env_server.log``。
- VLA server 的日志在 ``<output_dir>/vla_server.log``。
- Agent 本身的运行日志在 ``<output_dir>/run.log``。

这三份日志都保存在本次运行的临时目录中，排查失败任务时无需再从其他位置
收集日志。
