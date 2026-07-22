安装
====

RPent 用一条 ``pip install`` 即可安装。其 optional-dependency extra 会以
git 依赖的形式拉取 `RLinf <https://github.com/RLinf/RLinf>`_ fork 运行时、
openpi、LIBERO 仿真器以及 SAM 3.0, 因此不再需要单独 clone RLinf 或运行
安装脚本。

先决条件
--------

- Linux + NVIDIA GPU (LIBERO 通过 EGL 渲染)。
- 与显卡匹配的 CUDA 12.x 驱动。
- Python 3.10–3.11。
- ``git``、``bash``、以及能编译 MuJoCo / robosuite 的 C 工具链。

同时你还需要:

- 至少一个 LLM 提供商的 API key —— Anthropic、OpenAI, 或 OpenAI 兼容的
  chat 接口 —— 用于 reasoning brain。
- 一个 VLA checkpoint。LIBERO / Pi0.5 推荐使用
  `HuggingFace: rlinf-pi05-libero-130-fullshot-sft
  <https://huggingface.co/datasets/RLinf/rlinf-pi05-libero-130-fullshot-sft>`_。
- 本地 SAM 3.0 ``sam3.pt`` 文件, 可从 `Hugging Face: facebook/sam3
  <https://huggingface.co/facebook/sam3>`_ 或 `ModelScope: facebook/sam3
  <https://modelscope.cn/models/facebook/sam3>`_ 下载。

1. 用 pip 安装 RPent
--------------------

Clone RPent (用于 CLI 与运行配置), 再按需选择 extra 安装:

.. code-block:: bash

   git clone https://github.com/RLinf/RPent rpent && cd rpent
   pip install -e ".[full]"

``.[full]`` 是默认的端到端组合 —— openpi Pi0.5 VLA + LIBERO-PRO 仿真器
+ SAM 3.0, 运行在 RLinf 运行时之上。

可用的 extra:

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
     - 仅固定版本的官方 SAM 3.0 分割依赖

2. (可选) 真实机器人依赖
------------------------

Franka 与 SO-101 的支持正在逐步接入; 每个机器人的 driver 会以一个包的
形式放在 ``robots/<name>/`` 下, 并附带 ``README.md`` 说明其 SDK / 固件
要求。当前进度参见 :doc:`usage/franka` 与 :doc:`usage/so101`。

验证安装
--------

最快的验证方法是端到端跑通一个 LIBERO 任务 —— 见 :doc:`quickstart`。
如果成功, 说明 env server、VLA server、SAM3 server、reasoning brain 四者都健康

如果出错:

- env server 的 stdout / stderr 会写到
  ``<output_dir>/env_server.log``。
- VLA server 的日志在 ``<output_dir>/vla_server.log``。
- SAM3 server 的日志在 ``<output_dir>/sam3_server.log``。
- Agent 本身的运行日志在 ``<output_dir>/run.log``。

这些日志都放在这一次运行的 scratch 目录下, 所以失败的运行是自包含的、
易于排查。
