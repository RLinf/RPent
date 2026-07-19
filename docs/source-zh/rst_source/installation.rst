安装
====

RPent 依赖 `RLinf <https://github.com/RLinf/RLinf>`_ 的一个 fork 分支,
后者提供 LIBERO / RoboCasa 仿真器以及 VLA 训练 / 推理栈。这两个仓库
需要并排放在同一个 ``workspace/`` 目录下。

先决条件
--------

- Linux + NVIDIA GPU (LIBERO / RoboCasa 通过 EGL 渲染)。
- 与显卡匹配的 CUDA 12.x 驱动。
- Python 3.10+。
- `uv <https://github.com/astral-sh/uv>`_ (用于在 RLinf 提供的虚拟环境
  上同步 RPent 的额外依赖)。
- ``git``、``bash``、以及能编译 MuJoCo / robosuite 的 C 工具链。

同时你还需要:

- 至少一个 LLM 提供商的 API key —— Anthropic、OpenAI, 或 OpenAI 兼容的
  chat 接口 —— 用于 reasoning brain。
- 一个 VLA checkpoint。LIBERO / Pi0.5 推荐使用
  `HuggingFace: rlinf-pi05-libero-130-fullshot-sft
  <https://huggingface.co/datasets/RLinf/rlinf-pi05-libero-130-fullshot-sft>`_。

1. 并排克隆 RLinf 与 RPent
--------------------------

.. code-block:: bash

   mkdir workspace && cd workspace
   # RPent 依赖 RLinf 的一个 fork 分支; 更多迭代后会合并回 RLinf 主线。
   git clone https://github.com/jx-qiu/RLinf -b feature/physicalagent rlinf
   git clone https://github.com/RLinf/RPent rpent

期望的目录结构:

.. code-block:: text

   workspace/
   ├── rlinf/     # RLinf fork (仿真器 + VLA 基础设施)
   └── rpent/     # RPent (agent 框架 —— 本仓库)

2. 创建 RLinf 虚拟环境 (LIBERO + openpi)
-----------------------------------------

RLinf 提供一键安装脚本, 一次性搭好带 LIBERO + openpi (Pi0.5 运行时) 的
虚拟环境:

.. code-block:: bash

   cd rlinf
   bash requirements/install.sh embodied \
     --env libero --model openpi --use-mirror \
     --venv ../.venv-opi-libero
   cd ..
   source .venv-opi-libero/bin/activate

安装脚本会把虚拟环境创建在 ``rlinf/`` 的上一级, 方便 RPent 复用。进入下一步
前 **先激活它**。

3. 安装 RPent 的额外依赖
------------------------

激活了 RLinf 虚拟环境之后, 在其上同步 RPent 的额外依赖, 并运行 LIBERO
PRO/PLUS 的安装脚本:

.. code-block:: bash

   cd rpent
   uv sync --active --inexact
   bash scripts/install_libero_pro_plus.sh

``uv sync --active --inexact`` 会在不打扰 RLinf 已装包的前提下, 补齐
RPent 自己的 Python 依赖 (pydantic-ai、Claude Agent SDK、Codex SDK、
用于 dashboard 的 FastAPI 等)。``install_libero_pro_plus.sh`` 会 patch
LIBERO, 暴露出 RPent 全流程使用的 ``pro`` / ``plus`` 任务变体。

4. (可选) RoboCasa
------------------

RoboCasa (厨房尺度、长时序操作) 有自己的一键脚本:

.. code-block:: bash

   bash scripts/setup_robocasa.sh

完整的 RoboCasa365 + RLDX-1 安装与运行流程见
`docs/SETUP_ROBOCASA.zh.md
<https://github.com/RLinf/RPent/blob/main/docs/SETUP_ROBOCASA.zh.md>`_
(资产、controller 配置、VLA checkpoint)。

5. (可选) 真实机器人依赖
------------------------

Franka 与 SO-101 的支持正在逐步接入; 每个机器人的 driver 会以一个包的
形式放在 ``robots/<name>/`` 下, 并附带 ``README.md`` 说明其 SDK / 固件
要求。当前进度参见 :doc:`usage/franka` 与 :doc:`usage/so101`。

验证安装
--------

最快的验证方法是端到端跑通一个 LIBERO 任务 —— 见 :doc:`quickstart`。
如果成功, 说明 env server、VLA server、reasoning brain 三者都健康。

如果出错:

- env server 的 stdout / stderr 会写到
  ``<output_dir>/env_server.log``。
- VLA server 的日志在 ``<output_dir>/vla_server.log``。
- Agent 本身的运行日志在 ``<output_dir>/run.log``。

三份日志都放在这一次运行的 scratch 目录下, 所以失败的运行是自包含的、
易于排查。
