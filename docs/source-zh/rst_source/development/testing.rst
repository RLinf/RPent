测试与 CI
=========

RPent 的测试体系分为离线 CPU 单元测试、仓库布局检查和声明式 GPU 端到端套件三层。离线单元测试与布局检查可在普通 CPU 开发机上直接运行；GPU 端到端套件需要具备独立 NVIDIA GPU、EGL 及对应资产的主机。


测试结构
--------

``tests/`` 包含三个顶层目录：

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - 目录
     - 职责
   * - ``tests/unit_tests/``
     - 离线 CPU 单元测试，无需 GPU、仿真器或模型权重。不启动真实服务，使用轻量级内存替代实现验证接口、配置解析、数据格式转换与进程清理。
   * - ``tests/tooling/``
     - 独立检查脚本，无需安装 Python 环境即可运行，用于校验目录结构、``component.toml`` 依赖清单与模块导入边界。
   * - ``tests/e2e_tests/``
     - 声明式 GPU 端到端测试套件（``suites/*.toml``）与通用运行脚本（``run_gpu_suite.sh``），通过公开 CLI 依次覆盖当前环境暴露的全部工具。

单元测试目录与源码目录保持对应：

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - 源码目录
     - 单元测试目录
   * - ``rpent/<module>``
     - ``tests/unit_tests/rpent/<module>/``
   * - ``robots/<name>/``
     - ``tests/unit_tests/robots/<name>/``
   * - ``components/models/<name>/``
     - ``tests/unit_tests/components/models/``
   * - ``components/tools/<name>/``
     - ``tests/unit_tests/components/tools/``
   * - ``tests/tooling/``
     - ``tests/unit_tests/tooling/``
   * - ``component.toml`` 依赖清单
     - 由 ``tests/unit_tests/installs/`` 与 ``./tests/tooling/run.sh layout`` 自动校验

编写测试时遵循以下约定：

- 仅单个测试文件用到的辅助类与准备逻辑保留在当前测试文件内；多个测试文件共用的初始化逻辑再移至 ``conftest.py``。
- 测试文件按验证重点命名：``*_contracts.py`` 验证接口与配置，``*_loopback.py`` 验证本地端口或 HTTP 通信，``*_lifecycle.py`` 验证服务启停与清理，``*_smoke.py`` 验证基础启动流程。
- ``tests/unit_tests/rpent/install/`` 使用临时目录验证安装器自身逻辑；针对仓库内真实 ``component.toml`` 清单的校验放在 ``tests/unit_tests/installs/``。

Simulator CI 选择与运行机制
---------------------------

GitHub 在每次推送和 PR 时自动运行 CPU 单元测试，并在日志中输出代码行覆盖率。**Simulator CI** 按以下规则验证安装流程与 GPU 套件：

- **触发与门禁**：在提交合入 ``main`` 或非草稿 PR 贴上 ``run-ci`` 标签时运行。未贴标签或仍为草稿的 PR 会在选择阶段拦截；fork PR 推送新提交后需重新贴上 ``run-ci`` 标签。
- **安装与评测拆分**：任务拆分为组件级 ``install``\ （不占 GPU 槽位，通过 ``gpu_slots.py --kind install`` 控制安装并发）与用例级 ``eval``\ （``<suite>:<case>``）。GPU 槽位由 ``tests/tooling/gpu_slots.py`` 的本机 ``flock`` 槽位池按需分配（1 张或多张）。
- **缓存环境复用**：改动涉及安装器、打包或 ``component.toml`` 依赖清单时，先执行干净环境的 ``install``。纯运行时代码改动会跳过 ``install``，直接复用按依赖哈希缓存的虚拟环境执行 ``eval``。
- **按路径精准筛选**：贴上 ``run-ci`` 后，修改 RLDX 运行时代码只触发 ``robocasa:rldx``，修改共享运行时代码触发全部 ``eval`` 用例，只改文档则跳过 GPU 测试并直接通过。

每个 GPU 套件执行的都是有界工具覆盖检查，不等同于环境原生 ``terminated`` 或评估器判定任务成功。套件通过公开 CLI 依次调用当前环境暴露的全部工具，并最终调用 ``finish``。

新增机器人或模型所需测试
------------------------

接入新机器人或仿真环境
~~~~~~~~~~~~~~~~~~~~~~

在 ``robots/<name>/`` 下新增机器人（或同时在 ``components/sim/<name>/`` 下新增仿真器）时，需补充以下两部分：

1. **离线 CPU 单元测试**\ （位于 ``tests/unit_tests/robots/<name>/``）：

   - 验证机器人注册、CLI 参数、配置解析、Prompt 组装与工具定义。
   - 使用轻量级本地测试服务替代真实仿真器与模型，验证 CLI 执行、本地 RPC 通信与子进程正常退出。
   - 运行 ``./tests/tooling/run.sh layout`` 与 ``tests/unit_tests/installs/``，确认 ``component.toml`` 与导入边界通过校验。

2. **GPU 端到端套件**\ （新增 ``tests/e2e_tests/suites/<name>.toml``，参考同目录的 ``robocasa.toml`` 或 ``libero.toml``）：

   CI 会自动发现 ``suites/*.toml``，无需修改脚本或工作流：

   - 填写 ``component`` 指定安装组件，并在 ``[[cases]]`` 中为每个动作模型后端写一条用例（共用参数和环境变量放 ``base_args`` / ``env``）。
   - 资产、权重和解释器直接写 ``assets/...``、``models/...``、``venvs/...`` 相对路径，卡号用 ``{gpu}`` 占位；路径缺失时自动跳过该用例。
   - 可选填 ``action_tool`` 与 ``action_count_field`` 验证动作真正执行；有特殊安装收尾或独立服务需求时，再按需补充 ``[setup]`` 或 ``[cases.sidecar]``。

完整的机器人接入步骤见 :doc:`add_robot`。

接入新模型
~~~~~~~~~~

在 ``components/models/<name>/`` 下新增动作模型或感知模型时，需补充以下两部分：

1. **离线 CPU 单元测试**\ （位于 ``tests/unit_tests/components/models/``，特定机器人的适配逻辑位于 ``tests/unit_tests/robots/<robot>/``）：

   - 验证配置校验、启动参数检查，以及 RPent 物理量与模型输入输出之间的格式转换。
   - 使用轻量级测试后端验证本地 HTTP/socket 通信，包括多会话隔离、reset、错误处理与 worker 正常退出。

2. **GPU 端到端用例**\ （加入对应机器人的 ``tests/e2e_tests/suites/<robot>.toml``）：

   - 新增一条 ``[[cases]]``，填写该后端的 CLI 参数与共享目录下的资源相对路径。
   - 缺少权重、解释器或 endpoint 时会自动跳过对应用例，不影响同套件内其他用例。

完整的动作模型接入步骤见 :doc:`add_action_model`。

本地运行测试
------------

本地运行离线测试
~~~~~~~~~~~~~~~~

安装 ``infra-unit_cpu``\ （包含模型服务单元测试所需的 CPU 版 PyTorch），再运行 pytest：

.. code-block:: bash

   ./rpent-install.sh install infra-unit_cpu --venv .venv-unit-cpu
   .venv-unit-cpu/bin/python -m pytest tests/unit_tests -v
   # 将 <name> 替换为 robots/ 下的机器人目录名（例如 libero、robocasa）。
   .venv-unit-cpu/bin/python -m pytest tests/unit_tests/robots/<name> -v

仓库结构检查无需激活虚拟环境，直接执行：

.. code-block:: bash

   ./tests/tooling/run.sh layout

本地运行 GPU 套件
~~~~~~~~~~~~~~~~~

在具备独立 NVIDIA GPU、EGL 以及对应资产和权重的机器上，传入两个不存在的输出目录与虚拟环境目录运行套件。其中 ``<target>`` 为 ``tests/e2e_tests/suites/*.toml`` 的文件名（不含后缀，如 ``libero``、``libero-pro``、``robocasa``、``robotwin``）或 ``all``。单卡套件只需指定一个 GPU 编号（如 ``0``）；运行 ``robodojo`` 或 ``all`` 时需提供两张物理 GPU（如 ``0,1``）：

.. code-block:: bash

   export CUDA_VISIBLE_DEVICES=0  # 运行 robodojo 或 all 时改为两张卡，例如 0,1
   export MUJOCO_GL=egl
   export PYOPENGL_PLATFORM=egl

   bash tests/e2e_tests/run_gpu_suite.sh \
     <target> \
     /path/to/new-output-dir \
     /path/to/new-venv-root

套件失败时，可在 ``/path/to/new-output-dir`` 下查看各套件的 JUnit XML 报告、每个用例目录内的 ``env_server.log`` / 模型服务日志，以及 ``*-daemon-cleanup.txt`` 进程清理记录。

在同时安装了目标机器人与 ``infra-dev``\ （提供 ``pytest`` 及测试依赖，如 ``./rpent-install.sh install robot-libero infra-dev --venv .venv-libero-dev``）的同一个虚拟环境中，设置好 GPU 与 EGL 环境变量（资源不在默认 ``/mnt/public/rpent_shared`` 时另设 ``RPENT_SHARED_ROOT``）后，可用 pytest 的 ``-k`` 过滤表达式（如 ``pi05``、``rldx``）直接调试单个套件或用例：

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
     RPENT_E2E_SUITE=<target> pytest tests/e2e_tests/test_suites.py -v -k <expression>
