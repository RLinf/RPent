安装
====

一个 RPent 环境就是你写在命令行上的那些组件，加上它们 ``requires`` 的闭包。每个组件在 ``component.toml`` 中声明自己的依赖和 Python 版本范围。``rpent-install`` 创建虚拟环境，安装该闭包和核心 ``rpent`` 包，对每个安装的组件执行导入探针，并写入安装指纹。

环境要求
--------

- Linux ``x86_64`` 和 `uv <https://docs.astral.sh/uv/getting-started/installation/>`_。
- 所有选中组件都支持的 Python minor 版本。``rpent-install`` 采用整个闭包都能接受的 3.12、3.11、3.10 中最新的一个，并让 uv 安装它；运行命令的解释器不决定环境使用的版本。
- 所选组件的 PyTorch 构建需要兼容的 GPU 和驱动。

组件
----

组件名的前缀说明它是什么、代码在哪里：

.. list-table::
   :header-rows: 1
   :widths: 18 34 48

   * - 前缀
     - 位置
     - 内容
   * - ``robot-``
     - ``robots/<name>/``
     - 机器人切片：原生转换、工具和运行时接线。
   * - ``sim-``
     - ``components/sim/<name>/``
     - 机器人和模型共用的仿真器服务。
   * - ``model-``
     - ``components/models/<name>/``
     - 模型服务：VLA、定位或动作模型后端。
   * - ``tool-``
     - ``components/tools/<name>/``
     - 只依赖核心的工具，例如 flywheel 导出器；或由机器人切片适配的能力库，例如 RGB-D 工具。
   * - ``infra-``
     - ``components/infra/<name>/``
     - 环境需要但不拥有代码的依赖包（RLinf 运行时、测试工具、CPU Torch）。

每个组件都是一个目录，目录里放着它的 ``component.toml``。``rpent-install list`` 会列出当前源码目录提供的组件，目录里没有的组件也装不了。为不属于自己的代码安装依赖包的组件只贡献这份清单：``infra-`` 组件全部属于这种形态，``sim-libero_plus`` 则是为 ``sim-libero`` 自带的 env server 安装 LIBERO-plus 包。

核心包分两层安装。``rpent`` 是运行时层：组件与模型契约、RPC、模型 client 和安装器，仿真器或模型环境可以只装它而不带 planner 依赖栈。``rpent[planner]`` 在此之上加入 planner 适配器、执行会话、提示词和 Dashboard；工具契约与 toolkit 属于运行时层。导入 planner 的组件在 ``component.toml`` 中声明 ``planner = true``，包含该组件的环境会一并安装 planner 层；没有这个标记的环境只装运行时层。若组件设置了该标记而源码目录的 ``pyproject.toml`` 中没有 ``planner`` extra，``install`` 会报错并列出相关组件。

依赖方向是 ``rpent`` ← ``components/sim`` ← ``components/models`` ← ``robots``；``components/tools`` 只依赖 ``rpent``，其中的能力库还可以读取模型 client 的结果类型。因此机器人环境会自动带上它的仿真器、模型服务和工具：

.. code-block:: bash

   rpent-install list

``robot-libero`` 需要 ``sim-libero``、``model-openpi``、``model-sam3``、``tool-flywheel`` 和 ``tool-rgbd``；``robot-robocasa`` 需要 ``sim-robocasa`` 和 ``model-rldx``；``robot-robodojo`` 需要 ``sim-robodojo``、``model-sam3``、``model-openpi``、``model-awomo05`` 和 ``model-vpp2``。没有机器人需要的模型服务（``model-molmo``、``model-cosmos3``、``model-cosmos_policy``、``model-fast_wam``、``model-apxinf``、``model-uniwam``、``model-xpolicylab``）装进自己的环境。

``robot-`` 组件一次装齐：它的环境里同时有 planner、工具、仿真器和机器人需要的模型服务，所以 ``install robot-libero`` 不需要再为模型或导出器准备第二个环境。其他前缀的组件各自装进独立的环境（同样带核心 ``rpent`` 包），机器人环境通过该组件的 client 访问它，而不是直接导入。``robot-yam`` 是例外：它的切片自带 YAM 运行时，Pi0.5 策略则作为 ``model-openpi`` 单独安装。

查看与检查组件
--------------

.. code-block:: bash

   rpent-install list
   rpent-install check --all

``check`` 读取清单，计算每个组件的 ``requires`` 闭包，报告未知组件名、不支持的 Python 版本、互相矛盾的精确版本固定，以及组件通过 ``env_conflicts`` 声明的冲突。点名多个组件时检查它们共同组成的那个环境，也就是 ``install`` 会创建的环境；``--all`` 则逐个组件检查它自己的环境。它不会下载任何依赖。加上 ``--resolve`` 后，它还会用 ``uv pip compile`` 验证声明的依赖在 Linux ``x86_64`` 目标上可解，并在检查后删除生成的 requirements。

安装环境
--------

在源码目录中把组件安装到独立的虚拟环境：

.. code-block:: bash

   rpent-install install robot-libero --venv .venv-libero
   source .venv-libero/bin/activate

新检出的仓库还没有 ``rpent-install`` 命令。``rpent-install.sh`` 会先引导它，不需要预先安装 RPent：

.. code-block:: bash

   ./rpent-install.sh install robot-libero --venv .venv-libero

写出多个组件名可以组合成一个环境。非 robot 组件这样装在自己的主机上，机器人环境通过 client 访问它：

.. code-block:: bash

   rpent-install install model-openpi --venv .venv-openpi
   rpent-install install sim-libero_plus model-openpi model-sam3 --venv .venv-libero-plus

``rpent-install`` 需要在源码检出中运行，它从 ``robots/`` 和 ``components/`` 下的 ``component.toml`` 读取清单。非可编辑的 ``pip install rpent`` 只提供 CLI；运行具体的机器人或模型服务请使用源码检出，因为清单在那里。

环境只包含你写出的组件及其闭包。环境里有什么决定既有 venv 能否复用，所以点名一个通过 ``requires`` 带出另一个的组件，与把两者都点名，描述的是同一个环境；组件换了包也仍是同一个环境。持有另一组组件的 venv 会被拒绝；请为这个集合指定另一个 ``--venv``，或加 ``--force`` 替换它。``--force`` 只重建该路径上的环境，重建失败时恢复被替换的环境，不会删除不是虚拟环境的目录。已经装有包、但没有 ``rpent-install`` 指纹的 venv 同样会被拒绝：其中的包属于先安装它们的人，只有空环境（刚创建的 ``uv venv``）才会被直接使用。安装失败留下的环境里还有生成的 requirements，重跑同一条命令会接管并补完它；若 requirements 是另一套环境生成的则会被拒绝，与已装完的环境一样。生成的 requirements 和实际安装的包版本都写进环境自身，不会留在环境之外。不传 ``--venv`` 时，目标集合安装到源码目录下的 ``.rpent/venvs/<组件名以 + 连接>``；``.rpent/`` 是本地状态、已被 git 忽略，之后的 ``install`` 会复用同一个环境。``doctor`` 不由目标名推导路径：先激活该环境，或用 ``--venv`` 传入它的路径。

重复执行时，记录的安装输入未变的环境会被复用；组件清单、override 文件、索引设置或仓库的 ``[tool.uv]`` 设置变了，命令按新输入重建环境，较早清单装上的包不会留在环境里。安装器根据清单生成 requirements，再由 ``uv pip install`` 解析安装。清单中的直接依赖版本约束和 Git revision 仍然生效；随着上游发布新版本，不同时间全新安装的传递依赖版本可能不同。``--dry-run`` 只打印生成的 requirements，不执行安装。

任意数量的组件都可以贡献依赖包。索引设置（``index-url``、``extra-index-urls``、``index-strategy``）在整个闭包上合并；两个组件对它们的要求不一致时，该环境判定为无效，而不是悄悄选择其中一个。

常用环境
--------

下面的每条命令只能安装当前源码检出中存在的组件，``rpent-install list`` 会打印当前集合。某个机器人、仿真器或模型条目可能先出现在这里、对应组件目录随后才落地，在落地前执行会报 ``unknown component``。

.. code-block:: bash

   rpent-install install infra-minimal --venv .venv-minimal
   rpent-install install infra-dev --venv .venv-dev
   rpent-install install infra-unit_cpu --venv .venv-unit-cpu
   rpent-install install robot-libero --venv .venv-libero
   rpent-install install sim-libero_plus model-openpi model-sam3 --venv .venv-libero-plus
   rpent-install install robot-robocasa --venv .venv-robocasa
   rpent-install install robot-robotwin --venv .venv-robotwin
   rpent-install install robot-franka --venv .venv-franka
   rpent-install install robot-dual_franka --venv .venv-dual-franka
   rpent-install install robot-yam --venv .venv-yam
   rpent-install install robot-robodojo --venv .venv-robodojo
   rpent-install install model-molmo --venv .venv-molmo
   rpent-install install model-openpi --venv .venv-openpi

``infra-minimal`` 是核心环境：只有 RPent 及其自身依赖，不含任何机器人、仿真器或模型包，足以验证 CLI 和机器人发现。``infra-dev`` 再加上测试工具。``infra-unit_cpu`` 在测试工具之外再加一套 CPU Torch 构建的模型依赖，供单元测试使用。机器人环境覆盖受支持的真机和仿真器；平台准备、checkpoint 和资产见各仿真器或机器人页面。``sim-libero_plus`` 替代而不是叠加 ``sim-libero``：LIBERO-plus 要求 robosuite<1.5，两套包不能共用一个环境。plus 环境承载环境服务，机器人侧从 ``robot-libero`` 环境运行并连接它（见 :doc:`../simulators/libero`）。

环境诊断
--------

.. code-block:: bash

   rpent-install doctor --venv .venv-libero
   rpent-install doctor --venv .venv-libero --fingerprint

``doctor`` 将安装指纹与每项记录在案的安装输入比较（每个组件的清单 hash、它的 override 文件，以及仓库的 ``[tool.uv]`` 设置），检查记录的 Python 版本和平台，运行 ``uv pip check`` 与组件探针，并比较当前安装包集合与记录的 freeze hash。``uv`` 的解析在仓库根目录进行，所以比较的设置不随命令的执行目录变化；用户级 ``uv`` 配置（``~/.config/uv``）不在比较范围内。为另一组组件安装过的 venv 会报告为需要重装，而不是环境已漂移。``--fingerprint`` 打印存储的指纹及其 hash，不执行上述检查。

``doctor`` 描述的是 ``rpent-install`` 按清单创建的环境。部分真机和仿真器文档会在该环境上继续追加包，例如可编辑安装的 RLinf 检出、SDK 或适配器包。这些追加会改变实际安装的包集合，因此 ``doctor`` 会把环境报告为已漂移，重跑 ``install`` 会记录新的包集合。需要指纹只对应组件闭包时，请在按清单装出的环境上运行 ``doctor``。

执行 ``check --resolve`` 时，``linux-x86_64`` 默认以 ``manylinux_2_31`` 为目标（glibc 2.31，即 Ubuntu 20.04 或更新版本）：当前 GPU wheel 发布在更新的基线上，2.17 的目标解析不到它们。若某组件的依赖包只发布了更新的基线，可在它的清单里写 ``manylinux = "2_35"``，把整个闭包的目标基线提高；``sim-robodojo`` 因 Isaac Sim 就这样声明。对包含 cuRobo 构建的配置使用 ``--resolve`` 需要设置了 ``CUDA_HOME`` 的 Linux 主机，因为 cuRobo 在解析期间构建元数据。某个组件解析失败时会报告错误并继续检查其余组件。
