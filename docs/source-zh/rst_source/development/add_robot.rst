添加机器人或仿真环境
==============================

本指南介绍如何将新的真实机器人或仿真环境接入 RPent，让语言模型通过工具调用完成任务。完整的参考实现见 ``robots/libero/``。

接入原则
--------

- **复用 RPent 公共抽象。** Env、VLA、运行时和 Memory 优先复用已有组件，例如 ``BaseEnvClient``、``BaseEnvFacade``、``BaseVLAClient``、 ``BaseVLAFacade`` 和 ``MemoryManager``。
- **优先复用 RLinf 的 Env 或 VLA。** 如果 RLinf 已有对应实现，RPent 尽量只增加必要的适配层。
- **尽量和现有机器人接入方式保持一致。** ``RobotSpec``、Prompt、Toolkit 和运行时尽量参考已有实现，不为单个机器人引入新的公共机制。
- **无法复用时说明原因。** 如果 RPent 或 RLinf 已有相关 Env / VLA 但无法复用，请在 PR 描述中说明或提交 issue，帮助改进现有抽象。

接入步骤概览
------------

RPent 的整体进程划分、服务职责和通信方式见 :doc:`系统说明 <architecture>`。接入机器人或仿真环境时，按以下顺序实现扩展点：

1. 在 :ref:`入口 <add-robot-entry>` 中注册 ``RobotSpec`` 和 toolkit 工厂，并声明 ``component.toml``。
2. 实现 :ref:`env_client 和 env_server <add-robot-env-rpc>`。如需接入 VLA 服务和 model client，参见
   :ref:`添加一个 VLA（或其他基于模型的原语）<add-primitive-model-based>`。
3. :ref:`定义 prompt <add-robot-prompts>`。
4. :ref:`实现 toolkit 和 primitives <add-robot-toolkit>`。
5. :ref:`注册环境参数并生成 RunConfig <add-robot-config>`。
6. 实现 :ref:`runtime 钩子 <add-robot-runtime>`：同一个钩子既能为普通 CLI 初始化完整 runtime，也能为 Dashboard 初始化指定的 component 子集。
7. 补充 :ref:`离线 CPU 单元测试与声明式 GPU 端到端套件 <add-robot-testing>`\ （``tests/e2e_tests/suites/<name>.toml``）。

.. _add-robot-entry:

入口与组件清单
--------------

每个机器人对应 ``robots/`` 下的一个组件目录，并配有一份 ``component.toml`` 清单声明安装方式。以新机器人 ``myrobot`` 为例，目录结构如下：

.. code-block:: text

   robots/myrobot/
       component.toml          # name、probe、python、requires、dependencies
       __init__.py             # 惰性包入口
       robot_spec.py           # RobotSpec、工厂、Dashboard 描述和 runtime 钩子
       env_client.py           # MyEnvClient —— agent 侧 RPC 客户端（第 1 节）
       prompt_bundle.py        # system_prompt()/user_prompt() prompt 工厂（第 2 节）
       toolkit.py              # MyRobotToolkit、primitives 与工具定义（第 3 节）

   components/sim/mysim/       # 共享仿真器组件（接入仿真环境时）
       component.toml          # name = "sim-mysim" 及仿真器依赖
       env_server.py           # 服务端 facade 与 RPC 服务（第 1 节）

``robots/myrobot/component.toml`` 声明 ``rpent-install`` 读取的清单字段：

.. code-block:: toml

   schema = 2
   name = "robot-myrobot"
   probe = "robots.myrobot"
   planner = true
   python = ">=3.10,<3.13"
   requires = ["sim-mysim", "model-mymodel"]

   dependencies = []

仿真器包依赖（例如 ``dependencies = ["my-simulator==1.2"]``）与 ``env_server.py`` 放在 ``components/sim/<name>/`` 中，以便多个机器人复用；只有真机驱动没有共享仿真器组件时，才把 ``env_server.py`` 与硬件专属的 ``dependencies`` 直接放在 ``robots/<name>/`` 下。

所有环境均面向 Linux ``x86_64``，因此清单无需声明平台字段；核心字段说明如下：

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - 字段
     - 说明
   * - ``schema``
     - 清单格式版本（``2``）。
   * - ``name``
     - 必须为 ``robot-<目录名>``，前缀决定组件类型与代码位置。
   * - ``probe``
     - 可导入的包路径。``rpent-install install`` 与 ``doctor`` 会导入该包及其包含的入口模块（``robot_spec``、``env_client``），确认真实依赖已就绪。
   * - ``planner``
     - 设为 ``true`` 时，在运行时层 ``rpent`` 之上额外安装 ``rpent[planner]``；仅提供模型服务或仿真器的组件可省略此字段。
   * - ``python``
     - 支持的 Python 版本范围（例如 ``">=3.10,<3.13"``）；``rpent-install`` 会选择环境内所有组件共同支持的最新版本。
   * - ``requires``
     - 需安装到同一环境中的其他组件（通常是对应的仿真器、模型服务和工具）。
   * - ``dependencies``
     - 仅限本机器人独有的 Python 依赖；多个组件共用的依赖应放到 ``components/infra/`` 下的基础设施组件中。
   * - ``imports``
     - 只有本机器人直接复用另一个机器人的代码时填写对方名称，且两者之一须在 ``requires`` 中声明对方以确保一同安装。工具包与模型服务通过契约解耦，不互相 import。

``__init__.py`` 是包入口，必须保持惰性。CLI 参数解析（``rpent --robot <name> --help``）、组件探针（``rpent-install doctor``）和布局检查导入 ``robots.<name>`` 时，不得提前拉起可选的仿真器或模型依赖。通过 ``__getattr__`` 暴露工厂：

.. code-block:: python

   # robots/myrobot/__init__.py
   from typing import TYPE_CHECKING, Any

   if TYPE_CHECKING:
       from robots.myrobot.robot_spec import get_robot_spec, get_toolkit

   __all__ = ["get_robot_spec", "get_toolkit"]


   def __getattr__(name: str) -> Any:
       if name in __all__:
           from robots.myrobot import robot_spec

           return getattr(robot_spec, name)
       raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

``rpent/robots/base.py`` 中的注册表会按需导入 ``robots.<name>``，并调用这两个函数：

.. code-block:: python

   # robots/myrobot/robot_spec.py
   from rpent.dashboard.events import DashboardEventSink
   from rpent.memory import MemoryManager
   from rpent.robots.robot_spec import RobotSpec, RunConfig
   from rpent.robots.prompt_bundle import PromptBundle
   from rpent.utils.config import get_memory_dir
   from robots.myrobot.prompt_bundle import system_prompt, user_prompt

   MYROBOT_DASHBOARD_SPEC = {...}

   def get_robot_spec() -> RobotSpec:
       return RobotSpec(
           name="myrobot",
           prompts=PromptBundle(system=system_prompt, user=user_prompt),
           add_cli_args=_add_cli_args,
           parse_config=_parse_config,
           init_runtime=_init_runtime,
           resolve_dashboard=lambda args: MYROBOT_DASHBOARD_SPEC,
       )

   def get_toolkit(
       *,
       runtime_kwargs,
       dashboard_events: DashboardEventSink,
       config: RunConfig,
   ):
       from robots.myrobot.toolkit import MyRobotToolkit
       return MyRobotToolkit(
           runtime_kwargs=runtime_kwargs,
           dashboard_events=dashboard_events,
           memory=MemoryManager(
               root=config.prompt_vars.get("memory_dir") or get_memory_dir("myrobot"),
           ),
       )

   def _add_cli_args(parser, use_dashboard) -> None:
       """向共享 parser 注册机器人参数。见第 4 节。"""
       ...

   def _parse_config(args) -> RunConfig:
       """校验最终的 args，返回 RunConfig。见第 4 节。"""
       ...

   def _init_runtime(
       args,
       output_dir,
       dashboard_events: DashboardEventSink,
       components: set[str] | None,
   ):
       """初始化全部 runtime components，或只初始化指定子集。

       返回 (daemons, runtime_kwargs)。见第 5 节。
       """
       ...

``resolve_dashboard`` 是可选回调；环境不支持 Dashboard 控制时保持为 ``None``。支持时，由回调返回机器人包中定义的配置：其中 ``task`` 描述命令、校验字段、展示模板和输出目录名称；机器人专用的会话设置仍通过普通命令行参数传入；``runtime_components`` 定义界面中的服务列表；``primitives`` 按顺序列出 Dashboard 展示并允许直接执行的 Toolkit 动作。相机标签页根据每步记录的 PNG 图像文件自动生成。任务候选项应直接保存在该配置中，避免导入机器人包时依赖仿真器包。完整结构参考 ``robots/libero/robot_spec.py``。

``_resolve_robot(name)`` 通过 ``importlib.import_module(f"robots.{name}")`` 动态加载机器人包。因此，只需将机器人包放在 ``robots/`` 下，无需维护中央注册列表。

``resolve_dashboard(args)`` 回调接收解析后的会话参数，可根据 ``--am-backend`` 等选项选择组件。固定布局直接返回静态配置即可，如上例所示。实体环境检查和 Dashboard 启动均调用此回调。

``prepare_prompt_vars`` 是可选钩子：在 ``init_runtime`` 连接任务后细化 Prompt 变量，例如列出服务实际注册的工具。返回的映射会合并到 ``RunConfig.prompt_vars``，供用户 Prompt 与系统 Prompt 渲染；交互式 CLI 的首条消息仍采用合并前的结果作为可编辑默认值。若 ``get_toolkit`` 也需要这些运行时信息，应让它和 Prompt 根据同一份信息生成工具列表，确保 Prompt 列出的工具与 Toolkit 实际注册的工具一致。

下文依次说明这些模块需要实现的内容。``_add_cli_args`` 和 ``_parse_config`` 见第 4 节，runtime 钩子见第 5 节。

.. _add-robot-env-rpc:

1. ``env_client.py`` + ``env_server.py``
-----------------------------------------

这两个文件连接 agent 进程与 ``env_server``。client 在 agent 进程内将方法调用转换成 RPC 请求，``env_server`` 负责处理这些请求。

1.1 Env client（agent 侧）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

继承 :class:`rpent.components.env.client.BaseEnvClient`。它已经负责启动时校验 ``env.get_env_meta``、执行首次 reset、缓存 ``last_obs``，并实现公共的 ``reset``、``step``、``chunk_step``、``render_camera``、 ``get_camera_meta`` 和 ``get_task_language`` RPC。子类只需增加环境专用方法；扩展方法需要独立超时时，再扩展超时表。RPC 名称需要保持稳定，因为服务端 facade 会显式注册每个名称。

.. code-block:: python

   from rpent.components.env.client import BaseEnvClient

   class MyEnvClient(BaseEnvClient):
       _TIMEOUT_S = {
           **BaseEnvClient._TIMEOUT_S,
           "env.custom_method": 30.0,
       }

       def custom_method(self, arg):
           return self._client.call(
               "env.custom_method",
               args=(arg,),
               timeout_s=self._TIMEOUT_S["env.custom_method"],
           )

   env = MyEnvClient(rpc_client, expected_meta=expected_meta)

1.2 Env server（环境侧）
~~~~~~~~~~~~~~~~~~~~~~~~~~

在 ``env_server`` 中定义与 client API 对应的 facade 类，例如 ``MyEnvFacade``。该类继承
:class:`rpent.components.env.facade.BaseEnvFacade`；基类已提供公共 RPC 路由和
读写分派锁。子类实现公共环境方法，并通过 ``_register_rpc`` 增加环境专用路由。方法接收与 client 一致的位置参数和关键字参数，返回传输层支持的 Python / NumPy 值（不要返回 torch；agent 进程不导入 torch）。

.. code-block:: python

   from rpent.components.env.facade import BaseEnvFacade

   class MyEnvFacade(BaseEnvFacade):
       def __init__(self, env, meta):
           self._env = env
           self._meta = meta
           super().__init__()

       def _register_rpc(self):
           super()._register_rpc()
           # 自定义方法需额外注册
           self._rpc["env.custom_method"] = self.custom_method

       # BaseEnvFacade 要求的抽象方法必须实现
       def get_env_meta(self): ...
       def reset(self): ...
       def step(self, action): ...
       def chunk_step(self, actions, **kwargs): ...
       def get_camera_meta(self, camera_name, **kwargs): ...
       def render_camera(self, camera_name, **kwargs): ...
       def get_task_language(self): ...

       def custom_method(self, arg): ...

   facade = MyEnvFacade(env, meta)
   facade.serve(transport="http", host=host, port=port)

``BaseEnvFacade`` 通过 ``_register_rpc`` 注册公共路由，并使用读写锁串行化会改变状态的调用。只有确认某个扩展路由可以安全地与其他读操作并发时，才把它加入 ``_readonly_methods``。继承的 ``RpcFacade.serve`` 负责绑定传输方式（HTTP 或 socket）、提供 ``healthz`` 和 ``shutdown``、检测父进程退出并执行资源清理。

.. _add-robot-prompts:

2. ``prompt_bundle.py``
-----------------------

定义 ``system_prompt()`` 和 ``user_prompt()`` 两个 prompt 工厂，并在机器人的 ``robot_spec.py`` 中构造 ``PromptBundle(system=system_prompt, user=user_prompt)``\ （见上面的“入口与组件清单”）。每个工厂返回一个有序的 ``dict[str, PromptNode]``，其中包含带标题的分节； ``PromptBundle.render`` 负责组装和填充。一套 prompt 供 API loop、Claude Code 和 Codex 等 planner 共用。正文使用工具的裸名（如 ``move_to``），并说明 Claude Code 和 Codex SDK 会将其显示为 ``mcp__rpent__<name>``；无需分别维护 CLI 与 API 版本。

.. code-block:: python

   # robots/myrobot/prompt_bundle.py
   from robots.myrobot.prompts import system as system_parts
   from robots.myrobot.prompts import user as user_parts
   from rpent.prompt.utils import PromptNode

   def system_prompt() -> PromptNode:
       return {
           "INTRO": system_parts.PREAMBLE,
           "GOAL": system_parts.GOAL,
           "RULES": system_parts.RULES,
           "WORKFLOW": system_parts.WORKFLOW,
           "ENVIRONMENT": system_parts.ENVIRONMENT,
           "OUTPUT": system_parts.OUTPUT,
       }

   def user_prompt() -> PromptNode:
       return {
           "TASK": user_parts.TASK,
           "BEGIN": user_parts.BEGIN,
       }

将 prompt 内容保存在机器人包内，例如 ``robots/myrobot/prompts/system.py`` 和 ``user.py``。分节内容可以是普通字符串，也可以使用 ``BulletList`` 或 ``Numbered``。占位符 ``{{suite}}`` / ``{{task}}`` / ``{{seed}}`` / ``{{output_dir}}`` / ``{{recipe_tag}}`` 在渲染时填充。

.. _add-robot-toolkit:

3. ``toolkit.py``
------------------

该模块负责工具定义、动作原语、每步状态记录，以及允许通过 MCP 调用的工具列表。LIBERO 将这些内容分布在 ``tools.py`` 和 ``toolkit.py`` 中；新增机器人时，也可以将它们放在同一个 ``toolkit.py`` 中。

toolkit 模块通常包含四部分：

**Primitives 类**\ （例如 ``MyRobotPrimitives``）是 toolkit 持有的 Python 对象。它保存 ``EnvClient``、VLA ``model`` client 和单次运行所需的状态。每个原语工具（``move_to``、``pi0_pick``、``release`` 等）对应一个方法，并返回包含日志数据和可选 PNG 图片的 ``ToolResult``。

**工具定义和处理函数** 是使用 ``@tool`` 声明的函数或方法。参数类型注解和 Google 风格 docstring 提供 schema 与说明，包括 ``view_env_state``、``back_project`` 等读取工具。

**每步状态 dump** —— ``dump_state(driver, env_state, log)`` 通过 ``env_state.record_step(...)`` 创建由 ``EnvState`` 持有的步骤，并取得分配的 step index；该 ``StepRecord`` 会被立即追加并提交。大型观测通过 ``env_state.save(...)`` 保存——在 ``record_step`` 块内可省略 ``step`` 参数（默认指向刚创建的步骤），传显式 ``step=<int>`` 可指定其它步骤，``step=None`` 用于运行级工件。每次保存成功后，``EnvState`` 会自动把基础文件名加入该 ``StepRecord`` 的扁平 ``artifacts`` 集合；读取方直接使用规范化的工件文件名。

**Toolkit 类** 继承 ``rpent.tools.toolkit.Toolkit``：

- 在 ``super().__init__(...)`` 中传入 ``memory`` （一个
  必需的 :class:`~rpent.memory.MemoryManager`）和 ``state``。``memory_access`` 和
  ``inbox_cell_tag`` 在构造 ``MemoryManager`` 时配置；eval 默认只读。
- 在 ``__init__`` 中通过自定义的初始化辅助方法构建 primitives（LIBERO 中的方法名为 ``init_primitives``；它会调用 ``EnvState.reset()``、构造原语并 dump 第 0 步）,
- 用 ``self.add_tools(iter_tools(self._primitives))`` 或 ``self.add_tool(definition)`` 注册绑定后的工具；通过 ``definition.with_handler(...)`` 绑定内部资源或执行检查，由 Toolkit 采集动作后的状态；
- 重写 ``close()``，通过 ``EnvState`` 保存 agent 侧剩余工件（例如 ``state.save("episode.mp4", frames, step=None)``）。

``runtime_kwargs`` 由 ``robot_spec.py:get_toolkit`` 转发给 toolkit，再原样传入 primitives 的 ``__init__``。其中通常包含 ``{"env": MyEnvClient(...), "model": VLAClient(...), ...}``。

自定义外部 MCP 注册
~~~~~~~~~~~~~~~~~~~

默认情况下，RPent 会在 ``get_toolkit`` 返回后自动注册 ``--mcp-config``\ （YAML 文件路径）与 ``RunConfig.mcp_servers`` 中的全部外部 MCP 服务，大多数机器人无需编写额外代码（见 :ref:`external-mcp-servers`\ ）。如果机器人需要限定某个服务的可用工具，或要求某组工具与机器人动作互斥执行（``sync_with_robot=True``），可在 ``get_toolkit`` 返回前调用 ``Toolkit.add_mcp_tools``：

.. code-block:: python

   def get_toolkit(*, runtime_kwargs, dashboard_events, config):
       toolkit = MyRobotToolkit(
           runtime_kwargs=runtime_kwargs,
           dashboard_events=dashboard_events,
           memory=MemoryManager(
               root=config.prompt_vars.get("memory_dir") or get_memory_dir("myrobot"),
           ),
       )
       if any(server.name == "dock" for server in config.mcp_servers):
           toolkit.add_mcp_tools(
               config.mcp_servers,
               tools={"dock": ["set_charging_dock", "read_battery"]},
               sync_with_robot=True,
           )
       return toolkit

自动注册流程会跳过已通过 ``add_mcp_tools`` 注册的服务，不会重复挂载。如果传入了未声明的服务名，或指定了被该服务 YAML ``tools`` 列表排除的工具，``add_mcp_tools`` 会抛出 ``ValueError``。设置 ``sync_with_robot=True`` 后，这些工具与机器人动作互斥执行（不会并发运行），执行后会触发 ``get_env_state`` 采集观测，并将远端返回内容保存在观测的 ``tool_result`` 字段中。

建议遵循的约定
--------------

- ``output_dir`` 是 runner 为单次运行创建的工作目录。环境观测由 ``EnvState`` 管理；调用方只使用逻辑基础文件名，不自行拼接存储路径。 transcript 等运行管理输出与环境工件共享该目录。
- 工具提供 ``name``、``description`` 和 ``input_schema``，由各 planner 转换成对应模型接口的格式。
- 环境侧的返回值必须可 pickle，且不包含 torch 对象。
- 每个原语工具执行后要 dump 一次新的状态快照, 这样下一次 ``view_env_state`` 看到的是动作后的世界。
- ``dump_state`` 是 Agent 获取环境状态的唯一数据来源；任何新的模态（例如触觉、力）都通过它提供。

.. _add-robot-config:

4. ``_add_cli_args`` + ``_parse_config`` (runner 钩子)
------------------------------------------------------

机器人特有的 CLI 参数通过两个钩子接入 ``rpent/cli/main.py`` 的解析流程，并参与最终的 argparse 解析：

**``_add_cli_args(parser, use_dashboard) -> None``。** 将机器人参数注册到 main.py 已创建的共享 parser。``use_dashboard`` 决定原本必填的参数是否保持可选。每个 Dashboard TaskRun 会在 ``parse_config`` 调用前，由机器人 Dashboard spec 定义的任务命令提供其声明的字段。main.py 会在 ``parser.parse_args()`` 之前调用该钩子，因此 argparse 的 usage 和错误信息也会包含机器人参数。

**``_parse_config(args) -> RunConfig``。** 普通 CLI 模式下，该钩子在 ``parser.parse_args()`` 后调用；Dashboard 模式下，每个 TaskRun 会先把请求字段写入任务参数，再调用该钩子。该钩子校验这些字段并返回
:class:`~rpent.robots.RunConfig`：

- ``recipe_tag`` —— 单次运行的机器人标签，用于 recipe 路径、planner CLI 输出文件和 transcript 中的 ``recipe_tag`` 字段（LIBERO 使用 ``f"{suite.replace('libero_', '')}_t{task}_s{seed}"``）。
- ``output_dir`` —— 单次运行的临时目录路径。main.py 随后调用 ``init_output_dir`` 创建目录并配置日志。
- ``prompt_vars`` —— 传给 ``PromptBundle.render`` 的字典，通常包含运行标识和 prompt 引用的其他变量。
- ``task_desc`` —— 机器人特定的任务标识字典，会原样写入 transcript JSON 记录（LIBERO 使用 ``{"suite": ..., "task": ..., "seed": ...}``）。
- ``mcp_servers`` —— 可选的机器人自带 MCP 服务元组（可用 ``parse_mcp_servers`` 解析），运行时会与 ``--mcp-config`` 所指定 YAML 文件中的服务配置合并，详见 :ref:`external-mcp-servers`。

.. code-block:: python

   def _add_cli_args(parser, use_dashboard) -> None:
       required = not use_dashboard
       parser.add_argument("--suite", default=None, required=required)
       parser.add_argument("--task", type=int, default=None, required=required)
       # ... 其他机器人参数 ...

   def _parse_config(args) -> RunConfig:
       if not args.suite: raise ValueError("--suite is required")
       # ... 生成 recipe_tag、output_dir 和 prompt_vars ...
       return RunConfig(
           recipe_tag=recipe_tag,
           output_dir=output_dir,
           prompt_vars=prompt_vars,
           task_desc={"suite": args.suite, "task": args.task, "seed": args.seed},
       )

.. _add-robot-runtime:

5. Runtime 初始化钩子
---------------------

``init_runtime`` 返回 ``(owned_daemons, runtime_kwargs)``：

- ``owned_daemons: list[ProcessDaemon]`` 只包含当前进程实际启动的子进程，当前 runner 会在清理阶段停止它们。连接外部 endpoint 时，不能把外部服务加入该列表。
- ``runtime_kwargs: dict`` 会传给 toolkit 构造器，再由后者传入 primitives 的 ``__init__``。完整参数通常包含 ``{"env": MyEnvClient(...), "model": VLAClient(...)}``，以及其他辅助 client。

第四个参数 ``components`` 指定要初始化的服务名称。``None`` 表示全部服务，普通 CLI 会传入这个值。Dashboard 根据 ``dashboard.runtime_components`` 得到两个子集，每个 component 都必须显式声明 ``scope: "shared"`` 或 ``scope: "unique"``。Dashboard 先初始化一次 shared components，再为每个新的环境实例初始化 unique components。两次都调用同一个钩子，最后合并返回的 ``runtime_kwargs``。在 LIBERO 中，这两个子集分别是 ``{"vla", "sam3"}`` 和 ``{"env"}``。

实现应在启动任何服务前拒绝未知 component 名称。如果多个选中的本地服务初始化较慢，应先全部启动，再依次等待 ready，让初始化过程可以重叠。参考实现见 ``robots/libero/robot_spec.py`` 中的有序 component registry。

endpoint（``--env-endpoint``、``--am-endpoint``，以及 LIBERO 的 ``--sam3-endpoint``）解析和环境专用服务命令，应放在拥有对应服务的钩子中。这些 spawner 应通过 ``rpent.robots.runtime.try_spawn_server`` 和 ``try_wait_server`` 组合，使各环境的状态事件、就绪失败和 owned daemon 清理保持一致；runner 不处理这些环境细节。参考模式见 ``robots/libero/robot_spec.py`` 和 ``robots/robocasa/robot_spec.py``。

可选的运行结果 finalizer
------------------------

``RobotSpec.finalize_run`` 是面向所有机器人、与具体 benchmark 无关的通用运行结束钩子。RoboCasa 是当前使用方，用它记录单 cell 评测结果，供后续结果统计和聚合。任何需要发布机器可读评测产物的机器人都可以注册该钩子。其默认值为 ``None``，不会改变 runner 行为。配置该钩子后，普通终端 runner 会在关闭 toolkit 前读取 ``toolkit.solved()``，完成运行时清理后再把结构化的 ``RunFinalizationContext`` 传给钩子。产物 schema 与文件名由钩子负责，RPent 只定义生命周期边界。

JSON 产物应使用 ``write_json_atomic``，避免中断写入留下不完整结果：

.. code-block:: python

   from rpent.evaluation import RunFinalizationContext, write_json_atomic

   def _finalize_run(context: RunFinalizationContext):
       return write_json_atomic(
           context.output_dir / "result.json",
           {
               "robot": context.robot_name,
               "task": dict(context.task_desc),
               "success": context.environment_success,
           },
       )

通过 ``RobotSpec(..., finalize_run=_finalize_run)`` 注册回调。该钩子目前只用于普通终端运行，Dashboard 不会调用。benchmark manifest、机器人专用 runtime 字段和聚合逻辑应继续放在机器人目录中，而不是共享 CLI。

.. _add-robot-testing:

6. 需要补充的测试
-----------------

新增机器人时，应提供离线 CPU 单元测试，以及通过公开 CLI 覆盖所有暴露工具的声明式 GPU 端到端套件。

测试放在哪里
~~~~~~~~~~~~

以下 ``myrobot`` 替换为新机器人的包名：

- ``tests/unit_tests/robots/myrobot/``：离线单元测试，覆盖配置解析、client 参数处理、工具分派，以及 runtime 启动和清理逻辑。用 fake 替代仿真器和模型，保证可在 CPU 上运行。
- ``tests/e2e_tests/suites/myrobot.toml``：声明式 GPU 端到端套件，声明安装组件、共享资产与权重路径、工具示例参数，以及一条或多条 ``[[cases]]``\ （每个动作模型后端对应一条用例）。示例见 ``tests/e2e_tests/suites/libero.toml``。

GPU 套件测什么
~~~~~~~~~~~~~~

通用 E2E 运行器（``tests/e2e_tests/runner.py`` 与 ``tests/e2e_tests/test_suites.py``）使用离线工具遍历 ``FunctionModel``，针对真实仿真器和模型服务驱动公开 CLI：

- 每个暴露的工具都会至少调用一次且无报错，随后调用 ``finish``；``tool-coverage.json`` 记录已执行、缺失和失败的调用。
- 至少执行一次真实环境动作并在 ``states.json`` 中无错误记录，生成预期观测工件，且测试结束后自己启动的 daemon 全部正常退出。
- 这种有界接入测试不要求任务成功。缺少可选模型权重、解释器或 endpoint 时会自动跳过对应用例。

离线测试使用 ``pytest tests/unit_tests/robots/myrobot -v`` 运行。干净环境下的 GPU 套件运行方式见 :doc:`testing` 和 ``tests/e2e_tests/run_gpu_suite.sh``。
