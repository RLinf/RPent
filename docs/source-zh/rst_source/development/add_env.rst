添加 env 后端
==============

在 RPent 中，*env 后端* 把机器人模拟器（gym / robosuite / 外部物理引擎）
包装成 RPC 服务，对上层暴露统一的 ``reset`` / ``step`` / ``chunk_step``
接口。本页介绍如何基于统一基类
（:mod:`rpent.tools.env_facade_base` / :mod:`rpent.tools.env_client_base`）
添加新的 env 后端，并对比现有三个后端（libero / robocasa / robotwin）的
差异与共有结构。

.. contents::
   :local:
   :depth: 2

总体格局
--------

.. list-table::
   :header-rows: 1
   :widths: 18 27 27 28

   * - 基类
     - ``RpcFacade``
     - **无** — 自实现 dispatch
   * - 动作语义
     - gym 5-tuple
     - robosuite 4-tuple
     - ``execute_action_chunk`` + ``apply_qpos_updates``
   * - 渲染
     - EGL（gym 内置）
     - EGL + 单渲染线程派发
     - 无 sim 渲染（外部物理引擎）
   * - session
     - 无
     - 无（env 层不需要）
     - 无
   * - meta 握手
     - 有
     - 有
     - **无**
   * - chunk_step
     - 有
     - 有
     - **无**（用 ``execute_action_chunk``）

**关键发现**：libero/robocasa 在 env 层走同一套 ``RpcFacade`` 模板；
robotwin 是异类（env 改用 ``execute_action_chunk``、不走 ``RpcFacade``）。
统一层必须把 robotwin 当一等公民抽象，而不是把它的契约塞进 gym-step
模子里。

状态缓存原则
------------

**核心原则**: env server 侧**不**缓存任何观察结果. ``_last_obs`` /
``_terminated`` 等缓存变量**只存在于 client 侧**. server 是无状态的
(除 env 本身的物理状态外), 每次请求都重新读 env.

**为什么**:

- **多 client 并发**: 多个 ``RpcClient`` 连到同一 server 时, 每个 client
  维护自己的缓存, 互不污染. server 缓存会被并发 client 互相覆盖.
- **server 重启恢复**: server 崩溃重启后, client 重新 ``reset`` 即可重建
  状态. server 缓存则会在重启后失效, 但 client 不知情, 继续用旧缓存.
- **职责清晰**: server 只负责"执行 env 动作并返回 obs", client 负责
  "决定如何使用 obs 并缓存". 缓存策略是 client 的事.

**具体到基类**（见 :class:`~rpent.tools.env_facade_base.EnvFacade` 和
:class:`~rpent.tools.env_client_base.EnvClient`）:

- ``EnvFacade.__init__`` **不**初始化 ``self.last_obs`` /
  ``self._terminated``
- ``EnvFacade.chunk_step`` 默认实现**不**写 ``self.last_obs`` /
  ``self._terminated``（基类直接抛 ``NotImplementedError``, 子类实现时也不
  应写）
- ``EnvClient.__init__`` 通过 ``self.reset()`` 初始化 ``self.last_obs``
- ``EnvClient.reset`` / ``step`` / ``chunk_step`` **写** ``self.last_obs``
  （client 侧缓存, 调用方可直接访问 ``client.last_obs``）

.. warning::

   当前 robocasa ``robots/robocasa/env_server.py`` **已部分违反此原则**
   （server 端 ``_last_obs`` 用于 chunk_step 内嵌渲染）。统一设计后,
   chunk_step 内嵌渲染改为基于本次 RPC 调用内的临时局部变量, 不写 self。

共有 RPC 接口
-------------

框架层
~~~~~~~~~~~~~~~~~~

- 继承 ``RpcFacade``（或自实现等价物），实现
  ``_dispatch(method, args, kwargs, *, session_id)``
- ``_dispatch`` 用**注册字典**（``self._rpc``）路由, 子类在
  ``_register_rpc`` 中注册自己的方法（替代 getattr 动态路由）
- ``serve(transport, host, port, parent_watch)`` + argparse
  ``--transport/--host/--port/--parent-watch``
- ``_to_numpy_tree(x)`` 递归转 numpy（torch tensor → ``cpu().numpy()``;
  dict/list/tuple 递归）
- 客户端包装 ``RpcClient``, 调用
  ``self._client.call("env.method", args=, kwargs=, timeout_s=)``

业务层（libero + robocasa 共有）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - 方法
     - 签名
     - 返回值
   * - ``env.get_env_meta``
     - ``()``
     - 启动参数快照 dict（client/server 握手）
   * - ``env.reset``
     - ``()``
     - obs 或 ``(obs, info)``
   * - ``env.step``
     - ``(flat_action)``
     - libero: ``(obs, rew, term, trunc, info)``;
       robocasa: ``(obs, reward, done, info)``
   * - ``env.chunk_step``
     - ``(actions, *, return_all_frames=False)``
     - 5 元组 ``(obs_or_list, reward, done, info, n_applied)``
   * - ``env.get_camera_meta``
     - ``(cam, h, w)``
     - 相机内参/外参 dict

客户端层共有
~~~~~~~~~~~~

- ``expected_meta`` 握手（启动后立刻校验 server 返回的 meta）
- 初始 ``reset()``
- ``_TIMEOUT_S`` 字典（按方法分配超时）
- ``last_obs`` 缓存（client 侧缓存, 调用方可直接访问 ``client.last_obs``,
  见 *状态缓存原则* 一节）

.. note::

   以下接口是 robocasa only, **不上提到基类**:

   - ``eef_pos`` / ``eef_quat`` / ``gripper_qpos`` client property（从
     ``last_obs`` 读 robocasa 特有字段）
   - ``_resolve_cam`` 别名机制（``CAM_ALIAS`` 字典）

功能表格
--------

**后端覆盖列**: L=libero, R=robocasa, T=robotwin. "L+R"表示二者共有但不
包括 robotwin; "R only"表示仅 robocasa. **当前真正三者共有的接口只有**
``reset`` / ``step`` / ``chunk_step`` / ``get_env_meta`` / ``render_camera`` /
``get_camera_meta`` / ``close`` **这一组.** 其余要么 L+R 共有、要么 R only /
T only, 统一设计建议把它们作为基类抽象方法或可选 hook.

.. list-table::
   :header-rows: 1
   :widths: 13 28 12 47

   * - 大类功能
     - 函数列表
     - 后端覆盖
     - 功能实现及细节
   * - **生命周期**
     - ``reset``
     - L+R+T
     - L/R reset env; T 用 ``reset_exact(seed)`` 替代（语义不同 — 重新生成
       确定场景）
   * - **生命周期**
     - ``step``
     - L+R
     - L 返回 gym 5-tuple ``(obs, rew, term, trunc, info)``; R 返回 robosuite
       4-tuple ``(obs, reward, done, info)``; T 无 step（改用
       ``execute_action_chunk``）
   * - **生命周期**
     - ``chunk_step``
     - L+R
     - L 返回 ``(obs_or_list, rew, term[chunk], trunc[chunk], info)``; R 返回
       ``(obs_or_list, reward, done, info, n_applied)``（多 n_applied）; T 无
       （可用 ``execute_action_chunk`` 适配）
   * - **生命周期**
     - ``close`` / ``offload``
     - L+R+T
     - L/R ``close()``; T ``offload(clear_cache=True)``
   * - **元信息握手**
     - ``get_env_meta``
     - L+R
     - 返回 ``{task_name, split, seed, camera_h, camera_w, ...}`` 供 client
       启动后校验; T 无此契约（无 expected_meta 握手）
   * - **状态访问（缓存）**
     - ``raw_obs`` / ``current_raw_obs``
     - L+R
     - server 侧不缓存（统一设计, 见状态缓存原则）; 客户端通过 ``last_obs``
       缓存避免额外 RPC
   * - **状态访问**
     - ``get_terminated`` / ``terminated``
     - L(client)+R(server)
     - L 客户端 ``terminated`` / ``truncated`` 从 obs 推断; R 有
       ``get_terminated`` RPC; T 用 ``get_episode_status().agent_valid``
   * - **状态访问**
     - ``get_action_dim``
     - R only
     - R 通过 ``env.action_dim`` 暴露; L/T 客户端硬编码或从 obs 推断
   * - **状态访问**
     - ``get_ep_meta``
     - R only
     - R 暴露 episode meta（lang 等）; L 用 ``get_task_language``; T 无
   * - **状态访问**
     - ``eef_pos`` / ``eef_quat`` / ``gripper_qpos``
     - R only（client property）
     - R 客户端从 ``last_obs`` 读这些字段; L/T 无对应属性
   * - **渲染（高层）**
     - ``render_camera``
     - L+R
     - L 高层包装（垂直翻转 + 默认尺寸）; R 客户端走 ``_resolve_cam`` 别名 +
       翻转; T 无渲染
   * - **渲染（元信息）**
     - ``get_camera_meta``
     - L+R
     - 返回相机内参/外参 dict; T 无
   * - **渲染（低层）**
     - ``render_raw``
     - R only
     - R 低层 ``sim.render(cam, h, w, depth)`` + 深度
       ``get_real_depth_map`` 归一化; L 走 gym render 不暴露 raw; T 无
   * - **渲染（变换）**
     - ``get_camera_transform``
     - R only
     - R 用 robosuite ``get_camera_transform_matrix`` 返回 ``inv(T)``
       （pixel→world）; L/T 无
   * - **渲染（反投影）**
     - ``world_map`` / ``world_xyz_at``
     - R only（client）
     - R 客户端用深度 + 变换矩阵反投影得到 per-pixel world xyz; L/T 无
   * - **成功判定**
     - ``check_success``
     - L+R
     - L 走 gym 的 term/trunc 信号; R 有 ``check_success`` RPC 调
       ``env._check_success()``; T 用 ``episode_status.eval_success``
   * - **成功判定**
     - ``grasp_contact``
     - R only
     - R 调 robosuite ``_check_grasp``, 返回 ``(bool, obj_name)``; L/T 无
   * - **成功判定**
     - ``get_success_criteria_text``
     - R only
     - R 用 ``inspect.getsource(_check_success)`` 提取成功条件源码 + OU
       helper; L/T 无
   * - **成功判定**
     - ``get_task_progress``
     - R only
     - R 用 ``sys.settrace`` 捕获 ``_check_success`` 中间局部变量
       （washed_time/turned_on 等）; L/T 无
   * - **动作构造**
     - ``reassemble_env_action``
     - R only
     - 把 ``PandaOmronKeyConverter.unmap_action`` 的 dict 结果按
       ``composite_controller._action_split_indexes`` 重组成 flat 12-d action;
       L/T 无（动作空间不同）
   * - **传输层**
     - ``serve`` / ``_dispatch`` / ``_to_numpy_tree``
     - L+R（T 自实现）
     - L/R 走 ``RpcFacade``; T 自实现 ``_dispatch`` 用显式 ``handlers`` 字典、
       不取 ``self._lock``、不带 ``session_id``
   * - **EGL 单线程**
     - ``render_loop`` / ``work_queue``
     - R only
     - R ``serve()`` 重写为单渲染线程派发, 保证 EGL context 留在同一线程;
       L 用基类 ``serve()``; T 无渲染
   * - **episode 状态**
     - ``get_episode_status`` / ``get_robot_state`` / ``capture_observation``
     - T only
     - T 用 ``episode_status.agent_valid`` 替代 reward/done;
       ``capture_observation`` 返回原生 obs dict; L/R 无对应物
   * - **运动规划**
     - ``plan_arm_path``
     - T only
     - T 外部运动规划接口（通过 env_server RPC）; L/R 无

基类代码
--------

统一基类已落地到 ``rpent/tools/`` 下两个文件, 子类（libero / robocasa /
robotwin 的 env facade 和 env client）应继承并实现抽象方法:

- :mod:`rpent.tools.env_facade_base` —
  :class:`~rpent.tools.env_facade_base.EnvFacade`（server 侧）
- :mod:`rpent.tools.env_client_base` —
  :class:`~rpent.tools.env_client_base.EnvClient`（client 侧）

.. note::

   基类只固化三者共有的接口（``reset`` / ``step`` / ``chunk_step`` /
   ``get_env_meta`` / ``close``，均为抽象方法，子类必须实现）+ 框架层
   （``__init__`` 初始化 ``self._rpc`` 注册字典、``_register_rpc`` 注册
   共有方法、``_dispatch`` 用注册字典 + ``self._lock`` 串行化）.
   后端特有的方法**不放基类**, 由子类自行定义并通过
   ``_register_rpc`` 注册.

各 env 接入迭代差异
-------------------

libero（最早接入）
~~~~~~~~~~~~~~~~~~

- 设计基线: gym 5-tuple ``(obs, rew, term, trunc, info)``, ``term/trunc``
  形状 ``[chunk_size]``
- 客户端跟踪 ``terminated`` / ``truncated`` + ``check_done``
- 单 env 维剥离（``_strip_obs``, ``num_envs=1``）, 让上游统一处理 batch 维
- VLA 直接读 gym obs 的 ``main_images`` / ``wrist_images`` /
  ``task_descriptions``
- 无 session 隔离、无 EGL 单线程、无 ``get_success_criteria_text`` 内省

robocasa（第二次迭代）
~~~~~~~~~~~~~~~~~~~~~~

- **EGL 单线程派发**: 多 GPU/EGL 环境下 ``serve()`` 重写为 ``work_queue`` +
  ``render_loop``, 保证 MuJoCo EGL context 留在同一线程
- **chunk_step 内嵌渲染**: 服务器端始终为最终 obs 渲染 3 个 VLA 相机
  （256x256）, ``return_all_frames=True`` 时每步额外渲染 agentview;
  客户端不再发 N×3 个 ``render_camera`` RPC, 把 env↔vla 每 chunk 的 RPC 数
  从 6 降到 2
- **``_snapshot_vla_frame``**: 把 3 个 ``<cam>_rgb`` 注入最终 obs, 但**不**
  写入 ``_last_obs``（避免 1.7MB RGB 污染状态缓存）
- **内省 RPC**: ``get_success_criteria_text`` 用 ``inspect.getsource`` 提取
  ``_check_success`` 源码 + OU helper; ``get_task_progress`` 用
  ``sys.settrace`` 捕获 ``_check_success`` 中间局部变量（washed_time /
  turned_on 等）, 让 agent 有进度反馈而不是只有最终 bool
- **``grasp_contact``**: 方向无关的抓取检测（robosuite ``_check_grasp``）,
  返回 ``(bool, obj_name)``
- **``reassemble_env_action``**: 把 ``PandaOmronKeyConverter.unmap_action``
  的 dict 结果按 ``composite_controller._action_split_indexes`` 重组成
  flat 12-d action
- **``RLDX_RESET_SEED``**: 复现 eval 场景的随机种子（``random.seed`` +
  ``np.random.seed`` + ``env.rng/seed``）
- **``world_map`` / ``world_xyz_at``**: 用 robosuite 相机变换矩阵 + OpenGL
  深度反投影得到 per-pixel world xyz, agent 可在像素级定位物体
- **客户端 ``CAM_ALIAS``**: ``agentview`` → ``robot0_agentview_left`` 等别名,
  统一跨后端相机命名

robotwin（PR #84，第三次迭代 — 异类路径）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **不走 ``RpcFacade``**: env_server 自实现 ``_dispatch``, 用显式
  ``handlers`` 字典路由, **不**取 ``self._lock``, **不**带 ``session_id``
  形参
- **无 gym step**: 动作执行改为
  ``execute_action_chunk(action_type, actions)`` +
  ``apply_qpos_updates(updates)`` + ``reset_exact(seed)``
- **无 ``get_env_meta`` 握手**: 客户端启动后不校验 server 配置, 假设兼容
- **状态变更毒化**: ``RoboTwinExecutionError`` + ``_fatal_error`` — 任何
  状态变更 RPC 失败后客户端永久不可用, 避免脏状态污染 episode
- **``episode_status.agent_valid``**: 替代 reward/done 的成功/有效判定,
  由外部物理引擎返回
- **两档超时**: ``READ_TIMEOUT_S=120``（读）/ ``STATE_CHANGE_TIMEOUT_S=600``
  （写）, 替代 ``_TIMEOUT_S`` 字典
- **``_validate_rlinf_runtime()``**: 导入时能力检查, 确保 RLinf runtime 可用
- **``env.offload(clear_cache=True)``**: ``finally`` 中显式释放, 区别于
  robocasa 的 ``close()``
- **``plan_arm_path``**: 外部运动规划接口, sim 后端没有对应物
- **无渲染**: 物理引擎外部, env_server 不提供任何 ``render_raw`` /
  ``render_camera`` / ``get_camera_transform``

迭代脉络总结
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - 维度
     - libero → robocasa
     - robocasa → robotwin
   * - 动作契约
     - gym 5-tuple → robosuite 4-tuple（去掉 term/trunc）
     - 4-tuple → ``execute_action_chunk``（去掉 reward/done, 改用
       episode_status）
   * - 渲染
     - EGL（gym 内置）→ EGL 单线程 + chunk_step 内嵌
     - sim 渲染 → 无渲染（外部物理）
   * - 内省
     - 无 → ``get_success_criteria_text`` / ``get_task_progress``
     - 深度内省 → ``get_episode_status`` / ``get_robot_state``
   * - 容错
     - 无 → ``grasp_contact`` 兜底
     - 兜底 → ``_fatal_error`` 毒化（强一致）
   * - 客户端校验
     - 无 → ``expected_meta`` 握手
     - 握手 → 无（假设兼容）
   * - 生命周期
     - ``close()`` → ``close()`` + ``offload``
     - ``close`` → ``offload(clear_cache=True)``

统一设计建议
------------

1. **基类** :class:`~rpent.tools.env_facade_base.EnvFacade` ``(RpcFacade)``:
   固化 5 个抽象方法（``reset`` / ``step`` / ``chunk_step`` /
   ``get_env_meta`` / ``close``）+ 框架层（``__init__`` 初始化
   ``self._rpc`` 注册字典、``_register_rpc`` 注册共有方法、``_dispatch``
   用注册字典 ``self._rpc`` + ``self._lock`` 串行化）。**状态缓存原则**:
   server 侧不缓存 ``_last_obs`` / ``_terminated``, 所有缓存只在 client 侧
2. **抽象方法**: ``reset``、``step``、``chunk_step``、``get_env_meta``、
   ``close``（共 5 个，子类必须实现）; **不要强求 robotwin 套 gym step** —
   robotwin 的 ``step`` 抛 ``NotImplementedError`` 让调用方走 ``chunk_step``
   （适配层把 ``execute_action_chunk`` 包装成 ``chunk_step``）。
   ``get_action_dim`` **不上提到基类**（由子类按需暴露 + 在
   ``_register_rpc`` 中注册）
3. **可选 hook**: ``chunk_step`` 作为基类抽象方法（子类必须重写, 不再提供
   默认循环实现）。``render_raw`` / ``render_camera`` /
   ``get_camera_transform`` / ``world_map`` / ``check_success`` /
   ``get_terminated`` / ``grasp_contact`` / ``get_success_criteria_text`` /
   ``get_task_progress`` / ``get_action_dim`` / ``get_ep_meta`` **不放基类**,
   由子类自行定义并通过 ``_register_rpc`` 注册
4. **``chunk_step`` 契约**: 采用 robocasa 的 5 元组
   ``(obs_or_list, reward, done, info, n_applied)``; ``done`` 由后端决定语义
   （term OR trunc OR success）; ``return_all_frames`` 作为标准 perf-vs-video
   开关; robotwin 通过适配层把 ``execute_action_chunk`` 包装成 ``chunk_step``
   （``n_applied = chunk_size``, ``done = episode_status.agent_valid``）
5. **客户端基类** :class:`~rpent.tools.env_client_base.EnvClient`:
   上提 ``expected_meta`` 握手 + 初始 ``reset``、``_TIMEOUT_S`` 字典
   （``default`` / ``env.reset`` / ``env.step`` / ``env.chunk_step``）、
   ``last_obs`` **client 侧缓存**（状态缓存原则）。
   ``eef_pos`` / ``gripper_qpos`` / ``_resolve_cam`` **不上提到基类**（由
   子类自行实现）
6. **robotwin 适配层**: 在 ``EnvFacade`` 子类内 delegate 到原生
   ``RoboTwinEnv``, 把 ``execute_action_chunk`` / ``apply_qpos_updates`` /
   ``reset_exact`` 包装成 ``env.execute`` / ``env.apply_state`` / ``env.reset``,
   对外暴露统一接口; ``episode_status.agent_valid`` 翻译为 client 侧的
   ``last_obs`` 缓存（由适配层在每次 RPC 后更新）
