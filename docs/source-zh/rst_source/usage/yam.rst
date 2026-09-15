YAM
===

YAM 使用公共 RobotSpec、Toolkit、RPC、探索生命周期和 MemoryManager。
双臂动作接口参照 RoboTwin，探索与记忆复用 LIBERO 流程。task 103 是人工原语
诊断台，task 104 将 VLA 预测与执行分开；它们不会启动 Agent 自动操作。

安装与真实依赖
--------------

控制机使用 Linux、Python 3.11。Agent 机只需要 RPent；控制机需要 RealSense、
MuJoCo 和现场 RLinf/i2rt 环境；推理机需要兼容的 RLinf YAM 策略与 OpenPI/Torch。

.. code-block:: bash

   cd /path/to/RPent
   uv venv --python 3.11
   source .venv/bin/activate
   uv pip install -e '.[yam]'
   export RPENT_REPO_ROOT="$PWD"
   export RPENT_RLINF_ROOT=/path/to/station-RLinf

本适配依赖 RLinf YAM fork 的 ``3554fd2c`` **加现场修改**，不能用未修改的
官方 RLinf 直接复现。部署应保留 fork 提交、工作区补丁、未跟踪运行时源码、
环境版本清单和现场配置。实际需要 ``YamControlRuntime`` 的 command/hold/move_to/
反馈接口、``I2RTYamBackendFactory``、``YamKinematicsAdapter``，以及
``openpi_rlinf`` 模型加载器和 ``pi05_yam_joint`` 数据、策略变换。
RPent 不自动安装或发布这些现场修改。还需提供标定、i2rt 模型网格、checkpoint
及 norm stats；本扩展不包含训练好的 YAM 权重。

配置现场与任务
--------------

复制 ``robots/yam/config.example.json`` 到版本控制之外，填写相机序列号、
外参、桌面模型、已确认的 reset/home 和现场控制参数。示例中的泊车未启用，
必须填写实测位置并启用 ``park_on_close`` 才能启动服务。两种姿态均需填写
``left_qpos/right_qpos`` 各七维，以及 ``duration_s``、``max_joint_delta``、
``tolerance``、``timeout_s``。不要照搬其他现场的关节位置。
升级时保留已验证的伺服、碰撞、重力补偿设置。

世界坐标系为 ``left_base``；右臂通过实测双基座外参转换。``top`` 是固定相机，
``left/right`` 是腕部相机。RGBD 与腕部 FK 使用同次采集；已验证现场的三路
相机均为 640x480、30 Hz，在连接机械臂之前启动并预热。

任务集中注册在 ``robots/yam/tasks.py``。分别准备 A/B 的现场 JSON，将
``task_name`` 与 ``TASK_INSTRUCTIONS`` 中对应的完整 ``task_language`` 写入：

* ``tabletop_cleanup_a``：左袋百事，右袋可口。
* ``tabletop_cleanup_b``：左袋可口，右袋百事。

左右均以 top 视角为准，三瓶全部入袋。移动遮挡勺子的碗，白勺入左侧白碗、
粉勺入右侧粉碗，最后两碗返回标记。勺子初始位置可以变化。任务名跨运行稳定，
每次证据标识唯一。Dashboard 会拒绝 A/B 指令不一致；切换任务前需安全关闭
旧 ENV，再用对应配置启动，不会静默改写正在运行的环境任务。

独立服务
--------

在控制机已准备好的虚拟环境运行：

.. code-block:: bash

   python -m robots.yam.env_server --config /path/to/task_a.json \
     --transport socket --host 127.0.0.1 --port 8110

服务进程刚启动时不连接电机。下面首次人工 ``status`` 会初始化相机及机械臂，
是明确的硬件启动动作；执行前应准备好现场。程序只想检查是否启动时用
``env.is_started``，不要调用可能初始化硬件的状态读取接口。

.. code-block:: bash

   python -m robots.yam.operator_control --config /path/to/task_a.json \
     --endpoint socket://127.0.0.1:8110 --event status
   python -m robots.yam.operator_control --config /path/to/task_a.json \
     --endpoint socket://127.0.0.1:8110 --event reset_pose
   # 从 status 复制当前 episode_id；摆场就绪必须是当前现场事实。
   python -m robots.yam.operator_control --config /path/to/task_a.json \
     --endpoint socket://127.0.0.1:8110 --episode-id CURRENT_ID \
     --event ready --note '本回合场景已恢复，可以执行'

在推理机激活已准备的 RLinf 策略环境，安装 RPent，设置该机的
``RPENT_RLINF_ROOT`` 后运行：

.. code-block:: bash

   python -m robots.yam.vla_server --model-path /path/to/yam-checkpoint \
     --norm-stats-path /path/to/norm_stats.json \
     --transport socket --host 127.0.0.1 --port 8220

仅在 checkpoint 自带预期统计量时省略 ``--norm-stats-path``。
通过 SSH 把 VLA 转发到 Agent/控制机，或显式绑定可信内网接口并填写对应地址。
不要向不可信客户端暴露 pickle RPC。服务返回完整 30 步绝对 qpos14，仅执行端
裁剪长度；布局为左六关节、左夹爪、右六关节、右夹爪。夹爪 0 闭、1 开，
RGB 顺序固定为 ``top/left/right``。契约检查会拒绝不兼容服务。

Dashboard 与终端探索
--------------------

下面是明确指定的现场启动配置，不是库内默认值。模型能否使用取决于安装的
planner 和账号；地址与路径按启动机器填写。该模型显式使用 ``low``，不设置
不支持的 ``none`` 推理档位。

.. code-block:: bash

   rpent --robot yam --dashboard --explore --planner codex \
     --model gpt-6-astra --reasoning-effort low \
     --env-endpoint socket://127.0.0.1:8110 \
     --vla-endpoint socket://127.0.0.1:8220 \
     --memory-profile local --memory-dir /path/to/memory/yam \
     --max-episode-steps 1800 --explore-attempts-per-session 50 \
     --explore-sessions 1 --max-turns 300 --planner-timeout-s 14400 \
     --dashboard-host 127.0.0.1 --dashboard-port 8090

``max-episode-steps`` 必须与 ENV 一致。在浏览器所在电脑执行：

.. code-block:: bash

   ssh -N -L 8090:127.0.0.1:8090 USER@CONTROL_HOST

打开 ``http://127.0.0.1:8090``，发送 ``/rpent-task tabletop_cleanup_a 0``。
B 组准备对应服务后发送 ``/rpent-task tabletop_cleanup_b 0``。浏览页面不会上电；
每次任务检查 ENV，VLA 会话内共享，两者均不由 Dashboard 关闭。
手动原语与 Agent 调用互斥；手动结果随下一条 Agent 消息交接。
``/continue`` 仅继续当前就绪、未终止、无停止锁的 episode。自动续跑遵循同样
规则，并限制没有动作进展的循环；不会越过明确 finish、人工中断或未完成手动动作。

终端运行时去掉 Dashboard 参数，增加 ``--task-name tabletop_cleanup_a``。
纯原语模式用 ``--without-vla`` 替代 VLA 地址。要复现现场策略偏好，可发送：
“场景支持时使用 chunks=2、use_length=30，每次调用后观察。”这表示两次预测、
共请求 60 步，不是两步动作；更长的无语义复核时段可能包含提前松爪。

结果、记忆与退出
----------------

仅当前 episode 的 ``eval_success=True`` 代表任务成功。人工终端以
``--event success/failure/abort``、当前 episode ID 和证据说明登记结果。
``--command /done`` 等价于 ready；``/success``、``/failure``、``/abort``
分别等价于对应事件。网页会提示在人工终端登记这些回执，不把它们交给 LLM。
``reset`` 消费 ready 并建立新回合，不代表回 home 或已恢复实物场景。
等待每次最多 20 秒，返回非终止 ``pending``；新 episode 真正建立才计一次尝试。

使用官方 ``_internal/inbox``、``task_only``、``suite``、``global`` 记忆结构。
失败运行的有效笔记也可以合并；成功 recipe 仅包含当前已核实成功回合的动作。
A/B 入袋规则分别绑定任务。通过 ``result.json``、transcript、session 记录、
recipe/audit 与更新后的 ``MEMORY.md`` 区分证据；Agent 正常结束、接口测试通过
或原语成功都不能证明完整任务成功。

停止或关闭 Dashboard 只请求保持。处理持物、检查回位路径后在人工终端退出 ENV：

.. code-block:: bash

   python -m robots.yam.operator_control --config /path/to/task_a.json \
     --endpoint socket://127.0.0.1:8110 --event shutdown

服务先回已配置 home、验证到位，再关闭输出；失败时保留运行时供人工恢复。
强杀进程或断电无法保证此流程。

诊断与部署迁移
--------------

先停止 Dashboard Agent，再打开诊断台；ENV 必须已启动。诊断动作前用人工
``--event start`` 消费 ready，诊断台自身不会创建就绪回执。

.. code-block:: bash

   rpent --robot yam --task-name tabletop_cleanup_a --task-id 103 \
     --env-endpoint socket://127.0.0.1:8110 --max-episode-steps 1800
   rpent --robot yam --task-name tabletop_cleanup_a --task-id 104 \
     --env-endpoint socket://127.0.0.1:8110 --max-episode-steps 1800 \
     --vla-endpoint socket://127.0.0.1:8220 --output-dir /path/to/diagnostics

每行一个 JSON：``{"tool":"status"}`` 查询状态。103 支持 move_to、rotate_wrist、
set_gripper、release，通过 ``arguments`` 传参。104 先输入
``{"tool":"infer"}`` 保存不驱动机械臂的预测，再显式输入
``{"tool":"execute","arguments":{"use_length":30}}``，最多执行一次。
跨 episode、动作计数变化、关节位置改变或预测超过 30 秒均拒绝；超时导致执行
结果不确定时禁止重放。``quit`` 只保持，不松爪、回 home 或生成成功记忆。

旧 ``exploration_status`` 调用统一改为 ``status()``；返回顶层原生 episode
字段及 ``reason/can_continue``，不保留永久别名。
``move_to(arm, xyz, quat, gripper, substeps)`` 保留实际使用的 ``xyz_bounds``：
只在任务有效区域内做有限候选搜索，不是遍历连续圆形内所有点；substeps 不删除
必要路径点。``rotate_wrist`` 支持 gripper，开合统一使用
``set_gripper(arm, val, steps)``、``release(arm, val=1, steps)``，不增加
open/close 别名。xyz 单位米，坐标系 left_base，四元数顺序 wxyz。

在维护窗口替换明确代码文件，保留现场 JSON、标定、权重、数据和 memory；
更新启动参数并重新安装包，保存旧 bundle/patch 供回滚。源码和 wheel 均包含
机器人模块；设置 ``RPENT_REPO_ROOT`` 指向工作区，稳定日志、GUIDE 与记忆位置。
安装过程不会训练 VLA、上传 HF 或启动实机探索。
