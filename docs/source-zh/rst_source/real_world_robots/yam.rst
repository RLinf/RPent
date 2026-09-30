YAM
===

.. Product image: https://i2rt.com/products/yam-6-dof-arm

.. figure:: https://i2rt.com/cdn/shop/files/st0_768396f1-edfb-4839-96c1-f7b5dffa214a.png?v=1788854436&width=1200
   :alt: YAM 机械臂全貌，包括底座、关节和夹爪
   :figclass: rpent-robot-figure
   :align: center

   用于真机实验的 YAM 机械臂。

YAM 使用公共 RobotSpec、Toolkit、RPC、探索生命周期和 MemoryManager。
双臂动作接口参照 RoboTwin，探索与记忆复用 LIBERO 流程。task 103 是人工原语
诊断台，task 104 将 VLA 预测与执行分开；它们不会启动 Agent 自动操作。

安装与真实依赖
--------------

使用 Linux、Python 3.11。Agent 机也需要 ``RPent[yam]``：投影代码使用
OmegaConf 和 RealSense SDK 的计算函数，不连接相机。控制机和推理机使用下面
固定的 RLinf YAM 提交；它提供可取消的移动接口、运动学以及官方
``openpi.get_model`` 加载器需要的 ``pi05_yam_joint`` 数据与策略变换。

.. code-block:: bash

   git clone https://github.com/scilwb/RLinf.git /path/to/RLinf
   git -C /path/to/RLinf checkout --detach 4c65548e7ade32b13ae101211f588a34a9f98305
   cd /path/to/RPent
   uv venv --python 3.11
   source .venv/bin/activate
   export RPENT_REPO_ROOT="$PWD"
   export RPENT_RLINF_ROOT=/path/to/RLinf

控制机同时安装 RPent、RLinf、固定的 i2rt SDK，以及与 ``ruckig`` 兼容的构建后端：

.. code-block:: bash

   uv pip install --torch-backend cpu \
     --build-constraints "$RPENT_RLINF_ROOT/requirements/embodied/envs/yam-build-constraints.txt" \
     -e '.[yam]' -e "$RPENT_RLINF_ROOT" \
     -r "$RPENT_RLINF_ROOT/requirements/embodied/envs/yam.txt"
   uv pip check

推理机使用独立环境，安装固定的 OpenPI 提交和 CUDA 12.8 Torch：

.. code-block:: bash

   uv pip install --torch-backend cu128 \
     -e '.[yam]' -e "$RPENT_RLINF_ROOT[embodied]" \
     'rpent-openpi @ git+https://github.com/RLinf/openpi.git@a560f4dd8205b8423ecd4c8a0fabb5f54140b8a0' \
     -r "$RPENT_RLINF_ROOT/requirements/embodied/models/openpi.txt" \
     'torch==2.7.1' 'torchvision==0.22.1' 'torchcodec==0.5' \
     'tokenizers==0.22.2' 'numpy==1.26.4' 'opencv-python==4.11.0.86'
   uv pip check

``yam`` extra 将 ``huggingface-hub`` 限制为小于 1，因为 i2rt 1.1.2 要求
``click<8.2``。仅运行 Agent 时可使用 ``uv pip install -e '.[yam]'``。
标定、i2rt 模型网格、训练好的权重和统计文件需要另行提供；RPent 不包含这些资源。

机器人与任务配置
----------------

复制 ``robots/yam/config/example.yaml`` 到版本控制之外，填写相机序列号、
外参、桌面模型、已确认的 reset/home 和现场控制参数。示例中的泊车未启用，
必须填写实测位置并启用 ``park_on_close`` 才能启动服务。两种姿态均需填写
``left_qpos/right_qpos`` 各七维，以及 ``duration_s``、``max_joint_delta``、
``tolerance``、``timeout_s``。不要照搬其他现场的关节位置。
升级时保留已验证的伺服、碰撞、重力补偿设置。

世界坐标系为 ``left_base``；右臂通过实测双基座外参转换。``top`` 是固定相机，
``left/right`` 是腕部相机。RGBD 与腕部 FK 使用同次采集。示例中的三路相机
均采用 640x480、30 Hz，在连接机械臂之前启动并预热。

任务指令注册在 ``robots/yam/tasks.py``。``task_name``、``seed`` 和
``max_episode_steps`` 由命令行传入。ENV 和 Agent 默认使用 ``TASK_INSTRUCTIONS``
中同一份注册指令。普通终端运行可用 ``--task-language`` 覆盖；Dashboard 要求
使用所选任务的注册指令。切换任务前，先安全关闭旧 ENV，再启动对应任务。

在后续使用的各个终端中，将 ``TASK_NAME`` 设置为所选的已注册任务名：

.. code-block:: bash

   export TASK_NAME='your-registered-task'

独立服务
--------

在控制机已准备好的虚拟环境运行：

.. code-block:: bash

   python -m robots.yam.env_server --robot-config /path/to/robot.yaml \
     --task-name "$TASK_NAME" \
     --seed 0 --max-episode-steps 1000 \
     --transport socket --host 127.0.0.1 --port 8110

服务进程刚启动时不连接电机。下面首次人工 ``status`` 会初始化相机及机械臂，
是明确的硬件启动动作；执行前应准备好现场。程序只想检查是否启动时用
``env.is_started``，不要调用可能初始化硬件的状态读取接口。

.. code-block:: bash

   python -m robots.yam.operator_control --robot-config /path/to/robot.yaml \
     --endpoint socket://127.0.0.1:8110 --event status
   python -m robots.yam.operator_control --robot-config /path/to/robot.yaml \
     --endpoint socket://127.0.0.1:8110 --event reset_pose
   # 从 status 复制当前 episode_id；摆场就绪必须是当前现场事实。
   python -m robots.yam.operator_control --robot-config /path/to/robot.yaml \
     --endpoint socket://127.0.0.1:8110 --episode-id CURRENT_ID \
     --event ready --note '本回合场景已恢复，可以执行'

在推理机激活上面安装好的环境。共享模型服务使用官方统一的 RLinf OpenPI
加载接口：

.. code-block:: bash

   export RPENT_RLINF_ROOT=/path/to/RLinf
   python -m rpent.robots.components.pi05_vla_server \
     --embodiment yam \
     --model-path /path/to/yam-checkpoint \
     --norm-stats-path /path/to/norm_stats.json \
     --transport socket --host 127.0.0.1 --port 8220

仅在 checkpoint 自带预期统计量时省略 ``--norm-stats-path``。
通过 SSH 把 VLA 转发到 Agent/控制机，或显式绑定可信内网接口并填写对应地址。
不要向不可信客户端暴露 pickle RPC。服务返回完整 30 步绝对 qpos14，每次只执行
前 5 步；布局为左六关节、左夹爪、右六关节、右夹爪。夹爪 0 闭、1 开，
RGB 顺序固定为 ``top/left/right``。契约检查会拒绝不兼容服务。

Dashboard 与终端探索
--------------------

选择可用的 planner 模型，并按部署环境填写地址与路径。将下面的 ``MODEL_ID``
替换为已配置的 Codex 后端支持的模型，并选择该模型支持的推理档位。

.. code-block:: bash

   rpent --robot yam --dashboard --explore --planner codex \
     --model MODEL_ID --reasoning-effort low \
     --robot-config /path/to/robot.yaml \
     --env-endpoint socket://127.0.0.1:8110 \
     --vla-endpoint socket://127.0.0.1:8220 \
     --memory-profile local --memory-dir /path/to/memory/yam \
     --max-episode-steps 1000 \
     --dashboard-host 127.0.0.1 --dashboard-port 8090

``max-episode-steps`` 必须与 ENV 一致。在浏览器所在电脑执行：

.. code-block:: bash

   ssh -N -L 8090:127.0.0.1:8090 USER@CONTROL_HOST

打开 ``http://127.0.0.1:8090``，发送 ``/rpent-task <task-name> 0``，将
``<task-name>`` 替换为与 ENV 一致的已注册任务名。
浏览页面不会上电；
每次任务检查 ENV，VLA 会话内共享，两者均不由 Dashboard 关闭。
手动原语与 Agent 调用互斥；手动结果随下一条 Agent 消息交接。
``/continue`` 可让 Agent 读取新的人工作业就绪或裁决回执，也可继续就绪、
未终止且无停止锁的 episode。就绪或裁决回执本身不授权运动。Agent 等待人工时，
程序自动续跑保持暂停，直到显式 ``/continue`` 或新 episode；不会越过已完成的
finish 或未完成的手动动作。

终端运行时去掉 Dashboard 参数，增加 ``--task-name "$TASK_NAME"``。
纯原语模式用 ``--without-vla`` 替代 VLA 地址。

结果、记忆与退出
----------------

仅当前 episode 的 ``eval_success=True`` 代表任务成功。人工终端以
``--event success/failure/abort``、当前 episode ID 和证据说明登记结果。
``--command /done`` 等价于 ready；``/success``、``/failure``、``/abort``
分别等价于对应事件。网页会提示在人工终端登记这些回执，不把它们交给 LLM。
``reset`` 消费 ready 并建立新回合，不代表回 home 或已恢复实物场景。
缺少就绪或裁决时立即返回非终止 ``pending``；新 episode 真正建立才计一次尝试。

探索使用共用的 ``_internal/inbox``、``task-specific``、``task-family``、``global`` 记忆结构。
正常结束的失败运行仍可贡献经验；程序报错或 Dashboard 切换任务时，笔记保留
在 inbox 中，不自动发布。成功 recipe 保留当前已核实成功回合中实际发出的动作，
包括部分执行的动作；实际执行情况可查对应步骤记录。通过 ``result.json``、
transcript、session 记录、recipe/audit 与更新后的 ``MEMORY.md`` 区分证据；Agent 正常结束、接口测试通过
或原语成功都不能证明完整任务成功。

停止或关闭 Dashboard 只请求保持。处理持物、检查回位路径后在人工终端退出 ENV：

.. code-block:: bash

   python -m robots.yam.operator_control --robot-config /path/to/robot.yaml \
     --endpoint socket://127.0.0.1:8110 --event shutdown

服务先回已配置 home、验证到位，再关闭输出；失败时保留运行时供人工恢复。
强杀进程或断电无法保证此流程。

诊断与部署迁移
--------------

先停止 Dashboard Agent，再打开诊断台；ENV 必须已启动。诊断动作前用人工
``--event start`` 消费 ready，诊断台自身不会创建就绪回执。

.. code-block:: bash

   python -m robots.yam.manual --task-name "$TASK_NAME" --task-id 103 \
     --env-endpoint socket://127.0.0.1:8110 --max-episode-steps 1000
   python -m robots.yam.manual --task-name "$TASK_NAME" --task-id 104 \
     --env-endpoint socket://127.0.0.1:8110 --max-episode-steps 1000 \
     --vla-endpoint socket://127.0.0.1:8220 --output-dir /path/to/diagnostics

每行一个 JSON：``{"tool":"status"}`` 查询状态。103 支持 move_to、rotate_wrist、
set_gripper、release，通过 ``arguments`` 传参。104 先输入
``{"tool":"infer"}`` 保存不驱动机械臂的预测，再显式输入
``{"tool":"execute","arguments":{"use_length":5}}``，最多执行一次。
跨 episode、动作计数变化、关节位置改变或预测超过 30 秒均拒绝；超时导致执行
结果不确定时禁止重放。``quit`` 只保持，不松爪、回 home 或生成成功记忆。

旧 ``exploration_status`` 调用统一改为 ``status()``；返回顶层原生 episode
字段及 ``reason/can_continue``，不保留永久别名。
``move_to(arm, xyz, quat, gripper, substeps)`` 保留实际使用的 ``xyz_bounds``：
只在任务有效区域内做有限候选搜索，不是遍历连续圆形内所有点；substeps 不删除
必要路径点。``rotate_wrist`` 支持 gripper，开合统一使用
``set_gripper(arm, val, steps)``、``release(arm, val=1, steps)``，不增加
open/close 别名。xyz 单位米，坐标系 left_base，四元数顺序 wxyz。

``--robot-config`` 入口只接受 YAML；升级前需转换旧 JSON 配置。
源码和 wheel 安装均包含机器人模块；设置 ``RPENT_REPO_ROOT`` 指向所需工作区，
用于存放日志、GUIDE 和记忆。
