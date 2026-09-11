Dual Franka 真机调试与 PhysicalAgent 对齐（审核记录）
====================================================

本文档记录双臂 Franka 真机复现时的运行约定、手动单技能测试方法、日志检查方式，
以及当前代码中为了对齐旧 PhysicalAgent 部署而保留的特化行为。

它不是通用的双臂 Franka 入门文档。通用安装、Ray 集群、标定格式和基础启动流程请先阅读
:doc:`dual_franka`。

适用范围
--------

这份文档面向以下场景：

* 已经按照双臂 Franka 文档安装好 RPent、RLinf、franky/libfranka、RealSense、
  夹爪和 Ray。
* 需要复现旧 PhysicalAgent clean-desk / dirty-clean sorting 真机任务。
* 需要在不启动 planner 的情况下单独测试某个 primitive、VLA segment、SAM3
  segmentation 或 back-projection。
* 需要隔离“控制机器人用的 Codex 记录/配置/本地 memory”和日常辅助编程用的 Codex
  环境。

当前 live deployment 里的特化行为包括：

* agent 主要看配置中的 inline camera；当前实验台默认是 D455。
* 其他相机作为 artifact 保存，可以在需要时单独查看。
* ``back_project`` 和 ``segment`` 可以对注册的 RGBD projection view 工作，返回标注图
  给 agent/operator 检查。
* agent 看到的 TCP、projection point、``move_delta`` 和 ``rotate_delta`` 都使用统一
  ``right_base`` 世界坐标系。
* 长任务中如果关节健康度进入 warning/critical，planner 可以调用
  ``recover_joint_posture``；该 recovery 会尽量保持两侧夹爪开合状态。
* clean-desk VLA checkpoint 仍使用旧部署训练时的固定 policy prompt。VLA tool 接收的
  prompt 会被记录进日志，但当前 live 复现会用固定 prompt 覆盖实际 policy 输入。

目录和脚本
----------

相关文件位置如下：

.. code-block:: text

   robots/dual_franka/config/example.yaml
       当前默认双臂 Franka 配置。包含真实相机 serial、夹爪连接、workspace、
       projection_views、agent_observation 和 joint_health 阈值。

   scripts/rpent_live_env.sh
       live shell 环境入口。需要 source，不能直接执行。

   scripts/run_dual_franka_task3_codex.sh
       使用隔离 Codex 环境启动 dirty-clean sorting runner。

   scripts/run_dual_franka_interactive.sh
       与上面类似，但打开 RPent interactive 模式，允许 operator 在 planner turn
       之间输入文字反馈。

   scripts/dual_franka_manual_call.py
       不经过 planner，直接调用一个 primitive / read-only tool 的手动测试入口。

   scripts/run_manual_skill.sh
       manual call 的 convenience wrapper，会先 source ``rpent_live_env.sh``。

   .codex-rpent-live.example/config.toml.example
       项目内 Codex 隔离配置模板，不包含 key。

环境隔离
--------

live robot run 不应该复用日常辅助编程的 ``~/.codex`` 记录、memory、缓存和配置。每个真机
shell 先执行：

.. code-block:: bash

   cd /home/raojiaji/nieyi/franka_port_test/RPent
   source scripts/rpent_live_env.sh

该脚本会设置：

* ``CODEX_HOME=$RPENT_REPO_ROOT/.codex-rpent-live``
* ``RPENT_LIVE_MEMORY_DIR=$CODEX_HOME/memory``
* ``RLINF_REPO_PATH=$RPENT_TEST_ROOT/RLinf``
* ``PYTHONPATH=$RLINF_REPO_PATH:$RPENT_REPO_ROOT:$PYTHONPATH``
* 默认 env/VLA/SAM3 endpoint：``6001``、``6000``、``8114``
* 默认 planner/model/task 参数。

``.codex-rpent-live/`` 已经被 gitignore。真实 API key、auth、history、cache 都只能放在该
私有目录或 shell 环境变量里，不要提交到仓库。

.. warning::

   ``scripts/rpent_live_env.sh`` 当前默认 calibration path 指向旧 PhysicalAgent 项目：

   ``/home/raojiaji/nieyi/physicalagent/physical_agent/envs/dual_franka/calibration/hand_eye_calibration.json``

   这是为了让新 RPent 复现时使用旧日志对应的 hand-eye / base-frame 参数。换机器、换桌面
   或换标定后必须覆盖 ``RPENT_CALIBRATION_PATH``。

手动单技能测试
--------------

``scripts/dual_franka_manual_call.py`` 是真机调试最重要的入口之一。它不启动 Codex runner，
只调用一个已注册 tool/primitive，并把 ``result.json``、``states.json`` 和相机 artifact
写入 output dir。

列出当前注册技能：

.. code-block:: bash

   scripts/run_manual_skill.sh --list-primitives

查看某个技能的 schema：

.. code-block:: bash

   scripts/run_manual_skill.sh --schema segment

从 schema 生成一个最小 example：

.. code-block:: bash

   scripts/run_manual_skill.sh --example move_delta

读取当前状态并保存图像 artifact：

.. code-block:: bash

   scripts/run_manual_skill.sh --primitive view_env_state

完整 reset 到配置初始姿态：

.. code-block:: bash

   scripts/run_manual_skill.sh --primitive reset

测试一个小的右臂位移：

.. code-block:: bash

   scripts/run_manual_skill.sh \
     --primitive move_delta \
     --params '{"arm":"right","delta_xyz":[0.01,0,0]}'

测试 SAM3 + projection：

.. code-block:: bash

   scripts/run_manual_skill.sh \
     --primitive segment \
     --params '{"camera":"d455","prompt":"cardboard box","target_name":"box"}'

测试 VLA segment。env 和 VLA 服务必须已经运行；VLA 可能持续较久，所以 timeout 要放长：

.. code-block:: bash

   scripts/run_manual_skill.sh \
     --primitive vla_right_grasp \
     --params '{"prompt":"grasp the next task-allowed object","max_chunks":10}' \
     --timeout-s 1800

.. note::

   手动测试不再使用 ``resources/*.json`` preset。技能列表和参数 schema 的唯一来源是
   ``robots/dual_franka/tools.py`` 中的 ``TOOLS_SPEC``。这样不会出现“tool registry”和
   “样例 JSON 文件”两套事实来源。

完整 runner
-----------

dirty-clean sorting 的 wrapper：

.. code-block:: bash

   scripts/run_dual_franka_task3_codex.sh

interactive 版本：

.. code-block:: bash

   scripts/run_dual_franka_interactive.sh

完全展开后等价于：

.. code-block:: bash

   OUT=/home/raojiaji/nieyi/franka_port_test/RPent/logs/$(date +%Y%m%d-%H%M%S)-dirty-clean-t3
   export RLINF_REPO_PATH=/home/raojiaji/nieyi/franka_port_test/RLinf
   export PYTHONPATH=/home/raojiaji/nieyi/franka_port_test/RLinf:/home/raojiaji/nieyi/franka_port_test/RPent:${PYTHONPATH:-}
   export CODEX_HOME=/home/raojiaji/nieyi/franka_port_test/RPent/.codex-rpent-live

   .venv/bin/python -m rpent.cli.main \
     --robot dual_franka \
     --task-id 3 \
     --planner codex \
     --model gpt-5.5 \
     --reasoning-effort medium \
     --memory-profile local \
     --memory-dir /home/raojiaji/nieyi/franka_port_test/RPent/.codex-rpent-live/memory \
     --env-endpoint http://127.0.0.1:6001 \
     --vla-endpoint http://127.0.0.1:6000 \
     --sam3-endpoint http://127.0.0.1:8114 \
     --robot-config robots/dual_franka/config/example.yaml \
     --calibration-path /home/raojiaji/nieyi/physicalagent/physical_agent/envs/dual_franka/calibration/hand_eye_calibration.json \
     --output-dir "$OUT"

如果要换 endpoint、task、model 或 output dir，优先在 shell 里覆盖 ``RPENT_*`` 环境变量，
或者直接给 wrapper 追加命令行参数。

人工干预
--------

RPent 当前有两条人工干预路径：

``--interactive``
   终端交互模式。operator 可以在 planner turn 之间输入文字反馈，例如指出目标错了、
   任务状态已变化、或者要求停止。

``--dashboard``
   dashboard 模式。支持 queued messages、withdraw、在 tool 边界 interrupt，以及替换任务。
   真机长任务中如果需要更细的 operator 控制，优先使用 dashboard。

旧 PhysicalAgent 的文件队列 operator console（``start``、``success``、``failure``、
``done``、``quit``）没有被直接迁移。后续如果还需要完全复现旧交互习惯，可以把它做成 RPent
dashboard/interactive 的薄 adapter，而不是重新引入第二套调度系统。

日志检查
--------

每次真机运行后，建议至少检查这些文件和字段：

``camera_meta.json``
   确认 ``agent_observation.inline_cameras``、``auxiliary_cameras``、
   ``projection_views``、相机 serial 和 calibration key 是否符合本次机器配置。

``states.json`` / tool result
   确认 ``coordinate_frame`` 是 ``right_base``，左右臂 ``tcp_pose`` 是否已经统一到
   ``right_base``。同时检查是否保留了 ``raw_tcp_pose`` 和 ``raw_tcp_pose_frame``，便于回溯。

``view_env_state`` result
   确认 ``images`` / ``image_block_order`` 是本次期望给 agent 直接看的视角，其他视角是否出现在
   ``artifact_images`` 和 ``image_<view>_path``。

``segment`` / ``back_project`` result
   确认返回了 overlay/annotated image，并检查 marker/mask 是否真的落在目标物体或目标容器内部。
   ``selection_valid=true`` 只是几何边界检查通过，不等于语义点选正确。

VLA result
   检查 ``requested_prompt``、``effective_policy_prompt``、``prompt_overridden``。当前
   clean-desk checkpoint 应该看到 prompt 被覆盖为旧部署固定 policy prompt。

``joint_health``
   检查 warning/critical 是否合理。如果 contact-rich 任务中频繁因为 ``joint_contact`` 或
   ``last_motion_errors`` 触发 warning，需要把这些信号从 warning 降级成 summary-only，或做成
   yaml 可配置。

当前特化与后续清理
------------------

当前代码里有几类为了复现而保留的特化：

* ``robots/dual_franka/config/example.yaml`` 更像当前实验台部署配置，而不是通用 example。
  后续应该拆出单独的 lab profile。
* ``tasks.py`` 中 task 1/3 明确绑定 clean-desk、dirty-clean、D455、纸箱/金属篮、蛋挞 dirty
  marker。这些应该作为 optional demo tasks 保留，不应污染 robot-wide prompt。
* ``tools.py`` 中 clean-desk VLA prompt override 是为了对齐旧 checkpoint 分布。后续更干净的
  做法是把它移入 checkpoint profile、task profile 或 robot config。
* ``env_server.py`` 目前会访问 RLinf private internals 来实现 fresh state、wrapped observation
  和 gripper-preserving joint recovery。后续应把这些能力下沉成 RLinf public API。
* RLinf ``franky_controller.py`` 的阻抗/步长默认值已对齐旧 PhysicalAgent 部署。更通用的做法是
  保持 upstream 默认值，并通过环境变量或 per-robot profile 覆盖当前机器参数。

安全检查清单
------------

启动完整任务前建议按顺序确认：

1. 两台 Franka、夹爪和相机物理状态正常，操作员在急停附近。
2. Ray 集群是本次需要的集群，旧进程和错误端口已清理。
3. ``RLINF_REPO_PATH`` 指向当前测试用 RLinf checkout。
4. ``RPENT_CALIBRATION_PATH`` 指向本次机器正确的 hand-eye bundle。
5. 先用 manual call 测 ``view_env_state``，确认 D455/相机 artifact 正常。
6. 再单测 ``segment`` 或 ``back_project``，确认投影点和 overlay 正常。
7. 再单测 ``open_gripper`` / ``close_gripper`` / 小幅 ``move_delta``。
8. 最后才跑 VLA segment 或完整 task。
