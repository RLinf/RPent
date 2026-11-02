交互使用
========

你可以在终端输入指令，也可以通过 Dashboard 在浏览器中提交任务、查看规划器输出、相机画面和动作记录。以下示例使用已完成 :doc:`../get_started/quickstart` 配置的 LIBERO；其他平台先按对应环境页完成安装。

.. _terminal-interaction:

终端交互（TUI）
---------------

在任务命令中添加 ``--tui`` 或 ``-t``，即可在终端输入指令。请使用支持交互输入的终端（TTY）。

.. code-block:: bash

   rpent --robot libero --suite libero_10_task --task 0 --seed 1 \
     --planner claude_code --model claude-opus-4-8 --tui

使用 ``claude_code``、``codex`` 或 ``deepseek_harness`` 时，首次输入会预填
任务指令，可编辑后按 Enter 提交，也可直接提交。Claude Code 会先中断当前轮次
再接收后续消息，Codex 在当前轮次中接收消息并调整后续操作。输入 ``/help``
查看命令，输入 ``/quit``、``/exit`` 或 ``/q`` 结束会话。

DeepSeek Harness 完成 :ref:`登录 <planner-deepseek-harness>` 后，可运行：

.. code-block:: bash

   rpent --robot libero --libero-type pro --suite libero_10_task --task 0 --seed 0 \
     --planner deepseek_harness --dsh-home "$DSH_HOME" \
     --model gpt-6.1-sol --reasoning-effort xhigh --tui

模型或工具忙碌时，后续消息会排队，发送消息不会打断动作。输入 ``/interrupt``
并按 Enter，等待确认暂停后，再提交消息即可继续。已接受的 ``finish`` 会结束
当前任务；继续会话仅限当前进程。

在输入为空时按 Ctrl-D，表示输入结束（EOF）并结束运行，无需再按 Enter。
若有预填或已输入的文字，先清空；空行按 Enter 会被忽略。Ctrl+C 会结束进程，
不表示请求可恢复的暂停。

使用 ``api`` 规划器时，所选任务会自动开始；智能体回复后，终端接受后续指令并保留已有对话。输入 ``/exit`` 退出。例如：

.. code-block:: bash

   rpent --robot libero --suite libero_10_task --task 0 --seed 1 \
     --planner api --model anthropic:claude-opus-4-8 --tui

模型认证配置见 :doc:`configure_planner`。``--tui`` 不能与 ``--dashboard`` 同时使用。真机配置及人工确认方式见 :doc:`../real_world_robots/franka` 和 :doc:`../real_world_robots/dual_franka`。

.. _dashboard-usage:

Dashboard
---------

Dashboard 支持 ``api``、``claude_code``、``codex`` 和 ``deepseek_harness`` 规划器。在命令行传递 ``--planner`` 与 ``--model``，配置方式和普通运行一致，详见 :doc:`configure_planner`。

启动并提交任务
~~~~~~~~~~~~~~~~~~~~~

加上 ``--dashboard`` 可启动长生命周期的本地 Dashboard Session。系统会自动选择一个空闲端口，并在终端输出访问 URL：

.. code-block:: bash

   rpent --robot libero --dashboard \
     --planner claude_code --model claude-opus-4-8

DeepSeek Harness 使用与终端相同的登录目录和模型设置：

.. code-block:: bash

   rpent --robot libero --dashboard --planner deepseek_harness \
     --dsh-home "$DSH_HOME" --model gpt-6.1-sol --reasoning-effort xhigh

Session 配置全部来自命令行，切换任务时沿用；启动新进程需重新传入这些参数。
打开地址后会直接进入实时监控。共享服务就绪后，输入以下命令启动 TaskRun：

.. code-block:: text

   /rpent-task libero_object_swap 2 0

每个 TaskRun 使用独立环境，VLA 和 SAM3 服务由 Session 复用。可通过新的 ``/rpent-task`` 启动或切换任务；运行中也可以发送消息引导智能体，并按 Esc 请求中断。在终端按 Ctrl+C 可结束 Session。

使用 DeepSeek Harness 时，忙碌期间的消息会排队，不会打断执行。只有智能体
消费消息后，界面才标记为已发送；消费前可以撤回。按 Esc 请求暂停，等待进入
空闲状态后再使用手动控制；智能体忙碌时禁用手动操作。提交新消息或重新提交
未发送消息即可继续，智能体会收到手动操作后的新观测。

如果执行、结果交付或清理无法确认，Session 会关闭，不启动排队的替换任务。
保存的运行文件不提供跨进程恢复能力。

``--dashboard`` 不能与 ``--tui`` 或 ``--env-endpoint`` 同时使用；外部 ``--vla-endpoint`` 和 ``--sam3-endpoint`` 服务仍然可用。使用 ``--dashboard-language zh-cn`` 可切换中文 UI。

查看结果与结束会话
~~~~~~~~~~~~~~~~~~~~~~~~~~~

任务结束后，可回看动作时间线和视频；任务成功标准见对应环境页。保存路径会显示在运行输出中，规划器对话保存在该目录的 ``transcript_*.json`` 中。关闭网页不会停止后端服务；在启动终端按 Ctrl+C 结束会话。

默认只监听 ``127.0.0.1``。远程访问建议使用 :doc:`advanced_deployment` 中的 SSH 端口转发。
