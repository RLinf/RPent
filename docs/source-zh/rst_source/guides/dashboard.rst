交互使用
========

你可以在终端输入指令，也可以通过 Dashboard 在浏览器中提交任务、查看规划器输出、相机画面和动作记录。以下示例使用已完成 :doc:`../get_started/quickstart` 配置的 LIBERO；其他平台先按对应环境页完成安装。

.. _terminal-interaction:

终端交互（TUI）
---------------

在任务命令中添加 ``--interactive`` 或 ``-i``，即可在终端输入指令。请使用支持交互输入的终端（TTY）。

.. code-block:: bash

   rpent --robot libero --suite libero_10_task --task 0 --seed 1 \
     --planner claude_code --model claude-opus-4-8 --interactive

使用 ``claude_code``、``codex`` 或 ``gemini_cli`` 时，首次输入会预填任务指令，可编辑后按 Enter 提交，也可直接提交。输入 ``/help`` 查看命令，输入 ``/quit``、``/exit`` 或 ``/q`` 结束会话。

Claude Code、Codex 和 Gemini 的追加消息按输入顺序排队，每个完整的 agent turn 结束后发送一条，而不是每次工具调用结束后发送。使用 ``/interrupt`` 可中断当前 turn 并发送下一条排队消息；使用 ``/interrupt 新指令`` 则优先发送该指令，其余消息仍保留在队列中。这两种中断方式都会在原会话中继续。RPent 先取消并等待工具执行结束，再中断后端；只有确认停止后才发送新消息。中断失败时，消息保留在队列中，直到重试中断成功。已经完成的物理动作不会被撤销。

使用 ``api`` 规划器时，所选任务会自动开始；智能体回复后，终端接受后续指令并保留已有对话。输入 ``/exit`` 退出。例如：

.. code-block:: bash

   rpent --robot libero --suite libero_10_task --task 0 --seed 1 \
     --planner api --model anthropic:claude-opus-4-8 --interactive

模型认证配置见 :doc:`configure_planner`。``--interactive`` 不能与 ``--dashboard`` 同时使用。真机配置及人工确认方式见 :doc:`../real_world_robots/franka` 和 :doc:`../real_world_robots/dual_franka`。

.. _dashboard-usage:

Dashboard
---------

Dashboard 支持 ``api``、``claude_code``、``codex`` 和 ``gemini_cli`` 规划器。在命令行传递 ``--planner`` 与 ``--model``，配置方式和普通运行一致，详见 :doc:`configure_planner`。

启动并提交任务
~~~~~~~~~~~~~~~~~~~~~

加上 ``--dashboard`` 可启动长生命周期的本地 Dashboard Session。系统会自动选择一个空闲端口，并在终端输出访问 URL：

.. code-block:: bash

   rpent --robot libero --dashboard \
     --planner claude_code --model claude-opus-4-8

Session 配置全部来自命令行，打开地址后会直接进入实时监控。共享服务就绪后，输入以下命令启动 TaskRun：

.. code-block:: text

   /rpent-task libero_object_swap 2 0

每个 TaskRun 使用独立环境，VLA 和 SAM3 服务由 Session 复用。可通过新的 ``/rpent-task`` 启动或切换任务；运行中也可以发送消息引导智能体，并按 Esc 请求中断。在终端按 Ctrl+C 可结束 Session。

使用 Claude Code、Codex 或 Gemini 时，普通消息会排队等待当前 agent turn 完成。如需提前继续，可先提交消息再按 Esc：活动工具和后端 turn 停止后，才会发送下一条排队消息，其余消息保持原有顺序。成功执行 ``finish`` 后任务结束，不再发送排队消息。

``--dashboard`` 不能与 ``--interactive`` 或 ``--env-endpoint`` 同时使用；外部 ``--vla-endpoint`` 和 ``--sam3-endpoint`` 服务仍然可用。使用 ``--dashboard-language zh-cn`` 可切换中文 UI。

查看结果与结束会话
~~~~~~~~~~~~~~~~~~~~~~~~~~~

任务结束后，可回看动作时间线和视频；任务成功标准见对应环境页。保存路径会显示在运行输出中，规划器对话保存在该目录的 ``transcript_*.json`` 中。关闭网页不会停止后端服务；在启动终端按 Ctrl+C 结束会话。

默认只监听 ``127.0.0.1``。远程访问建议使用 :doc:`advanced_deployment` 中的 SSH 端口转发。
