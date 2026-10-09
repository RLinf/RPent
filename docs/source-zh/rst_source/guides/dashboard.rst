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

使用 ``claude_code`` 或 ``codex`` 时，首次输入会预填任务指令，可编辑后按 Enter 提交，也可直接提交。运行中输入消息可指导智能体：Claude Code 会先中断当前轮次再接收消息，Codex 则在当前轮次中接收消息并调整后续操作。输入 ``/help`` 查看命令，输入 ``/quit``、``/exit`` 或 ``/q`` 结束会话。

使用 ``api`` 规划器时，所选任务会自动开始；智能体回复后，终端接受后续指令并保留已有对话。输入 ``/exit`` 退出。例如：

.. code-block:: bash

   rpent --robot libero --suite libero_10_task --task 0 --seed 1 \
     --planner api --model anthropic:claude-opus-4-8 --interactive

模型认证配置见 :doc:`configure_planner`。``--interactive`` 不能与 ``--dashboard`` 同时使用。单臂 Franka 和双臂 Franka 的普通运行需要使用操作员终端；双臂探索中的交互方式见 :doc:`../real_world_robots/dual_franka`。

.. _dashboard-usage:

Dashboard
---------

Dashboard 支持 ``api``、``claude_code`` 和 ``codex`` 规划器。在命令行传递 ``--planner`` 与 ``--model``，配置方式和普通运行一致，详见 :doc:`configure_planner`。

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

``--dashboard`` 不能与 ``--interactive`` 或 ``--env-endpoint`` 同时使用；外部 ``--vla-endpoint`` 和 ``--sam3-endpoint`` 服务仍然可用。使用 ``--dashboard-language zh-cn`` 可切换中文 UI。

查看结果与结束会话
~~~~~~~~~~~~~~~~~~~~~~~~~~~

任务结束后，可回看动作时间线和视频；任务成功标准见对应环境页。保存路径会显示在运行输出中，规划器对话保存在该目录的 ``transcript_*.json`` 中。关闭网页不会停止后端服务；在启动终端按 Ctrl+C 结束会话。

默认只监听 ``127.0.0.1``。远程访问建议使用 :doc:`advanced_deployment` 中的 SSH 端口转发。
