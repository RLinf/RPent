使用 Dashboard
================

Dashboard 用于查看规划器输出、相机画面和动作记录，并在浏览器中提交任务。以下示例使用已完成 :doc:`../quickstart` 配置的 LIBERO；其他平台先按对应环境页完成安装。

启动并提交任务
---------------------

加上 ``--dashboard`` 可启动长生命周期的本地 Dashboard Session。系统会自动选择一个空闲端口，并在终端输出访问 URL：

.. code-block:: bash

   rpent --robot libero --dashboard \
     --planner claude_code --model claude-opus-4-8

Session 配置全部来自命令行，打开地址后会直接进入实时监控。共享服务就绪后，输入以下命令启动 TaskRun：

.. code-block:: text

   /rpent-task libero_object_swap 2 0

Dashboard 支持 ``api``、``claude_code`` 和 ``codex`` planner。在命令行传递 ``--planner`` 与 ``--model``，配置方式和普通运行一致，详见 :doc:`configure_planner`。

每个 TaskRun 使用独立环境，VLA 和 SAM3 服务由 Session 复用。可通过新的 ``/rpent-task`` 启动或切换任务；运行中也可以发送消息引导智能体，并按 Esc 请求中断。在终端按 Ctrl+C 可结束 Session。

``--dashboard`` 不能与 ``--interactive`` 或 ``--env-endpoint`` 同时使用；外部 ``--vla-endpoint`` 和 ``--sam3-endpoint`` 服务仍然可用。使用 ``--dashboard-language zh-cn`` 可切换中文 UI。

查看结果与结束会话
---------------------------

任务结束后，可回看动作时间线和视频；任务成功标准见对应环境页。保存路径会显示在运行输出中，规划器对话保存在该目录的 ``transcript_*.json`` 中。关闭网页不会停止后端服务；在启动终端按 Ctrl+C 结束会话。

默认只监听 ``127.0.0.1``。远程访问建议使用 :doc:`advanced_deployment` 中的 SSH 端口转发。
