LeKiwi
======

LeKiwi 是一个三轮全向移动底盘：三个 Feetech STS3215 轮电机挂在同一条串行总线上，外加一个前置 USB 摄像头。RPent 通过 RLinf 的 LeKiwi 真实世界环境控制它：规划器下发一个有界的机体速度 ``[x, y, theta]``，单位为底盘坐标系下的 m/s、m/s、deg/s，并在每次命令后读取实测轮速和同步的前置摄像头画面。

这里涉及两台计算机。机器人的板载计算机（通常是树莓派）持有舵机总线和摄像头，负责运行环境服务端；控制端（通常是同一网络下的笔记本）运行 RPent 和规划器。LeKiwi 的环境属于外部环境：Dashboard 会话只连接你给出的地址，永远不会自己拉起一个环境进程。

本文按顺序带你从一台已装好的机器人走到一次可运行的任务：准备机器人板载计算机、用一个 YAML 文件描述现场、启动环境服务端、从控制端接入。更早一步的硬件检查——电机编号、车轮检查、摄像头检查——见 `RLinf LeKiwi 指南 <https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/lekiwi.html>`_。

准备工作
--------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 组件
     - 要求
   * - 底盘
     - LeKiwi 移动底盘，包含三个 Feetech STS3215 轮电机、舵机控制板和套件电源。
   * - 摄像头
     - 安装在底盘前方的 UVC 摄像头，连接到机器人板载计算机。
   * - 机器人板载计算机
     - 带舵机控制板 USB 接口、内存足以运行环境服务端的 Linux 计算机，不需要 GPU。
   * - 控制端
     - 能通过网络访问机器人板载计算机的任何计算机，本身不需要任何硬件。

各部分如何协作
--------------

环境服务端持有硬件。它在自己的进程中组装 RLinf 的 ``RealWorldEnv``，直接打开舵机总线和摄像头，并提供 ``env.*`` 的 RPC 方法。板载计算机上不启动 Ray 集群：在这类小板上，调度器集群占用的内存会超过整机可用的内存。

一次 ``drive`` 调用会按请求的时长、以配置的控制频率驱动车轮，并在返回前停车，因此底盘只在调用执行期间运动。运行被中断、客户端崩溃或网络断开，都不会让底盘继续滑行。Dashboard 遥控改用 ``hold``：每次按键下发一个速度，服务端持续按该速度驱动，直到它过期或有停车命令到达。按住按键期间 Dashboard 会不断刷新，因此最后一次刷新后一秒它就失效；服务端把任何 ``hold`` 限制在两秒以内。

底盘自身没有成功信号，任务是否成功由规划器依据任务指令和每步记录中的证据判断。

准备机器人板载计算机
--------------------

使用 Linux 和 Python 3.11，安装带 ``lekiwi`` extra 的 RPent，该 extra 会一并安装 RLinf：

.. code-block:: bash

   cd /path/to/RPent
   uv venv --python 3.11
   source .venv/bin/activate
   uv pip install --torch-backend cpu -e '.[lekiwi]'
   uv pip check

已发布的 RLinf 版本尚未包含 LeKiwi 环境层，因此该 extra 固定到包含它的那个修订版本。``--torch-backend cpu`` 让 PyTorch 只装 CPU 版本，底盘只需要这些。

准备控制端
----------

在控制端以同样的方式安装 RPent：

.. code-block:: bash

   cd /path/to/RPent
   uv venv --python 3.11
   source .venv/bin/activate
   uv pip install -e '.[lekiwi]'
   uv pip check

控制端只需要能通过网络访问机器人板载计算机，端口与你启动服务端时使用的一致。

编写站点 YAML
-------------

在机器人板载计算机上运行检测脚本，生成包含检测到的串口和摄像头标识符的站点 YAML：

.. code-block:: bash

   rpent-lekiwi-detect --output /path/to/lekiwi-site.yaml

脚本不会让底盘运动：它按各个波特率对串行总线执行 ping，打开每个摄像头确认能取到一帧，并打印探测结果：

.. code-block:: text

   == LeKiwi site detection ==
   serial ports scanned: ['/dev/ttyACM0']
     /dev/ttyACM0: motor ids [7, 8, 9]
   wheel bus: /dev/ttyACM0
   camera usb-USB_Camera_USB_Camera_2505280051-video-index0: ok at 640x480
   wrote /path/to/lekiwi-site.yaml
   review before the first run:
     - robot.wheel_radius / robot.base_radius: measure with a ruler (m)
     - drive.max_linear_velocity / max_angular_velocity: policy; start low
     - drive.frame_size: camera output size; raise from 128x128 if needed
     - drive.step_frequency: control rate; 10 Hz suits the STS3215 bus

只探测其中一侧时可以加 ``--skip-motors`` 或 ``--skip-cameras``。生成的文件与 ``robots/lekiwi/config/example.yaml`` 结构相同：

.. code-block:: yaml

   robot:
     port: /dev/ttyACM0
     wheel_radius: 0.05
     base_radius: 0.125
     max_raw: 3000

   cameras:
     devices:
       front:
         serial: "<front-camera-serial>"
         type: webcam
         main: true

   drive:
     step_frequency: 10.0
     max_linear_velocity: 0.3
     max_angular_velocity: 60.0
     frame_size: [128, 128]

有两类值必须由人来定。几何参数 ``wheel_radius`` 和 ``base_radius``（单位为米）要用尺子量；``max_raw``（默认 3000）是舵机的原始速度上限。``drive`` 的速度上限则是你的速度包络：这是策略选择，探测不出来——从默认值起步，逐步放宽。下面的任务会进一步收紧每条命令，环境也会把每条命令裁剪到这些上限内。

生成结果与实际机器人不符时，直接修改这个文件：例如重命名前置摄像头的条目，或提高 ``frame_size`` 以使用摄像头原生支持的分辨率。

把这个 YAML 复制到控制端。两台机器都用 ``--robot-config`` 读取它：机器人板载计算机据此打开硬件，控制端据此获得速度上限和摄像头布局。

任务
----

任务注册在 ``robots/lekiwi/tasks.py`` 中，用 ``--task-id`` 选择。

.. list-table::
   :header-rows: 1
   :widths: 10 25 65

   * - ID
     - 任务
     - 规划器要做什么
   * - 0
     - ``drive_smoke_test``
     - 执行一次保守的前进、一次原地旋转，并在每次动作后检查摄像头画面；每次不超过 0.1 m/s、30 deg/s，持续最多 2 秒。
   * - 1
     - ``room_survey``
     - 按有界分段在房间中行进，并汇报摄像头看到的内容；每次不超过 0.2 m/s、45 deg/s，持续最多 3 秒。

首次运行请在清空的地面上使用任务 0。它会把整条链路走一遍——服务端、串口、摄像头、规划器——而不会让底盘移动很远。

启动环境服务端
--------------

在机器人板载计算机上，把服务端绑定到控制端可以访问的网卡：

.. code-block:: bash

   rpent-lekiwi-server \
     --robot-config /path/to/lekiwi-site.yaml \
     --transport http --host 0.0.0.0 --port 8120

它会打印供控制端接入的地址：

.. code-block:: text

   RPC server listening on http://0.0.0.0:8120

接触硬件之前，可以用 ``--print-config`` 打印解析后的 RLinf 配置（YAML 格式）并退出。用 Ctrl+C 停止服务端；它在退出时会停止轮子并松开力矩。

从控制端接入
------------

针对刚启动的服务端运行一个任务：

.. code-block:: bash

   rpent --robot lekiwi --task-id 0 \
     --env-endpoint http://ROBOT_HOST:8120 \
     --robot-config /path/to/lekiwi-site.yaml \
     --planner api --model MODEL_ID

``--env-endpoint`` 让它成为一次远程运行。省略它时，RPent 会在控制端本地拉起一个环境服务端，那只在舵机总线接在控制端时才有效。

要手动操控，可以启动一个不带规划器的 Dashboard 会话，在浏览器里驱动底盘：

.. code-block:: bash

   rpent --robot lekiwi --dashboard --planner manual \
     --env-endpoint http://ROBOT_HOST:8120 \
     --robot-config /path/to/lekiwi-site.yaml

LeKiwi 的环境是外部环境，所以 Dashboard 必须提供 ``--env-endpoint``。按住方向键行驶，松开即停车，用 ``End run`` 结束本次运行。录制视频与动作日志等完整说明见 :doc:`../guides/dashboard`。

工具与产物
----------

规划器可见三个工具：``drive(velocity, duration_s)``、``stop()`` 和只读的 ``view_env_state(step)``。``duration_s`` 每次调用最多 10 秒，请求的速度会被裁剪到站点配置的上限内。

默认输出目录是 ``logs/<时间戳>_lekiwi_t<task_id>/``，也可以用 ``--output-dir`` 指定。每一步都会保存同步的前置摄像头帧 ``front.png`` 及其元数据 ``camera_meta.json``；``view_env_state`` 返回该步记录和图像。Dashboard 会话还会为手动驾驶的每一段保存 ``teleop_NNN.mp4`` 和 ``teleop_actions.jsonl``。

故障排查
--------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - 现象
     - 原因与处理
   * - 报 ``rlinf.envs.real.lekiwi`` 的 ``ModuleNotFoundError``
     - 已安装的 RLinf 不含 LeKiwi 环境层。按上文安装包含它的修订版本。
   * - 控制端连不上
     - 服务端没有运行、绑定在 ``127.0.0.1``，或端口没有放通。改为绑定
       ``0.0.0.0``，并在两台机器之间开放该端口。
   * - 报 ``robot.port is required``
     - 站点 YAML 缺少 ``robot.port``。用检测输出填入，或手动补上。
   * - 报 ``exactly one camera device must set main: true``
     - 只给一个摄像头设备设置 ``main: true``，其余不要设置。
   * - 报 ``unknown LeKiwi task``
     - ``--task-id`` 不是 0 或 1。
   * - 摄像头能打开但不出帧
     - 被其他程序占用，或线缆不传数据。关闭其他摄像头程序后重跑
       ``rlinf-lekiwi-check-camera``。
   * - 串口报 ``Permission denied``
     - 当前用户无权打开串口设备。Linux 上把自己加入 ``dialout`` 组并重新登录。
   * - Dashboard 命令结束后底盘仍在动
     - 服务端没有停止。在 Dashboard 上按 Stop，或在服务端终端按 Ctrl+C；
       ``hold`` 命令在最后一次刷新后一秒也会自行过期。

安全事项
--------

底盘没有碰撞传感器，也没有板载避障：清场是操作者的责任。每次运行前清理地面，把 ``drive`` 的速度上限保持保守，并确保操作者随时够得到电源开关。如果摄像头画面或上报的轮速看起来不对，立即停止运行——一次 ``stop()`` 调用或 Dashboard 的 Stop 按钮只需一次 RPC 就能到达底盘。
