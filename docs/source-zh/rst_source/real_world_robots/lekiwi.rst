LeKiwi
======

LeKiwi 是一个三轮全向移动底盘：三个 Feetech STS3215 轮电机挂在同一条串行总线上，外加一个前置 USB 摄像头。RPent 通过 RLinf 的公开机器人和底盘接口控制它：规划器指定速度和持续时间，或相对旋转角度。服务端在每次动作结束后停车，并记录状态和前置摄像头画面。

这里涉及两台计算机。机器人的板载计算机（通常是树莓派）持有舵机总线和摄像头，负责运行环境服务端；控制端（通常是同一网络下的笔记本）运行 RPent 和规划器。

开始前，请先按 `RLinf LeKiwi 指南 <https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/lekiwi.html>`_ 完成电机编号、车轮和摄像头检查。确认硬件正常后，再按本文安装 RPent、生成机器人配置、启动环境服务端并运行任务。

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

准备机器人板载计算机
--------------------

.. code-block:: bash

   cd /path/to/RPent
   uv venv --python 3.11
   source .venv/bin/activate
   uv pip install --torch-backend cpu -e '.[lekiwi]'
   uv pip check

准备控制端
----------

在控制端运行 Agent 或录制命令，需要安装 ``lekiwi`` extra：

.. code-block:: bash

   cd /path/to/RPent
   uv venv --python 3.11
   source .venv/bin/activate
   uv pip install --torch-backend cpu -e '.[lekiwi]'
   uv pip check

控制端需要能通过网络访问机器人端。

编写 LeKiwi YAML
----------------

在机器人板载计算机上运行检测脚本，生成包含检测到的串口和摄像头标识符的 LeKiwi YAML：

.. code-block:: bash

   python -m robots.lekiwi.detect --output /path/to/lekiwi.yaml

脚本不会让底盘运动：它按各个波特率对串行总线执行 ping，打开每个摄像头确认能取到一帧，并打印探测结果：

.. code-block:: text

   == LeKiwi detection ==
   serial ports scanned: ['/dev/ttyACM0']
     /dev/ttyACM0: motor ids [7, 8, 9]
   wheel bus: /dev/ttyACM0
   camera /dev/v4l/by-id/usb-USB_Camera_USB_Camera_2505280051-video-index0: ok at 640x480
   wrote /path/to/lekiwi.yaml
   review before the first run:
     - robot.wheel_radius / robot.base_radius: measure with a ruler (m)
     - drive.max_linear_velocity / max_angular_velocity: policy; start low
     - drive.frame_size: camera output size; raise from 128x128 if needed
     - drive.step_frequency: control rate; 10 Hz suits the STS3215 bus
     - drive.camera_max_age: frame reuse window; 0.05 tracks a 15 fps camera

只探测其中一侧时可以加 ``--skip-motors`` 或 ``--skip-cameras``。生成的文件与 ``robots/lekiwi/config/example.yaml`` 结构相同：

.. code-block:: yaml

   robot:
     serial_port: /dev/ttyACM0
     wheel_radius: 0.05
     base_radius: 0.125
     max_raw: 3000

   cameras:
     devices:
       front:
         serial: "<front-camera-serial>"
         type: uvc
         main: true

   drive:
     step_frequency: 10.0
     camera_max_age: 0.05
     max_linear_velocity: 0.3
     max_angular_velocity: 60.0
     frame_size: [128, 128]

请实测并填写 ``wheel_radius`` （车轮半径）和 ``base_radius`` （底盘中心到车轮的距离），单位均为米。

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
     - 执行一次保守的前进、一次原地旋转，并在每次动作后检查摄像头画面；每次驱动不超过 2 秒、0.1 m/s 和 30 deg/s；每次旋转不超过 30 度。
   * - 1
     - ``room_survey``
     - 按有界分段在房间中行进，并汇报摄像头看到的内容；每次驱动不超过 3 秒、0.2 m/s 和 45 deg/s；每次旋转不超过 90 度。

启动环境服务端
--------------

在机器人板载计算机上，把服务端绑定到控制端可以访问的网卡：

.. code-block:: bash

   rpent-lekiwi-server \
     --robot-config /path/to/lekiwi.yaml \
     --transport http --host 0.0.0.0 --port 8120

它会打印供控制端接入的地址：

.. code-block:: text

   RPC server listening on http://0.0.0.0:8120

从控制端接入
------------

针对刚启动的服务端运行一个任务：

.. code-block:: bash

   rpent --robot lekiwi --task-id 0 \
     --env-endpoint http://ROBOT_HOST:8120 \
     --robot-config /path/to/lekiwi.yaml \
     --planner api --model MODEL_ID

RLinf 遥操与录制
-----------------

键盘遥操使用 RLinf，在 RLinf 源码目录运行命令。先停止普通 RPent Env，确保电机串口只由 RLinf 接收端使用。如需同时录像，使用 ``--camera-only`` 启动 RPent；该模式只打开 YAML 中标记为 ``main: true`` 的摄像头，不能驱动底盘。

在机器人电脑的两个终端分别运行：

.. code-block:: bash

   # 终端 1，在 RLinf 源码目录运行，持有电机串口。
   python -m toolkits.realworld_check.test_lekiwi_env teleop \
     --listen --port /dev/ttyACM0

   # 终端 2，在 RPent 源码目录运行，只持有主摄像头。
   rpent-lekiwi-server --camera-only \
     --robot-config /path/to/lekiwi.yaml \
     --transport http --host 0.0.0.0 --port 8120

在接键盘的电脑上，从 RLinf 源码目录运行：

.. code-block:: bash

   python -m toolkits.realworld_check.test_lekiwi_env teleop --remote ROBOT_HOST

按键和速度选项见 `RLinf 键盘遥操指南 <https://github.com/RLinf/RLinf/blob/main/docs/source-zh/rst_source/examples/embodied/lekiwi.rst>`_。

独立录制
--------

任一 Env 模式启动后，在能连接它的电脑上运行：

.. code-block:: bash

   python -m robots.lekiwi.record \
     --env-endpoint http://ROBOT_HOST:8120 \
     --recording-dir /path/to/recordings

命令立即开始录像，包含静止画面，文件保存在运行该命令的电脑上。Ctrl-C 或 SIGTERM 会完成视频写入后退出。启停录制不影响机器人控制。

默认录制 600 秒、每秒最多 10 帧，可用 ``--duration`` （最多 3600 秒）和 ``--fps`` （最多 60）调整。

工具
----

规划器可以调用 ``drive``、``rotate``、``stop`` 和只读的 ``view_env_state(step)``。运动工具示例：

.. code-block:: python

   drive(velocity=[0.1, 0.0, 0.0], duration_s=2.0)  # 向前驱动 2 秒。
   drive(velocity=[0.0, 0.1, 0.0], duration_s=1.0)  # 向左平移 1 秒。
   rotate(angle_deg=90)                            # 向左旋转 90 度。
