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

以下双目建图与导航使用 ZED 2i，按下表部署。命令在 RPent 目录执行，配置、录制和地图保存在仓库外的自选目录。

.. list-table::
   :header-rows: 1
   :widths: 20 45 35

   * - 运行位置
     - 职责
     - 主要依赖
   * - 机器人板载计算机
     - Env 持有相机和底盘；本地计算双目深度、分割地面与障碍，运行 Nav2 规划和控制。
     - RPent、配套 RLinf、OpenCV、ROS 2 Jazzy、Nav2、``rtabmap_util``；不需要 GPU 或 ZED SDK。
   * - 云端
     - 离线生成深度和地图；在线接收机器人图像，在已有地图中定位并返回位姿。
     - 离线深度使用 CUDA 与 RAFT-Stereo；建图和定位使用 RTAB-Map，定位服务还需 RPent 的原生扩展。在线定位可在 CPU 上运行。
   * - 操作电脑
     - 运行 RPent Dashboard 和规划器，连接机器人 Env。
     - RPent；不承担本地障碍感知，也不中转机器人与云端的定位图像。

机器人端的 ``rtabmap_util/obstacles_detection`` 只接收本地双目点云并分割地面、障碍，不在机器人端运行整套建图或云端定位服务。RAFT-Stereo 权重、建图工具和定位扩展安装在云端。

1. 机器人端安装
------------------

Env 使用独立的 RPent Python 环境，安装 RPent 与配套 RLinf 源码；已有兼容环境可复用。RPent 支持 Python 3.10–3.12，下面使用 3.12：

.. code-block:: bash

   cd /path/to/RPent
   uv venv --python 3.12
   source .venv/bin/activate
   uv pip install --torch-backend cpu -e ".[lekiwi]"
   uv pip install --no-deps -e /path/to/RLinf

ROS 2 只用于机器人导航，遥操和图片录制不需要。需要导航时按 `ROS 2 Jazzy 安装文档 <https://docs.ros.org/en/jazzy/Installation.html>`_ 安装 ROS，并准备 ``nav2_bringup``、``nav2_mppi_controller``、``launch_ros``、``robot_localization``、``rtabmap_util``、``sensor_msgs_py`` 和 ``rmw_cyclonedds_cpp``。

在 ROS 使用的 Python 环境安装桥接依赖和 RPent：

.. code-block:: bash

   /path/to/ros-python -m pip install numpy scipy opencv-python-headless pillow pyyaml omegaconf
   /path/to/ros-python -m pip install --no-deps -e /path/to/RPent

2. 机器人配置
----------------

.. code-block:: bash

   cp robots/lekiwi/config/example.yaml /path/to/robot.yaml

填写 ``robot.port``、相机的 UVC 设备标识 ``cameras.devices.front.serial``、相机标签上的 ``zed_serial``，以及 ``perception.calibration``。按 `Stereolabs 标定说明 <https://docs.stereolabs.com/docs/development/zed-sdk/modules/camera/camera-calibration>`_ 获取本机出厂标定文件。相对路径以机器人 YAML 所在目录为准。

``geometry`` 只填写实际数值：机身四侧距离与高度、相机相对底盘中心的安装位置和角度、导航半径、速度与加速度。坐标为 x 向前、y 向左、z 向上，长度单位为米，安装角度为度；机身尺寸包含固定机械臂。实测导航半径示例为 0.17 m，程序直接使用，不额外加 padding。

``eye_size`` 默认 ``[1280, 720]``；``drive`` 设置遥操速度和频率。``perception.stereo`` 与 ``perception.obstacles`` 可留空。导航读取系统安装的 ``nav2_bringup/params/nav2_params.yaml``，再应用 ``nav2_overrides.yaml`` 中的全向运动、点云及话题接入参数；尺寸与运动限制取机器人 YAML。

3. 云端安装
----------------

云端安装 RPent，复用第 1 节的 Python 安装命令。将以下项目 clone 到自选路径，依赖和模型下载按上游文档操作，不固定源码分支：

.. code-block:: bash

   git clone https://github.com/introlab/rtabmap.git /path/to/rtabmap
   git clone https://github.com/princeton-vl/RAFT-Stereo.git /path/to/RAFT-Stereo

- `RTAB-Map 安装文档 <https://github.com/introlab/rtabmap/wiki/Installation>`_：启用 Qt 图像导入工具和 GTSAM，将官方命令加入 ``PATH``。
- `RAFT-Stereo 文档 <https://github.com/princeton-vl/RAFT-Stereo>`_：安装 CUDA PyTorch 环境并下载 ``raftstereo-middlebury.pth``，在该环境安装 RPent。普通 Python 依赖可以使用国内镜像。

在运行定位服务的 Python 环境安装可选编译依赖，再用 CMake 编译并安装扩展。需要 C++17 编译器、Python 开发头文件，以及 RTAB-Map 和 OpenCV 开发库；只在云端执行，不需要在 Pi 安装：

.. code-block:: bash

   python -m pip install -e ".[lekiwi-localization]"
   cmake -S robots/lekiwi/rtabmap_extension -B /path/to/lekiwi-extension-build \
     -DPython_EXECUTABLE="$(python -c 'import sys; print(sys.executable)')" \
     -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
     -DCMAKE_PREFIX_PATH=/path/to/rtabmap-install \
     -DCMAKE_INSTALL_RPATH=/path/to/rtabmap-install/lib
   cmake --build /path/to/lekiwi-extension-build
   cmake --install /path/to/lekiwi-extension-build \
     --prefix "$(python -c 'import sysconfig; print(sysconfig.get_path("platlib"))')"

扩展在定位服务进程内运行，直接接收 NumPy 图像。同一地图实例的调用串行执行，RTAB-Map 运算期间释放 GIL。``--max-processing-ms`` 用于丢弃超时返回的结果，不能中断原生调用；原生调用卡死或进程崩溃时需要重启定位服务。Pi 端继续检查定位结果的新鲜度。

4. 启动遥操
----------------

机器人端启动 Env，统一持有底盘和相机：

.. code-block:: bash

   python -m robots.lekiwi.env_server --robot-config /path/to/robot.yaml \
     --host 0.0.0.0 --port 8120 --recording-dir /path/to/recordings

``0.0.0.0`` 是本机监听地址，不是云端 IP。操作电脑安装 RPent 后启动 Dashboard；将示例机器人地址 ``192.168.1.10`` 替换为实际地址。同机运行时用 ``127.0.0.1``。

.. code-block:: bash

   rpent --robot lekiwi --dashboard --planner manual \
     --env-endpoint http://192.168.1.10:8120 --robot-config /path/to/robot.yaml

打开终端打印的 Dashboard 地址，在输入框提交 ``/rpent-task 0`` 后进入遥操。按住方向按钮行驶，松开停车；视频与动作记录使用原有界面选项。

5. 遥操时录制双目图片
------------------------

双目 PNG 录制使用现有 Env 接口。在操作电脑另开 Python 交互会话，连接已经运行的 Env 并开始录制；地址替换为实际机器人地址：

.. code-block:: python

   from uuid import uuid4
   from robots.lekiwi.env_client import LeKiwiEnvClient
   from rpent.utils.rpc import make_rpc_client

   env = LeKiwiEnvClient(make_rpc_client("http://192.168.1.10:8120"))
   identity = uuid4().hex
   env.recording("start", identity=identity)

在 Dashboard 遥操。在同一 Python 会话调用 ``env.recording("state", identity=identity)`` 查看状态、帧数和保存目录；完成采集后执行：

.. code-block:: python

   env.recording("stop", identity=identity)
   env.close()

默认每次录制最多 600 秒，每秒最多保存 5 对图片。可在启动 Env 时用 ``--capture-seconds`` 和 ``--recording-fps`` 修改。结束 Dashboard 手动任务或关闭客户端不会停止双目 PNG 录制；需显式停止、等待时限到达或关闭 Env。Dashboard 原有视频与动作记录独立于双目 PNG 采集。

把机器人端生成的整个 ``capture-*`` 目录和机器人 YAML 复制到云端。录制目录包含 ``left/``、``right/``、``stereo.json`` 和 ``calibration.conf``。之后的离线处理只读文件，不需要连接机器人。

6. 离线建图与地图节点
----------------------------

在云端 RAFT-Stereo Python 环境导出校正图片、深度与 RTAB-Map 配置：

.. code-block:: bash

   python -m robots.lekiwi.mapping \
     --input /path/to/capture --output /path/to/map-output/rgbd \
     --robot-config /path/to/robot.yaml --raft-repo /path/to/RAFT-Stereo \
     --checkpoint /path/to/raftstereo-middlebury.pth

使用 `RTAB-Map 官方工具 <https://github.com/introlab/rtabmap/wiki/Tools>`_ 完成建图：

.. code-block:: bash

   cd /path/to/map-output
   export QT_QPA_PLATFORM=offscreen

   # 导入彩色图和深度
   rtabmap-dataRecorder -hide rgbd/rtabmap.ini raw.db

   # 估计相机运动并建图
   rtabmap-reprocess -default -odom -c rgbd/rtabmap.ini raw.db slam.db

   # 应用地面运动的平面约束
   rtabmap-reprocess -c rgbd/rtabmap.ini -noimu \
     --Reg/Force3DoF true --RGBD/ForceOdom3DoF true \
     slam.db rtabmap.db

   # 导出导航栅格地图
   rtabmap-export --map --opt 2 --output map rtabmap.db

地图效果不理想时，可参考 `官方栅格参数说明 <https://github.com/introlab/rtabmap/blob/master/corelib/include/rtabmap/core/Parameters.h>`_ 调整参数。

将 ``map.yaml`` 和其中引用的地图图片复制到机器人端；云端保留同次建图生成的 ``rtabmap.db``。用 RTAB-Map 官方数据库查看器打开地图，查看图节点及对应图像，记下目标节点 ID：

.. code-block:: bash

   rtabmap-databaseViewer /path/to/map-output/rtabmap.db

导航目标使用数据库中的节点 ID，或已经保存的节点标签（例如 ``door``）。位置和朝向直接取该节点的优化后位姿，不再创建目标 JSON 或额外的选点页面。标签保存在 RTAB-Map 数据库中；修改标签后更新云端数据库并重启定位服务。重新建图时，同时更新地图 YAML、图片和数据库。

目标解析复用 RTAB-Map 官方图接口，路径规划与避障由 Nav2 负责。

.. _lekiwi-navigation-start:

7. 云端定位
----------------

在 Pi 和云端的机器人 YAML 中填写相同的 ``localization.endpoint`` 和 ``localization.token``。可用 ``python -c "import secrets; print(secrets.token_hex(32))"`` 生成 token。云端启动定位服务，``--peer`` 填云端看到的机器人来源 IP：

.. code-block:: bash

   python -m robots.lekiwi.localization_server serve \
     --database /path/to/map-output/rtabmap.db \
     --bind 0.0.0.0 --port 18125 --peer 192.168.1.10 \
     --robot-config /path/to/robot.yaml

8. 机器人导航
----------------

停止采集 Env，在机器人端启动导航 Env：

.. code-block:: bash

   source /path/to/ros-ws/install/setup.bash
   python -m robots.lekiwi.env_server --mode navigation \
     --robot-config /path/to/robot.yaml --map /path/to/map-output/map.yaml \
     --ros-python /path/to/ros-python \
     --host 0.0.0.0 --port 8120

该命令使用真实底盘；添加 ``--mock`` 则使用模拟电机，相机仍是真实的。导航目标直接使用 Env 选择的模式，无需重复指定硬件参数。

通过下一节的 Dashboard 与 Agent 选择地图节点或已有标签（如 ``door``）。Agent 使用 ``get_navigation_state`` 检查就绪状态、``navigate_to`` 提交目标、``stop`` 停车并取消目标。停车后，等待 ``operation.state`` 不再是 ``running`` 再提交下一个目标。

导航持续运行，直到 Env 退出或组件故障，没有固定运行时长。Env 持有相机和底盘，ROS launch 管理导航节点；``stop`` 会保留这些进程。需要返回遥操或录制时，停止导航 Env，按第 4 节重新启动采集模式的 Env。Env 退出时关闭导航进程组，故障组件不自动重启。导航日志写入机器人 YAML 同目录的 ``navigation.log``。

重复或过期图像会被丢弃。``perception.stereo.maximum_age_seconds`` （默认 1 秒）同时用于本地感知和 Collision Monitor 的障碍数据超时检查。普通定位匹配失败不会作废上次仍有效的定位；坐标系错误或过大的修正会使定位失效。``localization.localization_max_age`` （默认 2 秒）控制定位有效期。

定位或 TF 暂时不可用时，导航适配层输出零速，保留当前目标最多 ``localization_max_age`` 秒；恢复后可继续，持续不可用则取消目标。等待仍计入目标超时，Nav2 自身返回失败时也会结束任务。图像时间戳为主机读取完成时间，不是硬件曝光时间。

相机 RPC 在暂无新鲜帧或本机通信暂时失败时跳过本轮，恢复后继续发布；服务端设备错误仍会报告失败。查询导航状态暂时失败只返回状态不可用，不撤销运动控制许可；导航进程退出时才停车并将目标标记为失败。运动侧的定位检查和指令有效期持续生效。

9. Dashboard 与 Agent
----------------------------------------

已有 Codex 登录配置时，可直接使用 Codex 规划器；模型名按账号可用配置填写：

.. code-block:: bash

   rpent --robot lekiwi --dashboard --planner codex --model gpt-6-astra \
     --robot-config /path/to/robot.yaml --env-endpoint http://192.168.1.10:8120
