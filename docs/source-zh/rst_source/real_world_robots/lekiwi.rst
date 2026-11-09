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
     - Env 持有相机和底盘；本地计算双目深度、分割地面与障碍，运行 Nav2 规划和控制；视觉定位也可在这里使用 CPU 执行。
     - RPent、配套 RLinf、OpenCV、ROS 2 Jazzy、Nav2、``rtabmap_util``；车端定位还需 RTAB-Map 原生扩展，不需要 GPU 或 ZED SDK。
   * - 云端
     - 离线生成深度和地图；机器人 CPU 或内存不足时，也可在这里运行在线视觉定位服务。
     - 离线深度使用 CUDA 与 RAFT-Stereo；建图和定位使用 RTAB-Map，定位服务还需 RPent 的原生扩展。在线定位可在 CPU 上运行。
   * - 操作电脑
     - 运行键盘采集或 RPent Dashboard 和规划器，连接机器人 Env。
     - RPent；不承担本地障碍感知，也不中转机器人与云端的定位图像。

机器人端的 ``rtabmap_util/obstacles_detection`` 负责地面和障碍分割，独立的 CPU 定位进程查询已有 RTAB-Map 数据库。定位扩展安装在运行定位进程的机器上；RAFT-Stereo 和离线建图工具仍可放在云端。车端定位要求 CPU 和内存能够同时支撑地图、相机、双目感知与 Nav2。

1. 机器人端安装
------------------

配套 RLinf 相机接口必须支持 ``BaseCamera.get_frame(return_timestamp=True)``，用于对齐图像和轮速时间。

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

3. 建图与定位依赖
------------------------

离线建图使用 CUDA 和 RAFT-Stereo；车端 CPU 定位只需 RTAB-Map、OpenCV 和下述原生扩展，机器人端无需安装 RAFT-Stereo。

离线建图时，在云端安装 RPent，复用第 1 节的 Python 安装命令。将以下项目 clone 到自选路径，依赖和模型下载按上游文档操作，不固定源码分支：

.. code-block:: bash

   git clone https://github.com/introlab/rtabmap.git /path/to/rtabmap
   git clone https://github.com/princeton-vl/RAFT-Stereo.git /path/to/RAFT-Stereo

- `RTAB-Map 安装文档 <https://github.com/introlab/rtabmap/wiki/Installation>`_：启用 Qt 图像导入工具和 GTSAM，将官方命令加入 ``PATH``。
- `RAFT-Stereo 文档 <https://github.com/princeton-vl/RAFT-Stereo>`_：安装 CUDA PyTorch 环境并下载 ``raftstereo-middlebury.pth``，在该环境安装 RPent。普通 Python 依赖可以使用国内镜像。

在运行定位服务的 Python 环境安装可选编译依赖，再用 CMake 编译并安装扩展。需要 C++17 编译器、Python 开发头文件，以及 RTAB-Map 和 OpenCV 开发库。在定位服务所在机器执行；使用车端定位时需在机器人端编译安装：

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

4. 启动采集 Env
------------------------

机器人端启动 Env，由它持有底盘、ZED 相机和 PNG 录制器：

.. code-block:: bash

   python -m robots.lekiwi.env_server --robot-config /path/to/robot.yaml \
     --host 0.0.0.0 --port 8120 --recording-dir /path/to/recordings

``0.0.0.0`` 是本机监听地址。采集期间由采集客户端独占使用这个 Env，不要同时连接 Agent 或其他驾驶客户端。导航使用后文的独立启动流程。

在操作电脑的 Python 3.10–3.12 环境安装 RPent 遥操依赖：

.. code-block:: bash

   cd /path/to/RPent
   uv venv --python 3.12
   source .venv/bin/activate
   uv pip install -e ".[lekiwi-teleop]"

该依赖组提供 LeRobot 键盘输入和 Rerun 预览，操作电脑无需安装 RLinf、ROS 或电机驱动。键盘输入需要本地桌面：macOS 需授予终端“辅助功能／输入监控”权限；Linux 使用 X11。此入口不支持无桌面的 SSH 或 Wayland 键盘采集。

5. 遥操并采集双目图片
----------------------------

在操作电脑执行，将 ``ROBOT_HOST`` 替换为机器人的主机名或 IP：

.. code-block:: bash

   rpent-lekiwi-capture --env-endpoint http://ROBOT_HOST:8120

命令打开 Rerun 相机预览，并在机器人端开始一段录制。终端显示录制状态、图片对数和机器人端保存目录。录制器保存第一对图片后才允许驾驶。按住 W/S 前进／后退，A/D 左移／右移，Z/X 左转／右转；松开方向键停车，Q、Esc 或 Ctrl-C 结束采集并停车。键盘监听是全局的，采集期间避免在其他应用中打字。

速度来自 Env 的 ``drive.max_linear_velocity`` 和 ``drive.max_angular_velocity`` 配置，Env 继续执行配置中的限幅。预览分辨率使用 ``drive.frame_size``，录制保留原始双目分辨率。直接目视小车驾驶时可加 ``--no-preview`` 关闭预览。

默认每段最多录制 600 秒，每秒最多保存 5 对图片，通过 Env 的 ``--capture-seconds`` 和 ``--recording-fps`` 修改。达到录制时限后客户端结束。键盘断开、预览／录制出错或控制循环卡顿时，客户端退出并尝试停车、结束录制；恢复后需重新运行命令。客户端失联时 Env 会使运动命令过期。若网络已断开，无法保证远端收尾成功，机器人端录制器可能继续运行到配置的时限或 Env 关闭。

客户端复用 LeRobot 的 ``KeyboardTeleop`` 和预览工具，通过现有 Env 的 ``hold``／``stop`` 和双目录制接口工作，轮电机仍由 RLinf 控制。采集不需要 Dashboard、规划器或示教机械臂，只保存双目 PNG 和标定信息，不另存 MP4 或动作日志。

把机器人端生成的整个 ``capture-*`` 目录和机器人 YAML 复制到离线建图机器。目录包含 ``left/``、``right/``、``stereo.json`` 和 ``calibration.conf``，左右目以包含源帧时间戳的同名文件配对。之后的离线处理只读文件，不需要连接机器人。

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

将 ``map.yaml`` 和其中引用的地图图片复制到机器人端；同次建图生成的 ``rtabmap.db`` 放到定位服务所在机器：可以是机器人，也可以是外部服务器。用 RTAB-Map 官方数据库查看器打开地图，查看图节点及对应图像，记下目标节点 ID：

.. code-block:: bash

   rtabmap-databaseViewer /path/to/map-output/rtabmap.db

导航目标使用数据库中的节点 ID，或已经保存的节点标签（例如 ``door``）。位置和朝向直接取该节点的优化后位姿，不再创建目标 JSON 或额外的选点页面。标签保存在 RTAB-Map 数据库中；修改标签后更新定位数据库并重启定位服务。重新建图时，同时更新地图 YAML、图片和数据库。

目标解析复用 RTAB-Map 官方图接口，路径规划与避障由 Nav2 负责。

Agent 可用 ``list_map_nodes`` 查找节点 ID 和已有标签，用 ``inspect_map_node`` 查看节点的历史图像。``view_map`` 默认显示地点标签，并在导航就绪时显示机器人位姿。未命名节点不画在默认图片上，但节点 ID 仍保留在返回数据中。指定 ``center_node_id`` 后，高亮候选节点并显示其编号；有标签时同时显示标签。添加 ``radius_m`` 可裁剪查看周围区域。图片展示静态地图，不包含实时障碍物。

.. _lekiwi-navigation-start:

7. 启动定位服务
------------------------

定位是独立的 CPU 服务，部署方式与 SAM3 服务相同。同一条命令可以在机器人或其他机器上运行。在所选机器上，使用已安装 RPent 和定位原生扩展的 Python 环境启动：

.. code-block:: bash

   python -m robots.lekiwi.localization_server serve \
     --database /path/to/map-output/rtabmap.db \
     --bind 127.0.0.1 --port 18125

服务加载 ``--database`` 指定的数据库，接收机器人导航工作进程发送的校正后左目图像、相机参数、采集时间戳和对齐的轮速里程计，不直接打开相机或电机总线。服务的 Python 环境可以独立于机器人端的 ROS 环境；部署位置不同，定位算法和请求格式相同。

保持服务运行，在另一个终端按下一节启动导航 Env。机器人 YAML 只向调用方提供服务地址：

.. code-block:: yaml

   localization:
     endpoint: http://127.0.0.1:18125

部署到其他机器时，在该机器执行同一条服务命令，将 ``--bind`` 改为 ``0.0.0.0``，并将机器人端的 ``localization.endpoint`` 改为 ``http://SERVER_HOST:18125``。服务端需要数据库，不需要机器人 YAML，也没有部署模式或 token 配置。与其他 RPent RPC 服务一样，远程访问应使用可信网络或 SSH 隧道；默认只监听本机回环地址。

图像定位、地图查询和目标解析共用这个 endpoint。无论服务部署在哪里，图像工作进程都会等待健康检查通过，最多等待 60 秒。这只表示数据库加载完成，不代表机器人已经定位成功。Env 管理导航工作进程，定位服务单独启停；Env 退出后定位服务继续运行。定位 RPC 调用失败会使图像工作进程退出，并停止导航图；原有定位新鲜度检查和运动门控继续生效。

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

9. Dashboard 与 Agent
----------------------------------------

已有 Codex 登录配置时，可直接使用 Codex 规划器；模型名按账号可用配置填写：

.. code-block:: bash

   rpent --robot lekiwi --dashboard --planner codex --model gpt-6-astra \
     --robot-config /path/to/robot.yaml --env-endpoint http://192.168.1.10:8120
