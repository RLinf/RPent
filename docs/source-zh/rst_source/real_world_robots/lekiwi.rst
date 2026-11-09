LeKiwi
======

使用 RPent 遥操 LeKiwi 采集 ZED 双目图像、构建地图，并让 Agent 导航到地图中的节点或命名地点。

概览
----

硬件准备见 `RLinf LeKiwi 指南 <https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/lekiwi.html>`_。本页使用 LeKiwi 底盘和 ZED 2i，按下表部署：

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - 运行位置
     - 用途
     - 依赖
   * - 机器人端（Linux）
     - Env 连接相机和底盘；Nav2 规划与避障。
     - RPent、配套 RLinf；导航需 ROS 2 Jazzy，无需 GPU 或 ZED SDK。
   * - 建图机器
     - 离线生成深度和地图。
     - RPent、CUDA、RAFT-Stereo、RTAB-Map。
   * - 操作电脑
     - 键盘采集、Dashboard 与 Agent。
     - RPent，网络可达机器人端。

定位服务使用 CPU，可运行在机器人端或其他机器。实机运行需有人监督，并能随时切断电机电源。

安装与资源准备
--------------

使用 Python 3.10–3.12，以下命令在 RPent 目录执行。将示例路径和 ``ROBOT_HOST`` 替换为实际值，配置和数据保存在仓库外。

机器人端
~~~~~~~~

安装 RPent 和配套 RLinf 源码；RLinf 相机接口需支持 ``get_frame(return_timestamp=True)``：

.. code-block:: bash

   cd /path/to/RPent
   uv venv --python 3.12
   source .venv/bin/activate
   uv pip install --torch-backend cpu -e ".[lekiwi]"
   uv pip install --no-deps -e /path/to/RLinf

导航还需安装 `ROS 2 Jazzy <https://docs.ros.org/en/jazzy/Installation.html>`_ 及 ``nav2_bringup``、``nav2_mppi_controller``、``launch_ros``、``robot_localization``、``rtabmap_util``、``sensor_msgs_py``、``rmw_cyclonedds_cpp``。在独立的 ROS Python 环境安装桥接依赖：

.. code-block:: bash

   /path/to/ros-python -m pip install numpy scipy opencv-python-headless pillow pyyaml omegaconf
   /path/to/ros-python -m pip install --no-deps -e /path/to/RPent

操作电脑
~~~~~~~~

安装 RPent 及键盘采集、预览依赖：

.. code-block:: bash

   cd /path/to/RPent
   uv venv --python 3.12
   source .venv/bin/activate
   uv pip install -e ".[lekiwi-teleop]"

操作电脑无需安装 RLinf、ROS 或电机驱动。

建图与定位
~~~~~~~~~~

在建图机器安装 RPent，并按上游文档准备以下资源：

- `RTAB-Map <https://github.com/introlab/rtabmap/wiki/Installation>`_：启用 Qt 图像导入工具和 GTSAM，将命令加入 ``PATH``。
- `RAFT-Stereo <https://github.com/princeton-vl/RAFT-Stereo>`_：准备 CUDA PyTorch 环境及 ``raftstereo-middlebury.pth``，在该环境安装 RPent。

.. code-block:: bash

   git clone https://github.com/introlab/rtabmap.git /path/to/rtabmap
   git clone https://github.com/princeton-vl/RAFT-Stereo.git /path/to/RAFT-Stereo

定位机器只需 RPent、RTAB-Map、OpenCV 及定位扩展。准备 C++17 编译器、Python 头文件、RTAB-Map 和 OpenCV 开发库后，在定位服务的 Python 环境执行：

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

机器人配置
----------

复制示例配置并填写本机参数：

.. code-block:: bash

   cp robots/lekiwi/config/example.yaml /path/to/robot.yaml

- ``robot``：电机串口、实测轮半径和底盘半径。
- ``cameras.devices.front``：UVC 设备标识 ``serial``、相机标签上的 ``zed_serial``；``eye_size`` 默认 ``[1280, 720]``。
- ``perception.calibration``：按 `Stereolabs 标定说明 <https://docs.stereolabs.com/docs/development/zed-sdk/modules/camera/camera-calibration>`_ 获取的出厂标定文件；相对路径以 YAML 所在目录为准。
- ``geometry``：实测机身尺寸（含固定机械臂）、相机安装位置与角度、导航半径及运动限制。坐标为 x 向前、y 向左、z 向上，长度为米，安装角度为度。
- ``drive``：遥操速度与频率。``perception.stereo`` 和 ``perception.obstacles`` 可保留默认值。

遥操与采集
----------

机器人端启动 Env，由它持有底盘、ZED 相机和 PNG 录制器：

.. code-block:: bash

   python -m robots.lekiwi.env_server --robot-config /path/to/robot.yaml \
     --host 0.0.0.0 --port 8120 --recording-dir /path/to/recordings

``0.0.0.0`` 是本机监听地址。采集期间由采集客户端独占使用这个 Env，不要同时连接 Agent 或其他驾驶客户端。导航使用后文的独立启动流程。

键盘输入需要本地桌面：macOS 需授予终端“辅助功能／输入监控”权限；Linux 使用 X11。此入口不支持无桌面的 SSH 或 Wayland 键盘采集。

在操作电脑执行，将 ``ROBOT_HOST`` 替换为机器人的主机名或 IP：

.. code-block:: bash

   rpent-lekiwi-capture --env-endpoint http://ROBOT_HOST:8120

命令打开 Rerun 相机预览，并在机器人端开始一段录制。终端显示录制状态、图片对数和机器人端保存目录。录制器保存第一对图片后才允许驾驶。按住 W/S 前进／后退，A/D 左移／右移，Z/X 左转／右转；松开方向键停车，Q、Esc 或 Ctrl-C 结束采集并停车。键盘监听是全局的，采集期间避免在其他应用中打字。

速度来自 Env 的 ``drive.max_linear_velocity`` 和 ``drive.max_angular_velocity`` 配置，Env 继续执行配置中的限幅。预览分辨率使用 ``drive.frame_size``，录制保留原始双目分辨率。直接目视小车驾驶时可加 ``--no-preview`` 关闭预览。

默认每段最多录制 600 秒，每秒最多保存 5 对图片，通过 Env 的 ``--capture-seconds`` 和 ``--recording-fps`` 修改。达到录制时限后客户端结束。键盘断开、预览／录制出错或控制循环卡顿时，客户端退出并尝试停车、结束录制；恢复后需重新运行命令。客户端失联时 Env 会使运动命令过期。若网络已断开，无法保证远端收尾成功，机器人端录制器可能继续运行到配置的时限或 Env 关闭。

采集只保存双目 PNG 和标定信息，不另存 MP4 或动作日志。

把机器人端生成的整个 ``capture-*`` 目录和机器人 YAML 复制到离线建图机器。目录包含 ``left/``、``right/``、``stereo.json`` 和 ``calibration.conf``，左右目以包含源帧时间戳的同名文件配对。之后的离线处理只读文件，不需要连接机器人。

离线建图
--------

在建图机器的 RAFT-Stereo 环境导出校正图像、深度与 RTAB-Map 配置：

.. code-block:: bash

   python -m robots.lekiwi.mapping \
     --input /path/to/capture --output /path/to/map-output/rgbd \
     --robot-config /path/to/robot.yaml --raft-repo /path/to/RAFT-Stereo \
     --checkpoint /path/to/raftstereo-middlebury.pth

使用 `RTAB-Map 工具 <https://github.com/introlab/rtabmap/wiki/Tools>`_ 建图并导出导航栅格：

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

将 ``map.yaml`` 和引用的地图图片复制到机器人端，将同次生成的 ``rtabmap.db`` 放到定位机器。可用数据库查看器查看节点及参考图像：

.. code-block:: bash

   rtabmap-databaseViewer /path/to/map-output/rtabmap.db

导航目标为已有节点 ID 或标签（如 ``door``），到达位置与朝向取该节点位姿。修改标签后更新数据库并重启定位服务；重新建图时同时更新 YAML、地图图片和数据库。

.. _lekiwi-navigation-start:

启动定位服务
------------

在安装了定位扩展的机器上运行，并保持服务运行：

.. code-block:: bash

   python -m robots.lekiwi.localization_server serve \
     --database /path/to/map-output/rtabmap.db \
     --bind 127.0.0.1 --port 18125

在机器人 YAML 中设置地址：

.. code-block:: yaml

   localization:
     endpoint: http://127.0.0.1:18125

若定位运行在其他机器，将服务的 ``--bind`` 改为 ``0.0.0.0``，地址改为 ``http://SERVER_HOST:18125``，通过可信网络或 SSH 隧道连接。定位服务独立启停；数据库加载完成不代表机器人已定位成功。

运行导航任务
------------

停止采集 Env，在机器人端启动导航 Env：

.. code-block:: bash

   source /path/to/ros-ws/install/setup.bash
   python -m robots.lekiwi.env_server --mode navigation \
     --robot-config /path/to/robot.yaml --map /path/to/map-output/map.yaml \
     --ros-python /path/to/ros-python \
     --host 0.0.0.0 --port 8120

默认使用真实电机；添加 ``--mock`` 可使用模拟电机和真实相机。

按 :doc:`../guides/configure_planner` 配置规划器后，在操作电脑连接 Dashboard：

.. code-block:: bash

   rpent --robot lekiwi --dashboard --planner codex --model gpt-6-astra \
     --env-endpoint http://ROBOT_HOST:8120

打开终端给出的地址，用已有节点编号或精确标签启动导航任务：

.. code-block:: text

   /rpent-task door

把 ``door`` 替换为数据库中已有的标签，也可使用节点编号，例如 ``/rpent-task 42``。Dashboard 目标不能含空格；标签含空格时请使用对应节点编号。无需 Dashboard 时，将启动命令中的 ``--dashboard`` 替换为 ``--navigation-target door``。

运行手动驾驶任务时，连接手动模式的 Env，并在启动 Dashboard 时加上 ``--task-id 0``；此时使用 ``/rpent-task <task_id>`` 选择任务（0–2）。键盘采集期间不要连接 Agent。

界面操作见 :doc:`../guides/dashboard`，运行记录见 :ref:`run-output-files`。到达后结合当前相机图像确认结果，Nav2 成功或 ``finish`` 均不代表物理任务已验收。

Agent 可调用的工具
------------------

**动作工具：**

- ``drive(velocity, duration_s=1.0)``：按底盘坐标系速度 ``[x, y, theta]`` 行驶，单位为 m/s、m/s、deg/s；单次最多 10 秒，结束停车，速度受配置限幅。需手动控制模式。
- ``navigate_to(target, timeout_s=60.0)``：导航到已有节点 ID（整数）或标签（字符串），等待结果；需导航模式且定位、感知就绪。
- ``stop()``：停车并取消当前导航目标。等待 ``operation.state`` 不再为 ``running`` 后再提交目标。

**查询与结束工具：**

- ``list_map_nodes(label=None)``：列出节点 ID 和标签，可按已有标签精确筛选。
- ``inspect_map_node(node_id)``：查看节点位姿和历史参考图像；节点表示拍摄位置。
- ``view_map(center_node_id=None, radius_m=None)``：查看静态地图、地点标签和可用的机器人位置；可高亮节点并裁剪周围区域，不显示实时障碍物。
- ``get_navigation_state()``：读取导航就绪状态、地图位姿和当前或上次任务状态。
- ``view_env_state(step=-1)``：读取已记录的状态和前置图像，``-1`` 表示最新记录。
- ``finish(status, summary)``：结束当前 Agent 运行并汇报结果。
