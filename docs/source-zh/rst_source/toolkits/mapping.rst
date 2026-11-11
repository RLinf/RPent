.. _offline-mapping:

离线建图
========

.. note::

   本建图方案仅在 ZED 2i 上验证过，使用 LeKiwi 采集的数据。

使用 RTAB-Map 对已经对齐的 RGB-D 图像进行离线建图，生成地图数据库、二维占据栅格和彩色三维点云，并可导出优化后的运动轨迹。

先安装 RTAB-Map，再准备 RGB-D 图像和对应标定，填写数据路径与相机安装位姿，最后执行建图和导出。RGB-D 可以来自已有数据、相机 SDK 或 RAFT-Stereo；准备完成后使用相同的建图配置和命令。

1. 安装 RTAB-Map
----------------

从源码安装时，先克隆 RTAB-Map 官方独立工具仓库：

.. code-block:: bash

   git clone https://github.com/introlab/rtabmap.git /path/to/rtabmap
   cd /path/to/rtabmap

按 `上游安装说明 <https://github.com/introlab/rtabmap/wiki/Installation#standalone-version>`_ 安装依赖并编译 RTAB-Map，然后检查以下命令是否可用：

.. code-block:: bash

   command -v rtabmap-dataRecorder
   command -v rtabmap-reprocess
   command -v rtabmap-export

此前流程使用 RTAB-Map 0.23.13。

.. _mapping-custom-depth:

2. 准备 RGB-D 图像与标定
-------------------------

本节的目标是准备一个包含 ``rgb/``、``depth/`` 和 ``camera.yaml`` 的数据目录。根据已有数据选择准备方式：

* **已有 RGB-D**：检查下面的输入格式，符合要求后进入 :ref:`mapping-config`。
* **相机 SDK 可导出深度**：按 :ref:`mapping-sdk-depth` 导出 RGB-D。
* **只有双目图片或左右并排视频**：按 :ref:`mapping-raft-depth` 生成 RGB-D，需要对应的双目标定和可运行 RAFT-Stereo 的 NVIDIA GPU。

RGB-D 输入格式
~~~~~~~~~~~~~~

准备以下文件：

* **RGB 图像**：去畸变后的彩色 PNG；每张 RGB 与对应深度来自同一时刻。
* **深度图像**：与 RGB 同尺寸、逐像素对齐的单通道 uint16 PNG。像素值表示沿相机光轴的 Z 距离，单位为毫米，无效像素和超出 65.535 米的深度置零。使用数值深度图，不要使用彩色深度可视化图。
* **相机标定**：RTAB-Map/OpenCV 格式的 ``camera.yaml``，包含对应 RGB 的尺寸和内参，供 RTAB-Map 将深度像素转换为三维点。

每对 RGB 和深度使用相同文件名，以递增、大于 0 的秒时间戳命名，并保持小数位数一致。使用相对时间时，保留实际采集间隔：

.. code-block:: text

   rgbd/
     rgb/1.0.png
     rgb/1.2.png
     depth/1.0.png
     depth/1.2.png
     camera.yaml

``camera.yaml`` 必须描述最终 RGB 图像的相机模型，分辨率、内参和去畸变状态都要匹配。SDK 导出时需同时提供该标定；RAFT-Stereo 准备脚本会生成它。

.. _mapping-sdk-depth:

使用相机 SDK 的深度
~~~~~~~~~~~~~~~~~~~

按厂商官方环境和导出流程，从已有录制中导出同一帧的去畸变 RGB、对齐的数值深度，以及对应分辨率的相机内参。使用厂商工具或自行编写导出程序，将输出整理为上面的 RGB-D 目录；本工具包不提供 SDK 导出脚本：

* 将数值深度换算为毫米，保存为单通道 uint16 PNG；无效值和超出 65.535 米的值置零。
* 保持 RGB 与深度同尺寸、逐像素对齐，并使用相同文件名；提供与导出 RGB 对应的 RTAB-Map 标定 YAML。

例如，ZED SDK 的 `官方深度示例 <https://github.com/stereolabs/zed-sdk/blob/master/tutorials/tutorial%203%20-%20depth%20sensing/python/depth_sensing.py>`_ 使用 ``VIEW.LEFT`` 读取左图、``MEASURE.DEPTH`` 读取对齐深度。按上述格式保存连续图像和对应标定。准备好 ``rgb/``、``depth/`` 和 ``camera.yaml`` 后，进入 :ref:`mapping-config`。

.. _mapping-raft-depth:

使用 RAFT-Stereo 生成深度
~~~~~~~~~~~~~~~~~~~~~~~~~

这部分将同步的双目数据转换为 RGB-D。RPent 的 ``prepare_stereo.py`` 负责校正图像，官方 RAFT-Stereo ``demo.py`` 估计像素位移，``raft_to_rgbd.py`` 再将位移换算为毫米深度。

**安装 Python 环境**

以下命令创建供预处理和 RAFT 推理共用的 Python 3.11 环境，适用于支持 CUDA 12.4 的 Linux/NVIDIA GPU 机器。先安装 `uv <https://docs.astral.sh/uv/getting-started/installation/>`_，然后运行：

.. code-block:: bash

   cd /path/to/RPent
   uv venv --python 3.11 .venv-mapping
   source .venv-mapping/bin/activate
   uv pip install -e . "numpy<2" "opencv-python-headless==4.10.0.84" \
     matplotlib pillow tqdm
   uv pip install torch==2.5.1 torchvision==0.20.1 \
     --index-url https://download.pytorch.org/whl/cu124
   git clone https://github.com/princeton-vl/RAFT-Stereo.git /path/to/RAFT-Stereo

后续保持该环境激活，按命令切换目录。此版本组合尚未完成 GPU 验证；上游环境要求见 `RAFT-Stereo 官方说明 <https://github.com/princeton-vl/RAFT-Stereo#requirements>`_。

**第一步：准备双目数据和标定**

输入二选一：

* **图片目录**：``sequence/left/`` 和 ``sequence/right/``，两侧 PNG、JPEG 或 TIFF 文件名完全对应。
* **双目视频**：本地 MP4、AVI、MKV 或 MOV，同步左右图并排排列，左图在前；解码器需支持对应编码。

先复制仓库中的标定模板：

.. code-block:: bash

   cd /path/to/RPent
   cp toolkits/mapping/stereo.yaml /path/to/stereo.yaml

在 ``stereo.yaml`` 中填入相机或数据集提供的实际标定，替换模板中的全部占位符。它供 ``prepare_stereo.py`` 读取；脚本校正图像后，会生成供 RTAB-Map 使用的 ``camera.yaml``。各字段含义如下：

* ``image_size``：单目图像宽、高。例如 2560 × 720 的左右并排视频填写 ``[1280, 720]``。
* 左右目的 ``K`` 和 ``D``：OpenCV 针孔模型的内参和畸变。
* ``R`` 和 ``T_m``：从左目光学坐标到右目光学坐标的变换，满足 ``X_right = R * X_left + T_m``，平移单位为米。

``R`` 和 ``T_m`` 描述左右相机之间的关系；相机安装位姿在下一步通过 ``--camera-pose`` 单独填写。

使用右相机位于左相机右侧的水平针孔双目。已校正图像填写校正后内参、零畸变、单位旋转矩阵和 ``T_m: [-baseline_m, 0, 0]``。鱼眼图像先用相机工具导出为校正后的针孔双目图像。

**第二步：校正和整理左右图**

先实测左相机相对机器人参考系的固定安装位姿，再通过 ``--camera-pose`` 填入：前三项为左目光心的位置 x/y/z（米），后三项为机械坐标系姿态 roll/pitch/yaw（度）。机器人和相机机械坐标系均采用 x 向前、y 向左、z 向上。脚本根据输入位姿换算校正后的姿态，不会自动测量安装位姿。

将示例中相对机器人原点高 0.5 米、水平朝前的位姿替换为实测值。输出目录必须尚不存在且位于输入目录之外。

**图片输入示例：** 以下假设图片以补零编号命名、采样率为 5 Hz。

.. code-block:: bash

   cd /path/to/RPent
   python -m toolkits.mapping.prepare_stereo \
     --input /path/to/sequence \
     --calibration /path/to/stereo.yaml \
     --camera-pose 0 0 0.5 0 0 0 \
     --fps 5 \
     --output /path/to/prepared

编号文件名按字典序排序。若文件名已经是十进制秒时间戳，将 ``--fps 5`` 替换为 ``--timestamps-from-names``。

**视频输入示例：** 用下面命令替代图片命令。假设视频为 15 fps，``--step 3`` 每三帧取一对，即每秒 5 对；相机位姿同样需替换为实际值。

.. code-block:: bash

   cd /path/to/RPent
   python -m toolkits.mapping.prepare_stereo \
     --input /path/to/stereo.mp4 \
     --calibration /path/to/stereo.yaml \
     --camera-pose 0 0 0.5 0 0 0 \
     --step 3 \
     --output /path/to/prepared

省略 ``--fps`` 即读取视频的恒定帧率，省略 ``--step`` 则处理每帧。可变帧率视频先导出为带时间戳的图像对。

两种输入得到相同结构的 ``prepared/``：校正后的 ``pairs/<timestamp>/im0.png`` （左图）和 ``im1.png`` （右图），以及 ``camera.yaml`` 和 ``stereo.json``。

**第三步：运行官方 RAFT-Stereo**

保持共用环境激活，在 RAFT-Stereo 源码目录下载官方权重。下载脚本使用系统命令 ``wget`` 和 ``unzip``：

.. code-block:: bash

   cd /path/to/RAFT-Stereo
   bash download_models.sh

也可从官方 README 提供的 `Middlebury 权重链接 <https://drive.google.com/file/d/1m3KoukUmKDoMv-ySOO6vBzYfWLyj9yqd/view>`_ 单独下载 ``raftstereo-middlebury.pth``，保存到 ``/path/to/RAFT-Stereo/models/``。

脚本下载的权重解压在 ``models/`` 下。将下面的 ``--restore_ckpt`` 替换为下载得到的 ``raftstereo-middlebury.pth`` 实际路径，并选择可用的 NVIDIA GPU。该示例使用标准权重和官方 demo 默认网络结构。为 ``--output_directory`` 选择新的 ``flow/`` 输出目录。

.. code-block:: bash

   cd /path/to/RAFT-Stereo
   CUDA_VISIBLE_DEVICES=0 python demo.py \
     --restore_ckpt /path/to/raftstereo-middlebury.pth \
     --left_imgs "/path/to/prepared/pairs/*/im0.png" \
     --right_imgs "/path/to/prepared/pairs/*/im1.png" \
     --save_numpy \
     --output_directory /path/to/flow

``--left_imgs`` 和 ``--right_imgs`` 指向 ``prepare_stereo.py`` 生成的校正图像，保留引号，让 demo 展开通配符。``--save_numpy`` 将每对图像的像素位移保存为 ``flow/<timestamp>.npy``，同时生成可视化图。下一步读取数值位移文件 ``.npy``。

**第四步：转换为 RGB-D**

保持同一 Python 环境激活，切回 RPent 源码目录。下面命令读取 ``prepared/`` 中的图像、标定，以及 RAFT-Stereo 生成的 ``flow/``，输出 RGB-D。为 ``--output`` 选择尚不存在、位于 ``prepared/`` 之外的目录：

.. code-block:: bash

   cd /path/to/RPent
   python -m toolkits.mapping.raft_to_rgbd \
     --input /path/to/prepared \
     --flow-dir /path/to/flow \
     --output /path/to/rgbd

生成的 ``rgbd/`` 包含一一对应的 ``rgb/*.png`` 和 ``depth/*.png``，以及 ``camera.yaml``。深度为 uint16 毫米，已与 RGB 对齐；无效值和超出 65.535 米的深度置零。

转换使用 ``depth = fx * baseline / disparity``，其中视差取官方 demo 保存光流的相反数，因此图像、RAFT 输出和标定必须保持同一分辨率。文件名保留时间间隔，并使用统一的小数位数。

生成完成后，按 :ref:`mapping-rgbd-source` 配置 RGB、深度、标定和相机位姿。

.. _mapping-config:
.. _mapping-rgbd-source:

3. 配置 RTAB-Map RGB-D 输入
----------------------------

确认数据目录符合上一节的 RGB-D 输入格式后，在 RPent 源码根目录将 ``rtabmap_rgbd.ini`` 复制到新的建图工作目录，命名为 ``rtabmap.ini``：

.. code-block:: bash

   cd /path/to/RPent
   mkdir /path/to/new-map
   cp toolkits/mapping/rtabmap_rgbd.ini /path/to/new-map/rtabmap.ini

``rgbd/`` 存放输入图像与标定，``new-map/`` 存放建图配置、数据库和导出结果。下面的设置均修改工作目录中的 ``rtabmap.ini``。

填写数据路径
~~~~~~~~~~~~

将模板中的路径替换为实际位置。``calibrationName`` 指向与 RGB 图像对应的 ``camera.yaml``：

.. code-block:: ini

   [Core]
   Rtabmap\WorkingDirectory=/path/to/new-map

   [Camera]
   calibrationName=/path/to/rgbd/camera.yaml
   RGBDImages\path_rgb=/path/to/rgbd/rgb
   RGBDImages\path_depth=/path/to/rgbd/depth

以上只展示需要修改的路径，保留模板中的其他参数。模板已选择 RGB-D 图片驱动（``type=0``、``rgbd\driver=7``），毫米深度的缩放因子为 1，从文件名读取时间戳，关闭同步等待，并以速率 0 快速离线导入。

填写相机安装位姿
~~~~~~~~~~~~~~~~

在 ``[Camera]`` 段填写 ``localTransform``，六个数依次为 x/y/z（米）和 roll/pitch/yaw（弧度）。它表示相机相对机器人参考系的固定安装位姿，以相机光心为位置原点、采用 x 向前、y 向左、z 向上的机械坐标轴。

* **RAFT 生成的 RGB-D**：将 ``prepare_stereo.py`` 日志中显示的 ``localTransform=...`` 填入配置。该值由你实测填写的 ``--camera-pose`` 转换得到，已将角度从度转为弧度，并补偿图像校正旋转。
* **SDK 导出或已有 RGB-D**：根据实测安装位姿填写，角度使用弧度；若导出时改变了相机坐标轴方向，还需按最终 RGB 图像的坐标系换算。

模板示例表示相机相对机器人原点高 0.5 米、水平朝前。只有机器人参考原点位于地面时，z 才等于离地高度。使用时替换为实际值：

.. code-block:: ini

   [Camera]
   localTransform=0 0 0.5 0 0 0

安装位姿不需要包含逐帧运动轨迹；下一节的视觉里程计会估计运动。填写完成后，进入 :ref:`mapping-build`。

.. _mapping-build:

4. 生成地图
-----------

完成 RGB-D 输入配置后，执行 `RTAB-Map 官方工具 <https://introlab.github.io/rtabmap/api/latest/tools.html>`_，依次导入图像、估计轨迹并导出优化后的地图。以下命令在 Bash 中运行，日志同时显示在终端并保存在输出目录：

.. code-block:: bash

   cd /path/to/new-map
   set -o pipefail
   export QT_QPA_PLATFORM=offscreen
   rtabmap-dataRecorder -hide rtabmap.ini raw.db 2>&1 | tee import.log
   rtabmap-reprocess -default -odom -c rtabmap.ini raw.db rtabmap.db 2>&1 | tee mapping.log
   rtabmap-export --map --cloud --opt 2 --output map rtabmap.db 2>&1 | tee export.log

这三个命令依次完成以下工作，确认上一条成功后再执行下一条：

1. ``rtabmap-dataRecorder`` 按文件名时间戳导入 RGB-D 图像对，生成 ``raw.db``。
2. ``rtabmap-reprocess -odom`` 从 RGB-D 估计视觉里程计，并在建图时优化轨迹，生成 ``rtabmap.db``。
3. ``rtabmap-export`` 从地图数据库导出二维占据栅格和三维点云。

首次建图可直接进入下一节查看结果。需要额外轨迹文件或平面运动约束时，使用下面的可选命令。

可选处理
~~~~~~~~

**导出优化轨迹（可选）**

从 ``rtabmap.db`` 导出优化后的运动轨迹，生成同目录下的 ``trajectory_poses.txt``：

.. code-block:: bash

   cd /path/to/new-map
   rtabmap-export --poses --poses_format 10 --opt 2 --output trajectory rtabmap.db

``trajectory_poses.txt`` 每行包含 ``stamp x y z qx qy qz qw``，位置单位为米，坐标轴为 x 向前、y 向左、z 向上。

**平面运动约束（可选）**

相机固定在沿平坦地面运动的平台上时，可增加一轮平面运动约束，将结果另存为 ``planar.db``：

.. code-block:: bash

   rtabmap-reprocess -c rtabmap.ini -noimu \
     --Reg/Force3DoF true --RGBD/ForceOdom3DoF true rtabmap.db planar.db
   rtabmap-export --map --cloud --opt 2 --output planar planar.db

5. 查看建图结果
---------------

建图完成后，查看输出文件并检查地图和轨迹。

输出文件
~~~~~~~~

导入、建图和导出命令成功完成后，``/path/to/new-map/`` 中会保存以下文件。其中，轨迹文件由上一节的可选导出命令生成。

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 输出文件
     - 内容与用途
   * - ``rtabmap.db``
     - 原生地图数据库，包含视觉数据和位姿图。保留它，可检查地图、重新处理数据，以及后续配置 RTAB-Map 定位。
   * - ``map.pgm`` + ``map.yaml``
     - 二维占据栅格，描述空闲、障碍和未知区域，并提供分辨率与原点。ROS/Nav2 地图服务器可加载这两个文件。
   * - ``map_cloud.ply``
     - 带颜色的三维点云，用于查看重建场景的形状、覆盖范围和重建伪影。
   * - ``trajectory_poses.txt`` （额外导出）
     - 由轨迹导出命令从 ``rtabmap.db`` 生成，记录优化后位姿的时间戳、位置和朝向，用于查看运动路径。
   * - ``raw.db``
     - 导入的 RGB-D 帧，可用于更换参数后重新建图。
   * - ``import.log``、``mapping.log``、``export.log``
     - 各阶段的命令输出，用于检查进度和排查失败原因。

若执行了平面运动约束处理，还会生成 ``planar.db``、``planar.pgm``、``planar.yaml`` 和 ``planar_cloud.ply``。

轨迹记录建图节点经过优化后的估计位姿，地图和轨迹使用局部地图坐标系。这些输出可用于后续的机器人定位与导航配置。

检查地图
~~~~~~~~

结合实测核对地图尺度、障碍、覆盖范围和未知区域。后续使用 RTAB-Map 进行视觉定位时，保留包含视觉数据的 ``rtabmap.db``。

比较不同深度来源时，保持图像、分辨率、采样、标定、相机位姿、建图和去噪设置一致。

常见问题可按以下方式检查：

* **命令缺失或 Qt 报错**：检查 RTAB-Map 独立工具安装和 ``QT_QPA_PLATFORM`` 设置。
* **图像缺对或尺寸错误**：检查 RGB 与深度图的文件名、尺寸、对齐关系和标定；使用 RAFT 预处理时，还需检查左右图是否成对。
* **RAFT 显存不足**：按单帧分辨率选择显存充足的 GPU；``--step`` 控制采样数量和总处理量。
* **地图为空或碎裂**：检查数据库中的跟踪结果，以及图像的运动模糊、纹理和标定。
* **预处理失败**：查看保留的部分文件排查原因，修正后使用新的输出目录重试。
