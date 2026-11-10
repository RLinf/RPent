.. _offline-mapping:

离线建图
========

.. note::

   本建图方案仅在 ZED 2i 上验证过，使用 LeKiwi 采集的数据。

使用 RTAB-Map 对双目图片或左右并排的视频进行离线建图，生成地图数据库、二维占据栅格和彩色三维点云，并可导出优化后的运动轨迹。

如果希望使用相机 SDK 或 RAFT-Stereo 计算的深度，见 :ref:`mapping-custom-depth`：将 RGB 和对齐深度作为 RGB-D 输入，复用相同的建图命令。

安装 RTAB-Map
-------------

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

使用 RTAB-Map 双目建图
----------------------

准备输入与标定
~~~~~~~~~~~~~~

准备以下数据：

1. **同步的左右图**：每对图像来自同一时刻，分别放入 ``left/`` 和 ``right/``，并使用相同文件名。
2. **双目标定**：将相机或数据集提供的标定保存为 RTAB-Map 格式的 ``camera_left.yaml`` 和 ``camera_right.yaml``，放在同一目录。标定中的单目尺寸、内参、畸变和左右相机位置关系应与输入图像对应；右目投影矩阵需包含实际基线（两相机之间的距离）。RTAB-Map 据此将左右图的像素差换算为距离。
3. **时间戳文件名**：以秒为单位，按拍摄时间递增，起始值大于 0，小数位数保持一致。例如每秒采集 5 对图像，可命名为 ``1.0.png``、``1.2.png``、``1.4.png``。使用相对时间时，保留实际采集间隔。

图片目录示例：

.. code-block:: text

   stereo/
     left/1.0.png
     left/1.2.png
     right/1.0.png
     right/1.2.png
     camera_left.yaml
     camera_right.yaml

使用视频时，准备恒定帧率、当前解码器可读取的左右并排视频，例如 ``/path/to/stereo.mp4``。每帧左半边为左目、右半边为同一时刻的右目。2560 × 720 视频对应每目 1280 × 720 的标定。

示例：为 ZED 2i 录制生成标定文件
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

以 LeKiwi 通过 UVC 采集的 ZED 2i 数据为例，假设录制目录为 ``/path/to/capture``。其中 ``left/`` 和 ``right/`` 保存原始左右目 PNG。下面将这台相机的出厂标定转换为 ``camera_left.yaml`` 和 ``camera_right.yaml``，供 RTAB-Map 使用。

**1. 找到录制对应的出厂标定**

按 `Stereolabs 标定说明 <https://docs.stereolabs.com/docs/development/zed-sdk/modules/camera/camera-calibration>`_，根据拍摄这组图像的相机序列号获取出厂标定 ``SN<serial>.conf``，保存为 ``/path/to/capture/calibration.conf``。

本示例采集时使用的 ``zed.conf`` 就是该出厂标定的本地副本，也可直接将它复制到录制目录并命名为 ``calibration.conf``。

**2. 确认单目图像尺寸**

查看 ``left/`` 中一张原始图片的宽、高，并确认右图尺寸相同。例如左图为 1280 × 720，就填写 ``--image-size 1280 720``；若输入为 2560 × 720 的左右并排视频，每目的尺寸也是 1280 × 720。

脚本支持 1280 × 720（``HD``）和 672 × 376（``VGA``）。使用录制图像的原始尺寸。

**3. 运行转换脚本**

在 RPent 源码根目录，使用安装了 NumPy、OpenCV 和 PyYAML 的 Python 环境执行：

.. code-block:: bash

   python -m toolkits.mapping.zed_calibration \
     --calibration /path/to/capture/calibration.conf \
     --image-size 1280 720 \
     --output /path/to/capture

命令会在 ``--output`` 目录生成 ``camera_left.yaml`` 和 ``camera_right.yaml``。选择尚无这两个文件的输出目录；重复转换时可换用新目录。生成的 YAML 包含原始内参、畸变、双目校正矩阵和投影矩阵，平移单位为米，右目投影矩阵包含实际基线。

**4. 将结果填入建图配置**

按下一节 :ref:`mapping-config` 复制图片配置后，将 ``rtabmap.ini`` 中的以下项设为：

.. code-block:: ini

   calibrationName=/path/to/capture/camera_left.yaml
   StereoImages\path_left=/path/to/capture/left
   StereoImages\path_right=/path/to/capture/right
   StereoImages\rectify=true

这里的 ``true`` 让 RTAB-Map 按生成的标定校正原始左右图。若使用原始左右并排视频，也按实际单目尺寸转换出厂标定，并在视频配置中开启这一校正选项。

.. _mapping-config:

写入建图配置
~~~~~~~~~~~~

根据输入类型选择一个配置，在 RPent 源码根目录执行对应命令。

**双目图片：** 使用 ``rtabmap_stereo_images.ini``。``type=1`` 选择双目，``stereo\driver=2`` 选择图片驱动；从文件名读取时间戳、关闭同步等待，并以 ``imgRate=0`` 快速离线导入。

.. code-block:: bash

   mkdir /path/to/new-map
   cp toolkits/mapping/rtabmap_stereo_images.ini /path/to/new-map/rtabmap.ini

**双目视频：** 使用 ``rtabmap_stereo_video.ini``。``stereo\driver=3`` 选择视频驱动，空的 ``StereoVideo\path2`` 表示单文件左右并排输入。将 ``imgRate=15`` 改为实际视频帧率。

.. code-block:: bash

   mkdir /path/to/new-map
   cp toolkits/mapping/rtabmap_stereo_video.ini /path/to/new-map/rtabmap.ini

编辑复制后的 ``rtabmap.ini``，填写以下内容：

* **路径**：将所有 ``/path/to/...`` 替换为实际工作目录、输入和标定路径。``calibrationName`` 指向 ``camera_left.yaml``。
* **图像校正**：原始图像使用原始标定，并设置 ``StereoImages\rectify=true``；已校正图像使用对应的校正后标定，保留默认值 ``false``。图片和视频均使用此配置项。
* **相机安装位姿**：将 ``localTransform`` 改为实际 x/y/z（米）和 roll/pitch/yaw（弧度）。它表示相机机械坐标系在地面参考系中的位姿：x 向前、y 向左、z 向上，地面 z=0。模板的 ``0 0 0.5 0 0 0`` 表示相机水平朝前、离地 0.5 米。

.. _mapping-build:

生成地图
~~~~~~~~

执行 `RTAB-Map 官方工具 <https://introlab.github.io/rtabmap/api/latest/tools.html>`_，依次导入图像、估计轨迹并导出优化后的地图。以下命令在 Bash 中运行，日志同时显示在终端并保存在输出目录：

.. code-block:: bash

   cd /path/to/new-map
   set -o pipefail
   export QT_QPA_PLATFORM=offscreen
   rtabmap-dataRecorder -hide rtabmap.ini raw.db 2>&1 | tee import.log
   rtabmap-reprocess -default -odom -c rtabmap.ini raw.db rtabmap.db 2>&1 | tee mapping.log
   rtabmap-export --map --cloud --opt 2 --output map rtabmap.db 2>&1 | tee export.log

``-odom`` 从连续图像估计相机运动，生成与图像对应的视觉里程计；建图时进一步优化轨迹。图片按文件名时间戳导入，视频按配置中的帧率读取。

相机固定在沿平坦地面运动的平台上时，可增加一轮平面运动约束，将结果另存为 ``planar.db``：

.. code-block:: bash

   rtabmap-reprocess -c rtabmap.ini -noimu \
     --Reg/Force3DoF true --RGBD/ForceOdom3DoF true rtabmap.db planar.db
   rtabmap-export --map --cloud --opt 2 --output planar planar.db

.. _mapping-custom-depth:

可选：使用自己的深度
--------------------

可使用相机 SDK 导出的深度，或通过 RAFT-Stereo 从双目图像生成深度。将 RGB 和深度整理为下述格式后接入 RTAB-Map。

RGB-D 输入格式
~~~~~~~~~~~~~~

准备以下文件：

* **RGB 图像**：去畸变后的 PNG。
* **深度图像**：与 RGB 同尺寸、逐像素对齐的单通道 uint16 PNG。像素值表示沿相机光轴的 Z 距离，单位为毫米，无效像素为 0。
* **相机标定**：RTAB-Map/OpenCV 格式的 ``camera.yaml``，包含对应 RGB 的尺寸和内参，供 RTAB-Map 将深度像素转换为三维点。

每对 RGB 和深度使用相同文件名，以递增、大于 0 的秒时间戳命名，并保持小数位数一致：

.. code-block:: text

   rgbd/
     rgb/1.0.png
     rgb/1.2.png
     depth/1.0.png
     depth/1.2.png
     camera.yaml

使用数据源对应的工具导出图像和标定，并将配置中的 ``calibrationName`` 指向实际标定文件。RAFT-Stereo 路线会生成对应的 ``camera.yaml``。

使用相机 SDK 的深度
~~~~~~~~~~~~~~~~~~~

按厂商官方环境和导出流程，从已有录制中导出同一帧的去畸变 RGB、对齐的数值深度，以及对应分辨率的相机内参。将输出整理为上面的 RGB-D 目录：

1. 将数值深度换算为毫米，保存为单通道 uint16 PNG；无效值和超出 65.535 米的值置零。
2. 保持 RGB 与深度同尺寸、逐像素对齐，并使用相同文件名；提供与导出 RGB 对应的 RTAB-Map 标定 YAML。
3. 执行 :ref:`mapping-rgbd-source`。

例如，ZED SDK 的 `官方深度示例 <https://github.com/stereolabs/zed-sdk/blob/master/tutorials/tutorial%203%20-%20depth%20sensing/python/depth_sensing.py>`_ 使用 ``VIEW.LEFT`` 读取左图、``MEASURE.DEPTH`` 读取对齐深度。按上述格式保存连续图像和对应标定即可接入。

使用 RAFT-Stereo 生成深度
~~~~~~~~~~~~~~~~~~~~~~~~~

使用同步的左右 RGB 图像和对应双目标定，通过 RAFT-Stereo 计算像素位移，再按焦距和基线换算为毫米深度。处理顺序如下：

.. code-block:: text

   双目图片或左右并排视频 + stereo.yaml
     → prepare_stereo.py：校正、整理左右图 → prepared/
     → RAFT-Stereo 官方 demo：估计像素位移 → flow/*.npy
     → raft_to_rgbd.py：根据焦距和基线换算深度 → rgbd/
     → RTAB-Map：使用 RGB-D 输入建图

第一步：用已有标定校正左右图
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在 RPent 源码根目录，使用包含 NumPy、SciPy、PyYAML 和 OpenCV 的 Python 环境。输入二选一：

* **图片目录**：``sequence/left/`` 和 ``sequence/right/``，两侧 PNG、JPEG 或 TIFF 文件名完全对应。
* **双目视频**：本地 MP4、AVI、MKV 或 MOV，同步左右图并排排列，左图在前；解码器需支持对应编码。

先复制仓库中的标定模板：

.. code-block:: bash

   cp toolkits/mapping/stereo.yaml /path/to/stereo.yaml

在 ``stereo.yaml`` 中填入相机或数据集提供的实际标定：

* ``image_size``：单目图像宽、高。例如 2560 × 720 的左右并排视频填写 ``[1280, 720]``。
* 左右目的 ``K`` 和 ``D``：OpenCV 针孔模型的内参和畸变。
* ``R`` 和 ``T_m``：从左目光学坐标到右目光学坐标的变换，满足 ``X_right = R * X_left + T_m``，平移单位为米。

使用右相机位于左相机右侧的水平针孔双目。已校正图像填写校正后内参、零畸变、单位旋转矩阵和 ``T_m: [-baseline_m, 0, 0]``。鱼眼图像先用相机工具导出为校正后的针孔双目图像。

**图片输入示例：** 以下假设图片以补零编号命名、采样率为 5 Hz。``--camera-pose`` 是原始左相机在地面参考系中的位置（x/y/z，米）和机械坐标系姿态（roll/pitch/yaw，度），参考系 x 向前、y 向左、z 向上。将示例中离地 0.5 米、水平朝前的位姿替换为实际值。

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

两种输入得到相同结构的 ``prepared/``：校正后的 ``pairs/<timestamp>/im0.png`` （左图）和 ``im1.png`` （右图），以及 ``camera.yaml``、``local_transform.txt``、``stereo.json``。后面所有命令都使用这个目录。输出目录必须尚不存在且位于输入目录之外；时间戳从 1 秒开始，保留采样间隔。输出位姿已补偿左目校正旋转，角度单位为弧度。

第二步：运行官方 RAFT-Stereo
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

克隆官方仓库：

.. code-block:: bash

   git clone https://github.com/princeton-vl/RAFT-Stereo.git /path/to/RAFT-Stereo
   cd /path/to/RAFT-Stereo

按上游 README 的 `Requirements <https://github.com/princeton-vl/RAFT-Stereo#requirements>`_ 准备并激活 RAFT 运行环境，在 RAFT-Stereo 仓库目录下载官方权重：

.. code-block:: bash

   bash download_models.sh

权重解压在 ``models/`` 下。将下面的 ``--restore_ckpt`` 替换为下载得到的 ``raftstereo-middlebury.pth`` 实际路径，并选择可用的 NVIDIA GPU。该示例使用标准权重和官方 demo 默认网络结构。

.. code-block:: bash

   cd /path/to/RAFT-Stereo
   CUDA_VISIBLE_DEVICES=0 python demo.py \
     --restore_ckpt /path/to/raftstereo-middlebury.pth \
     --left_imgs "/path/to/prepared/pairs/*/im0.png" \
     --right_imgs "/path/to/prepared/pairs/*/im1.png" \
     --save_numpy \
     --output_directory /path/to/flow

``--left_imgs`` 和 ``--right_imgs`` 指向第一步生成的校正图像，保留引号，让 demo 展开通配符。``--save_numpy`` 将每对图像的像素位移保存为 ``flow/<timestamp>.npy``，同时生成可视化图。使用新的 ``flow/`` 输出目录。

第三步：转换为深度并交给 RTAB-Map
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

切回第一步使用的 RPent Python 环境。下面命令同时读取 ``prepared/`` 中的图像、标定，以及第二步生成的 ``flow/``，输出 RGB-D：

.. code-block:: bash

   cd /path/to/RPent
   python -m toolkits.mapping.raft_to_rgbd \
     --input /path/to/prepared \
     --flow-dir /path/to/flow \
     --output /path/to/rgbd

生成的 ``rgbd/`` 包含一一对应的 ``rgb/*.png`` 和 ``depth/*.png``，以及 ``camera.yaml``、``local_transform.txt``。深度为 uint16 毫米，已与 RGB 对齐；无效值和超出 65.535 米的深度置零。输出目录必须尚不存在。

转换使用 ``depth = fx * baseline / disparity``，其中视差取官方 demo 保存光流的相反数，因此图像、RAFT 输出和标定必须保持同一分辨率。文件名保留时间间隔，并使用统一的小数位数。

最后执行下一节 :ref:`mapping-rgbd-source`：将 ``rtabmap_rgbd.ini`` 的 RGB、深度和标定路径指向该 ``rgbd/``，使用输出的相机位姿，再运行相同的 RTAB-Map 建图命令。

.. _mapping-rgbd-source:

将深度接入 RTAB-Map
~~~~~~~~~~~~~~~~~~~

无论深度来自 SDK 还是 RAFT，使用 ``rtabmap_rgbd.ini``，其中已包含通用建图参数。在 RPent 源码根目录执行以下命令，复制到新的输出目录；参数说明见 :ref:`mapping-config`。``type=0``、``rgbd\driver=7`` 选择 RGB-D 图片驱动，毫米深度的缩放因子为 1，从文件名读取时间戳并关闭同步等待，以速率 0 快速导入：

.. code-block:: bash

   mkdir /path/to/new-map
   cp toolkits/mapping/rtabmap_rgbd.ini /path/to/new-map/rtabmap.ini

将示例路径和 ``localTransform`` 改为实际值。RAFT 路线使用 ``raft_to_rgbd.py`` 输出的 ``camera.yaml``，将 ``local_transform.txt`` 中的六个数填入 ``localTransform``；其中角度已转换为弧度并补偿图像校正旋转。可在终端读取该文件：

.. code-block:: bash

   cat /path/to/rgbd/local_transform.txt

SDK 路线使用导出 RGB 对应的标定，并在 ``localTransform`` 中填写实际相机位姿。

接着执行 :ref:`mapping-build` 的相同命令。此时 ``raw.db`` 保存 RGB-D 帧，``-odom`` 从 RGB-D 重新计算视觉里程计。切换深度来源做对比时，保持图像、分辨率、采样、标定、相机位姿、建图和去噪设置一致。

最终会得到什么
--------------

导入、建图和导出命令成功完成后，``/path/to/new-map/`` 中会保存以下文件。其中，优化后的轨迹需要执行下一节的额外导出命令。

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
     - 建图位姿的时间戳、位置和朝向，可用于绘制和检查重建出的运动路径。
   * - ``raw.db``
     - 导入的双目或 RGB-D 帧，可用于更换参数后重新建图。
   * - ``import.log``、``mapping.log``、``export.log``
     - 各阶段的命令输出，用于检查进度和排查失败原因。

轨迹记录建图节点经过优化后的估计位姿，地图和轨迹使用局部地图坐标系。这些输出可用于后续的机器人定位与导航配置。

检查地图
--------

在终端检查输出文件、地图分辨率与原点、建图日志，并导出优化后的轨迹：

.. code-block:: bash

   cd /path/to/new-map
   ls -lh raw.db rtabmap.db map.pgm map.yaml map_cloud.ply
   cat map.yaml
   tail -n 30 import.log mapping.log export.log
   rtabmap-export --poses --poses_format 10 --opt 2 --output trajectory rtabmap.db
   head trajectory_poses.txt

``trajectory_poses.txt`` 每行包含 ``stamp x y z qx qy qz qw``，位置单位为米，坐标轴为 x 向前、y 向左、z 向上。检查日志中的跟踪丢失和回环记录，并结合实测核对地图尺度、障碍、覆盖范围和未知区域。后续使用 RTAB-Map 进行视觉定位时，保留包含视觉数据的 ``rtabmap.db``。

常见问题可按以下方式检查：

* **命令缺失或 Qt 报错**：检查 RTAB-Map 独立工具安装和 ``QT_QPA_PLATFORM`` 设置。
* **图像缺对或尺寸错误**：检查左右图文件名、图像尺寸与标定是否一致。
* **RAFT 显存不足**：按单帧分辨率选择显存充足的 GPU；``--step`` 控制采样数量和总处理量。
* **地图为空或碎裂**：查看日志和数据库，检查跟踪丢失、运动模糊、弱纹理及标定。
* **预处理失败**：查看保留的部分文件排查原因，修正后使用新的输出目录重试。
