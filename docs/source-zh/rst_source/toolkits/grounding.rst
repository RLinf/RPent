物体标注
========

标注工具集为已构建的地图添加开放词汇的物体标签。感知服务在 GPU 上运行
Grounding DINO 与 SAM 2，离线标注工具把得到的掩码投射到地图深度上、关联成
实例，并把标注写回地图目录。地图本身来自 :doc:`mapping`。

准备工作
--------

标注需要 CUDA GPU 与 ``grounding-sam2`` extra：

.. code-block:: bash

   uv pip install -e '.[grounding-sam2]'

它还需要两个权重。Grounding DINO 必须使用官方 release 的原始权重——Hugging
Face 上 transformers 格式的 ``model.safetensors`` 无法加载。

.. code-block:: bash

   wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth \
       -P /path/to/checkpoints
   hf download facebook/sam2.1-hiera-large sam2.1_hiera_large.pt \
       --local-dir /path/to/checkpoints

   export GDINO_CHECKPOINT_PATH=/path/to/checkpoints/groundingdino_swint_ogc.pth
   export SAM2_CHECKPOINT_PATH=/path/to/checkpoints/sam2.1_hiera_large.pt

不要改动权重文件名：感知服务端会根据文件名选择 SAM 2 的模型配置。首次启动
时它还会下载 ``bert-base-uncased`` 分词器，因此这一步需要网络。

启动感知服务
------------

``toolkits.grounding.server`` 把 Grounding DINO（开放词汇检测）与
SAM 2（掩码细化）封装在一次 ``gdino_sam2.detect_segment`` 调用背后。

.. code-block:: bash

   python -m toolkits.grounding.server \
       --transport http --host 127.0.0.1 --port 8115

服务客户端
----------

``toolkits.grounding.client`` 既可用于本地服务，也可用于远程服务：

.. list-table::
   :header-rows: 1

   * - 调用
     - 输入
     - 返回
   * - ``GdinoSam2Client.detect_segment(image, labels=...)``
     - RGB 帧与最多 16 个标签。
     - 带得分的框与逐检测的实例掩码。

.. _workspace-mapping-grounding:

标注物体
--------

标注可以在任何一次构建之后单独运行。``toolkits.grounding.ground_map`` 读取
已构建的地图，均匀选取关键帧（或用 ``--keyframes`` 指定索引），并把标注写回
地图：

.. code-block:: bash

   python -m toolkits.grounding.ground_map \
       --map-path /path/to/maps/workspace \
       --labels chair desk monitor \
       --perception-endpoint http://127.0.0.1:8115 \
       --max-keyframes 12

地图路径就是构建写出的目录。在重建地图上，每个掩码先按服务端的缩放和中心裁
剪映射到深度网格，再用该帧的内参与相机位姿反投影，并与其他关键帧上同标签的
观测关联。没有深度的地图只保留二维标注，此时需要 ``--allow-2d`` 才能写入。

.. list-table::
   :header-rows: 1

   * - 参数
     - 默认值
     - 作用
   * - ``--max-keyframes``
     - 12
     - 均匀选取多少个关键帧做标注。
   * - ``--keyframes``
     - 均匀选取
     - 显式指定帧索引，用逗号或空格分隔。
   * - ``--min-score``
     - 0.35
     - 保留的最低检测得分。
   * - ``--min-mask-pixels``
     - 32
     - 掩码的最小面积（源像素）。
   * - ``--min-confidence``
     - 1.0
     - 反投影一个像素所需的最低深度置信度。
   * - ``--merge-factor``
     - 0.75
     - 同标签质心距离小于两者典型半径的该倍数时合并为一个实例。
   * - ``--output-dir``
     - 地图目录
     - 把产物写到别处，保持地图不变。
   * - ``--allow-2d``
     - 关闭
     - 在没有深度的地图上保留二维标注。

命令写出 ``objects.json``、逐帧掩码叠加图目录 ``annotations/``、带标签的
``semantic_points.npz`` 与 ``semantic_points.ply``，以及 ``evaluation.json``。
标注通过 ``save_annotations`` 写回，因此客户端无需访问构建地图那台机器的文
件系统。

``evaluation.json`` 给出每个标签的计数，以及不同帧中最近同标签观测距离的中
位数。该距离衡量跨帧一致性；精确率和召回率需要独立的真值标注，因此不作报告。

限制
----

模型几何和检测器带来两点限制。完全落在模型中心裁剪之外的掩码没有深度，只保
留二维检测——归档帧保留了完整画面，但竖屏画面在裁剪时会丢掉正方形视野上下的
行。Grounding DINO 也可能对同一个物体返回多个重叠掩码；关联只在三维质心落入
半径阈值时才合并观测，因此关键帧较稀疏时，同一个物体可能留下多条记录。标签
匹配不区分大小写，而 slug 必须唯一，因此不能同时请求 ``Chair`` 和 ``chair``。

查看结果
--------

``python -m toolkits.mapping.view`` 把逐标签计数、物体表格与检测叠加图渲染到
报告目录；见 :ref:`查看已构建的地图 <workspace-mapping-view>`。带标签的点云
也可以通过 ``semantic_points.ply`` 在外部查看器中打开。
