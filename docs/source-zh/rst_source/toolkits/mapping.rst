地图构建
========

建图工具集把一次工作空间走查变成可复用的三维地图。它提供 RPC 服务与与传输
无关的客户端，并附带按段构建录像的离线工具和渲染已标注地图的报告。物体标注
与定位是独立工具集，见 :doc:`grounding` 与 :doc:`localization`。

地图是重建结果而非平面图：除非用底盘位姿锚定，其几何只在相似变换意义下正
确，任何距离都继承地图记录的尺度。:ref:`workspace-mapping-frames` 说明每项
结果使用的坐标系与尺度。

准备工作
--------

建图需要 CUDA GPU。一次建图会话默认接收 300 帧，约为 24 GB 显存能重建的规
模；视频构建工具会把更长的录制按段构建，而不是要求更大的设备。

.. code-block:: bash

   uv pip install -e '.[lingbot-map,map-view]'

重建需要一个权重。下载一次并导出路径：

.. code-block:: bash

   # 中国大陆用户可加 HF_ENDPOINT=https://hf-mirror.com。
   hf download robbyant/lingbot-map lingbot-map.pt --local-dir /path/to/checkpoints
   export LINGBOTMAP_CHECKPOINT_PATH=/path/to/checkpoints/lingbot-map.pt

启动建图服务
------------

``toolkits.mapping.server`` 同一时间持有一个建图会话（开始、推送帧、构建、
停止），并把每张地图写到 ``LINGBOTMAP_OUTPUT_DIR`` 之下。

.. code-block:: bash

   LINGBOTMAP_OUTPUT_DIR=/path/to/maps \
       python -m toolkits.mapping.server \
       --transport http --host 127.0.0.1 --port 8116

导出 ``LINGBOTMAP_CHECKPOINT_PATH`` 后，``build`` 会返回 GCT 流式模型算出的
相机位姿与逐帧深度，以及从该深度反投影得到的世界点云。未导出时，服务端改为
归档帧与位姿：构建流程相同，但地图没有深度，无法支撑定位或通行检查。请在
启动服务端的环境中导出该变量。

服务客户端
----------

``toolkits.mapping.client`` 既可用于本地服务，也可用于远程服务：

.. list-table::
   :header-rows: 1

   * - 调用
     - 输入
     - 返回
   * - ``LingBotMapClient.start_session(map_name=...)``
     - 地图名：由字母、数字、``_``、``.`` 和 ``-`` 组成的单个路径段。
     - 会话 id。
   * - ``LingBotMapClient.push_frame(rgb, pose=None, timestamp_s=...)``
     - RGB ``[H,W,3]`` ``uint8`` 数组或 PNG 字节；可选机器人地图坐标系中的
       底盘位姿 ``[x, y, yaw, roll, pitch, yaw_head]``；录制时钟上的秒数。
     - 已接收的帧数。
   * - ``LingBotMapClient.build()``
     - 无。
     - ``map_path``、帧数与 ``map.json`` 元数据；地图目录写在服务所在主机上。
   * - ``LingBotMapClient.load_frame()``、``load_depth()``
     - 地图路径与帧索引。
     - 归档的 PNG 字节，以及深度与置信度数组。

地图路径始终是构建写出的目录，而不是其中的某个文件：``map.json`` 保存元数据
契约，帧与 ``.npy`` 数组与它并列。客户端通过 RPC 读取帧与深度，
``save_annotations`` 把标注写回，因此客户端无需访问构建地图那台机器的文件
系统。

从视频构建地图
--------------

``toolkits.mapping.build_map`` 对录制采样，并按段构建地图，因此超过帧数上限
的走查不需要更大的显卡：

.. code-block:: bash

   python -m toolkits.mapping.build_map \
       --video workspace_walkthrough.mp4 \
       --map-name workspace \
       --endpoint http://127.0.0.1:8116

命令每秒采样五帧（``--fps``），并在 ``--segment-frames`` 帧后结束一个会话，
默认 300，需与服务端的 ``--max-frames`` 一致。能放进一段的录制写为
``maps/workspace``；更长的录制写为 ``maps/workspace_00``、
``maps/workspace_01`` 等，每段一张地图。各段独立重建、彼此不共享位姿，应分
别当作独立的地图。

加上 ``--labels chair desk monitor --perception-endpoint
http://127.0.0.1:8115`` 可对每张地图做物体标注；见
:ref:`workspace-mapping-grounding`。命令打印 JSON 摘要：

.. code-block:: json

   {
     "video": "workspace_walkthrough.mp4",
     "map_name": "workspace",
     "fps": 5.0,
     "segment_frames": 300,
     "maps": [
       {
         "map_name": "workspace_00",
         "map_path": "/path/to/maps/workspace_00",
         "frames": 300,
         "frames_pushed": 300,
         "backend": "gct_stream",
         "has_depth": true
       }
     ]
   }

``--output summary.json`` 把同一份文档写入文件。地图名必须是单个路径段，由字
母、数字、``_``、``.`` 和 ``-`` 组成，并以字母或数字开头；按段编号时会追加
``_00``、``_01`` 等。服务端拒绝构建到已存在的地图目录，避免旧帧和旧标注混入
新的构建。请换一个名字，或先删除旧目录。

构建出的地图目录如下：

.. code-block:: text

   maps/workspace/
   ├── frames/                 archived source frames (000000.png, ...)
   ├── poses.json              per-frame base poses that were pushed
   ├── keyframe_index.json     which frames the reconstruction treats as keyframes
   ├── map.json                the metadata contract described below
   ├── depth.npy               per-frame depth          (GCT backend)
   ├── world_points.npy        per-frame world points   (GCT backend)
   ├── world_points_conf.npy   per-point confidence      (GCT backend)
   ├── cameras_c2w.npy         camera-to-world poses     (GCT backend)
   ├── cameras_c2w_robot_frame.npy      aligned poses    (with alignment)
   ├── world_points_robot_frame.npy     aligned points   (with alignment)
   ├── objects.json            grounded instances        (after grounding)
   ├── annotations/            per-frame mask overlays   (after grounding)
   ├── semantic_points.npz     labelled 3D points        (after grounding)
   ├── semantic_points.ply     the same cloud for external viewers (after grounding)
   ├── evaluation.json         grounding consistency     (after grounding)
   └── localization_eval.json  replay metrics            (after evaluation)

``map.json`` 是后续阶段读取的契约。会改变结果解读方式的字段有：

.. list-table::
   :header-rows: 1

   * - 字段
     - 含义
   * - ``frame``
     - 相机位姿所在的坐标系。``model`` 表示重建自身的坐标系。
   * - ``backend``
     - ``gct_stream`` 表示重建，``frame_archive`` 表示只有帧。
   * - ``has_depth``、``has_world_points``
     - 是否存在重建数组。
   * - ``gravity``
     - 模型坐标系中估计的朝上方向（``status``、``up``）。
   * - ``floor``
     - 模型坐标系中的地面平面：``normal``、``offset``，以及地面到相机的距离
       ``camera_height``。
   * - ``scale``
     - ``status`` 为 ``metric`` 或 ``arbitrary``；已知时给出
       ``meters_per_unit``。
   * - ``cameras``、``camera_to_world``
     - 内参、外参，以及保存相机到世界位姿的文件。
   * - ``alignment``
     - 对推入底盘位姿的拟合结果，或带原因的 ``unavailable``。
   * - ``keyframes``
     - 关键帧索引文件及数量。

不使用机器人测试建图服务
------------------------

建图服务端响应普通 RPC 调用，因此接入运行前用几行 Python 就能确认它是否正
常。推入合成帧可以走通会话与构建路径：

.. code-block:: python

   import numpy as np
   from rpent.utils.rpc.http_rpc import HttpRpcClient
   from toolkits.mapping.client import LingBotMapClient

   client = LingBotMapClient(HttpRpcClient("http://127.0.0.1:8116"))
   client.start_session(map_name="smoke_test")
   for index in range(10):
       frame = np.zeros((480, 640, 3), dtype=np.uint8)
       frame[:, :, index % 3] = 60 + index  # 合成帧
       client.push_frame(frame, pose=None, timestamp_s=index * 0.5)
   result = client.build()
   print(result.map_path, result.frames)  # .../maps/smoke_test 10

构建成功会打印地图目录和帧数。若要丢弃会话，改调 ``client.stop_session()``。

要得到可检查的地图，可用上文介绍的离线命令直接从走查视频构建：

.. code-block:: bash

   python -m toolkits.mapping.build_map \
       --video workspace_walkthrough.mp4 \
       --map-name from_video \
       --endpoint http://127.0.0.1:8116

录制时相机大致朝前。该命令不推送底盘位姿，因此地图保持在模型坐标系中，尺度
任意。可对生成的目录运行 :mod:`toolkits.grounding.ground_map` 添加物体标注。

.. _workspace-mapping-view:

查看已构建的地图
----------------

``toolkits.mapping.view`` 把地图渲染为一个报告目录，无需加载重建数组即可查
看：

.. code-block:: bash

   python -m toolkits.mapping.view --map-path /path/to/maps/workspace

报告包含以下文件：``report/semantic_topdown.png`` 是按估计重力方向定向的俯视
图，没有重力估计时改用相机轨迹；``report/semantic_3d.png`` 是带相机轨迹的等
距视图；``report/detections.html`` 给出逐标签计数、物体与检测表格，以及全部
逐帧叠加图；``report/objects.csv`` 每个实例一行。俯视图以相机轨迹为 ``x``/``y``
中心，并把估计的地面置于 ``z = 0``。坐标轴与表格以地图单位标注；地图记录了
米制锚点时，尺度说明会给出一个单位对应多少米。``--max-points`` 限制每幅图
绘制的点数，``--output-dir`` 可把报告写到地图目录之外。

.. _workspace-mapping-frames:

坐标系与尺度
------------

**坐标系。** 相机位姿、点云与定位结果都使用 ``map.json`` 记录的坐标系，即重
建自身的 ``model`` 坐标系。重力方向与地面平面也在该坐标系中。

每一帧都携带底盘位姿时，建图服务端还会把重建出的相机轨迹与这些位置拟合一个
相似变换，并写入 ``alignment``：``scale`` 为每个模型单位对应的机器人地图单
位，另有 ``rotation``、``translation``、``rms_error_m``，以及使用了多少帧、
多少帧没有位姿。机器人坐标系下的位姿副本与点云单独保存；模型坐标系数组仍是
定位与通行检查的权威来源。拟合后的坐标系把相机轨迹放在 ``z = 0`` 上、``+z``
朝上，因此地面位于 ``z = 0`` 之下。

**尺度。** 只有底盘位姿锚定过的地图才是米制的（``scale.status`` 为
``metric``）。否则距离是任意的模型单位，米制距离没有意义。定位与通行检查
使用地图单位报告距离，除非已知尺度；通行检查可用的单位见
:doc:`localization`。

.. _workspace-mapping-limits:

限制
----

**地图在帧数上限处停止。** 会话在达到 ``--max-frames`` 后拒绝新帧，默认
300，约为 24 GB 显存能重建的规模。一次 400 帧的构建在 24 GB 设备上耗尽显
存。``toolkits.mapping.build_map`` 会自动把更长的录像切成多段。实时会话必须
在上限处停止，因此请降低 ``--fps``、在更大的设备上调高 ``--max-frames``，或
改为从视频重建。显存耗尽失败后，下一次构建前先重启建图服务。

**各段是独立的地图。** 每段单独重建并保留自己的模型坐标系，因此一段的位姿
不能直接用于另一段。

**帧归档地图没有几何。** 未启用 GCT 权重时，地图只保存帧与位姿。它可以做二
维标注，也可以用于定位，但没有深度、重力方向或地面平面，通行检查会报告
``unusable``。
