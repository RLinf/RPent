工作空间建图与通行检查
========================

一次工作空间的走查可以变成可复用的三维地图。RPent 从机器人或相机录制的帧
中构建地图，为地图添加物体标注，把实时相机帧定位到地图中，并在底盘前进前
报告前方地面是否畅通。建图服务、标注与定位工具以及通行检查都与具体机器人
无关；只要这些服务可达，机器人的工具箱就会把它们接入。

本页涵盖：

1. 安装 extra 与模型权重。
2. 启动建图、感知与定位服务。
3. 在操作者遥控时构建地图。
4. 在已构建的地图上标注物体并查看结果。
5. 定位归档帧或实时帧，并评估定位。
6. 从终端、RPC 调用或 agent 工具检查通行。
7. 正确解读坐标系与单位，并排查常见故障。

地图是重建结果而非平面图：除非用底盘位姿锚定，其几何只在相似变换意义下正
确，任何距离都继承地图记录的尺度。下文说明每项结果使用的坐标系与单位。

准备工作
--------

建图、标注、定位与通行检查都需要 CUDA GPU。300 帧的地图可以放进 24 GB 显
存；更长的录制需要降低帧数或换用更大的设备。按需安装对应的 extra：

.. list-table::
   :header-rows: 1

   * - extra
     - 启用的能力
   * - ``lingbot-map``
     - 用 LingBot-Map GCT 模型做重建，而不是只归档帧
   * - ``grounding-sam2``
     - 感知服务背后的 Grounding DINO 检测与 SAM 2 掩码
   * - ``localize``
     - DISK + LightGlue 定位与基于地图的通行检查
   * - ``map-view``
     - 已标注地图的离线报告

.. code-block:: bash

   uv pip install -e '.[lingbot-map,grounding-sam2,localize,map-view]'

重建需要一个权重。下载一次并导出路径：

.. code-block:: bash

   # 中国大陆用户可加 HF_ENDPOINT=https://hf-mirror.com。
   hf download robbyant/lingbot-map lingbot-map.pt --local-dir /path/to/checkpoints
   export LINGBOTMAP_CHECKPOINT_PATH=/path/to/checkpoints/lingbot-map.pt

物体标注另需两个权重。Grounding DINO 必须使用官方 release 的原始权重——
Hugging Face 上 transformers 格式的 ``model.safetensors`` 无法加载。

.. code-block:: bash

   wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth \
       -P /path/to/checkpoints
   hf download facebook/sam2.1-hiera-large sam2.1_hiera_large.pt \
       --local-dir /path/to/checkpoints

   export GDINO_CHECKPOINT_PATH=/path/to/checkpoints/groundingdino_swint_ogc.pth
   export SAM2_CHECKPOINT_PATH=/path/to/checkpoints/sam2.1_hiera_large.pt

不要改动权重文件名：感知服务端会根据文件名选择 SAM 2 的模型配置。首次启动
时它还会下载 ``bert-base-uncased`` 分词器，因此这一步需要网络。

启动服务
--------

每项能力都是独立的 RPC 服务，因此一台 GPU 机器可以为只有网络连接的机器人
托管它们。按需启动。

**建图。** ``rpent.robots.components.lingbotmap_server`` 同一时间持有一个建
图会话（开始、推送帧、构建、停止），并把每张地图写到
``LINGBOTMAP_OUTPUT_DIR`` 之下。

.. code-block:: bash

   LINGBOTMAP_OUTPUT_DIR=/path/to/maps \
       python -m rpent.robots.components.lingbotmap_server \
       --transport http --host 127.0.0.1 --port 8116

导出 ``LINGBOTMAP_CHECKPOINT_PATH`` 后，``build`` 会返回 GCT 流式模型算出的
相机位姿与逐帧深度，以及从该深度反投影得到的世界点云。未导出时，服务端改为
归档帧与位姿：工具流程相同，但地图没有深度，无法支撑定位或通行检查。请在
启动服务端的环境中导出上述变量。

**感知。** ``rpent.robots.components.gdino_sam2_server`` 把 Grounding DINO
（开放词汇检测）与 SAM 2（掩码细化）封装在一次
``gdino_sam2.detect_segment`` 调用背后。

.. code-block:: bash

   python -m rpent.robots.components.gdino_sam2_server \
       --transport http --host 127.0.0.1 --port 8115

**定位与通行检查。** ``rpent.robots.components.localize_server`` 加载一张已
构建的地图，估计查询帧的位姿，并从该位姿检查一条走廊。DISK 与 LightGlue 在
首次使用时下载权重。

.. code-block:: bash

   python -m rpent.robots.components.localize_server \
       --map-path /path/to/maps/workspace \
       --transport http --host 127.0.0.1 --port 8117

当地图没有米制尺度、而结果需要以米表示时，传入 ``--camera-height-m``；详见
:ref:`workspace-mapping-frames`。

从 agent 工具集调用
~~~~~~~~~~~~~~~~~~~

``rpent.tools.mapping.MappingMixin`` 把面向 agent 的工具加入机器人工具集，
并且只在对应客户端已经接入时注册该工具：

.. list-table::
   :header-rows: 1

   * - 客户端
     - 增加的工具
   * - ``LingBotMapClient``
     - ``build_map``：开启会话、推送实时帧并完成构建。
   * - ``GdinoSam2Client``
     - ``detect_objects``，以及 ``build_map`` 内的标签标注。
   * - ``LocalizeClient``
     - 针对服务端所加载地图的 ``check_clearance``。

工具集通过 ``_map_frame()`` 提供每一步同步的 RGB 帧；机器人上报里程计时还会
提供底盘位姿。这里不会拉起任何服务：先启动上面的服务端，再接入指向它们的
客户端。未接入客户端时，工具不会出现在模型的工具列表中，避免模型调用必然失败
的接口。

构建地图
--------

当工具集暴露 ``build_map`` 时，agent 通过该工具构建地图。工具开启会话，请操作
者缓慢地遥控机器人绕行工作空间，并在操作者驾驶期间每 0.5 秒推送一次同步的
相机帧（机器人上报里程计时还会推送底盘位姿）。操作者回复后，工具完成构建。
若给出了标签，它随后从已构建的地图读回关键帧，并用感知服务做标注。见
:ref:`workspace-mapping-grounding`。

地图名必须是单个路径段，由字母、数字、``_``、``.`` 和 ``-`` 组成，并以字
母或数字开头。服务端拒绝构建到已存在的地图目录，避免旧帧和旧标注混入新的构
建。请换一个名字，或先删除旧目录。

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

录制超过帧数上限会以明确的错误结束；见 :ref:`workspace-mapping-limits`。

不使用机器人测试建图服务
------------------------

建图服务端响应普通 RPC 调用，因此接入运行前用几行 Python 就能确认它是否正
常。推入合成帧可以走通会话与构建路径：

.. code-block:: python

   import numpy as np
   from rpent.robots.components.lingbotmap_client import LingBotMapClient
   from rpent.utils.rpc.http_rpc import HttpRpcClient

   client = LingBotMapClient(HttpRpcClient("http://127.0.0.1:8116"))
   client.start_session(map_name="smoke_test")
   for index in range(10):
       frame = np.zeros((480, 640, 3), dtype=np.uint8)
       frame[:, :, index % 3] = 60 + index  # 合成帧
       client.push_frame(frame, pose=None, timestamp_s=index * 0.5)
   result = client.build()
   print(result.map_path, result.frames)  # .../maps/smoke_test 10

构建成功会打印地图目录和帧数。若要丢弃会话，改调 ``client.stop_session()``。

要真正跑重建并得到可检查的地图，可以用走查视频：读取视频，每秒抽取几帧
（走动时画面变化缓慢，~5 fps 足够），然后推入：

.. code-block:: python

   import cv2
   from rpent.robots.components.lingbotmap_client import LingBotMapClient
   from rpent.utils.rpc.http_rpc import HttpRpcClient

   video = cv2.VideoCapture("workspace_walkthrough.mp4")
   fps = video.get(cv2.CAP_PROP_FPS) or 30.0
   sample_every = max(1, round(fps / 5.0))  # ~5 fps
   frames = []
   while True:
       ok, frame = video.read()
       if not ok:
           break
       frames.append(frame)
   video.release()

   client = LingBotMapClient(HttpRpcClient("http://127.0.0.1:8116"))
   client.start_session(map_name="from_video")
   for index, frame in enumerate(frames[::sample_every]):
       rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # cv2 解码为 BGR
       client.push_frame(rgb, pose=None, timestamp_s=index / 5.0)
   result = client.build()
   print(result.frames, "frames ->", result.map_path)

录制时相机大致朝前。一个会话默认接受 300 帧；在更大的设备上可以调高服务端
的 ``--max-frames``，录制更长的视频前请先读
:ref:`workspace-mapping-limits`。该示例不推送底盘位姿，因此地图保持在模型
坐标系中，尺度任意。可对生成的目录运行下文的离线标注。

.. _workspace-mapping-grounding:

标注物体
--------

标注不必在 ``build_map`` 内完成。``rpent.tools.ground_map`` 读取已构建的地
图，均匀选取关键帧（或用 ``--keyframes`` 指定索引），并把标注写回地图：

.. code-block:: bash

   python -m rpent.tools.ground_map \
       --map-path /path/to/maps/workspace \
       --labels chair desk monitor \
       --perception-endpoint http://127.0.0.1:8115 \
       --max-keyframes 12

在重建地图上，每个掩码先按服务端的缩放和中心裁剪映射到深度网格，再用该帧的
内参与相机位姿反投影，并与其他关键帧上同标签的观测关联。没有深度的地图只保
留二维标注，此时需要 ``--allow-2d`` 才能写入。

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

``evaluation.json`` 给出每个标签的计数，以及不同帧中最近同标签观测距离的中
位数。该距离衡量跨帧一致性；精确率和召回率需要独立的真值标注，因此不作报告。

模型几何和检测器带来两点限制。完全落在模型中心裁剪之外的掩码没有深度，只保
留二维检测——归档帧保留了完整画面，但竖屏画面在裁剪时会丢掉正方形视野上下的
行。Grounding DINO 也可能对同一个物体返回多个重叠掩码；关联只在三维质心落入
半径阈值时才合并观测，因此关键帧较稀疏时，同一个物体可能留下多条记录。标签
匹配不区分大小写，而 slug 必须唯一，因此不能同时请求 ``Chair`` 和 ``chair``。

查看已标注的地图
----------------

``rpent.tools.map_view`` 把已标注的地图渲染为一个报告目录，无需加载重建数组
即可查看：

.. code-block:: bash

   python -m rpent.tools.map_view --map-path /path/to/maps/workspace

报告包含以下文件：``report/semantic_topdown.png`` 是按估计重力方向定向的俯视
图，没有重力估计时改用相机轨迹；``report/semantic_3d.png`` 是带相机轨迹的等
距视图；``report/detections.html`` 给出逐标签计数、物体与检测表格，以及全部
逐帧叠加图；``report/objects.csv`` 每个实例一行。俯视图以相机轨迹为 ``x``/``y``
中心，并把估计的地面置于 ``z = 0``。坐标轴与表格以地图单位标注；地图记录了
米制锚点时，尺度说明会给出一个单位对应多少米。``--max-points`` 限制每幅图
绘制的点数，``--output-dir`` 可把报告写到地图目录之外。

定位一帧
--------

定位先按建图服务端的相同方式预处理查询帧，再用 DISK + LightGlue 与归档关键
帧匹配，用该关键帧的深度、内参与相机位姿把匹配像素反投影到三维，最后通过
PnP RANSAC 解出查询帧在 ``map.json`` 所记录坐标系中的相机到世界位姿。它补上
了 LingBot-Map 留下的缺口：重建本身不会对后续运行做重定位。

把归档帧当作留出查询做回放，并把指标写入 ``localization_eval.json``：

.. code-block:: bash

   python -m rpent.tools.localize \
       --map-path /path/to/maps/workspace \
       --evaluate --query-stride 10

回放会隐藏查询帧本身，以及距它不足 0.1 个地图单位、朝向相差不足 10 度的归
档相机，避免查询与自身或近乎重复的视角匹配。``--exclusion N`` 改为（或额外）
隐藏查询前后 N 个帧索引内的归档帧；相机在运动时，这也会一并移除有用的重叠
视角，因此它适合作为压力测试，不适合作为主要成功率。

定位单张归档帧或实时帧：

.. code-block:: bash

   python -m rpent.tools.localize \
       --map-path /path/to/maps/workspace \
       --image /path/to/live_frame.png

   # 也可以按索引定位归档帧。
   python -m rpent.tools.localize \
       --map-path /path/to/maps/workspace --frame-index 240

冷启动时可靠的是穷举模式，它会检查全部模型关键帧。跟踪过程中可把上一次的帧
号传给 ``--prior-frame``，只匹配最近关键帧附近的一个窗口；Python API 还接受
``prior_pose``，按位姿距离对关键帧排序。没有先验时，32×32 的全局颜色描述子
会先筛出 ``top_k`` 个关键帧（``--mode retrieval``）。检索只用于缩小范围：每
个候选仍须通过 PnP，下面的回放也显示它会漏掉穷举模式能找到的关键帧。

只有同时满足以下条件，结果才为 ``ok``：PnP 内点至少 25 个、内点比例至少
0.10、内点重投影 RMS 不超过 10 像素。若第二个几何上不同的位姿拥有最佳位姿
至少一半的内点，且两者相差超过 1.0 个地图单位或 20 度，结果为 ``ambiguous``，
表示仅凭这一帧无法区分两个位置。其余情况为 ``lost``。只有 ``ok`` 计为成功；
返回的位姿是地图坐标系中的相机到世界变换，需要结合相机安装变换才能换到基座
坐标系。

通过 RPC 定位
~~~~~~~~~~~~~

在地图与 GPU 所在的机器上启动服务，每次调用发送一帧 RGB 图像：

.. code-block:: bash

   python -m rpent.robots.components.localize_server \
       --map-path /path/to/maps/workspace \
       --transport http --host 0.0.0.0 --port 8117

.. code-block:: python

   import imageio
   from rpent.robots.components.localize_client import LocalizeClient
   from rpent.utils.rpc.http_rpc import HttpRpcClient

   client = LocalizeClient(HttpRpcClient("http://127.0.0.1:8117"))
   frame = imageio.imread("live_frame.png")  # RGB [H,W,3]
   result = client.localize(frame, exhaustive=True)
   print(result.status, result.keyframe, result.inliers)
   print(result.pose)  # 地图坐标系中的相机到世界 4x4 矩阵

实测精度
~~~~~~~~

在一段 366 帧的工作空间录像上（183 个模型关键帧，每十帧取一个查询，穷举搜
索，共 19 个查询），18 个被成功定位且全部准确，1 个丢失。平移误差中位数
0.047 个地图单位（p75 0.084，最大 0.099），旋转误差中位数 0.93 度（p75
1.20，最大 3.03）。在 RTX 4090 上，延迟中位数 9.3 秒、最大 20.2 秒，其中包
含首次填充特征缓存的查询。检索模式取八个候选时，15 个查询准确定位，2 个标
记为 ``ambiguous``，2 个丢失，延迟中位数 0.58 秒。在另一段 143 帧的法院走
廊录像上（29 个查询，穷举搜索），28 个准确定位，1 个 ``ambiguous``，没有丢
失；平移误差中位数 0.021 个单位，旋转误差中位数 0.80 度（最大 12.9），延迟
中位数 4.5 秒。两张地图的尺度都是任意的，因此这些距离是模型单位，不是米。

上述结果来自对建图所用录像本身的回放，并隐藏了近乎相同的视角。跨会话定位
（对同一空间重新录制一段视频）尚未验证。重复或对称的场景可能让两个位置获得
相当的支持度，此时服务会报告 ``ambiguous``，而不是任选一个。低纹理帧可能得
不到足够的内点；工作空间回放中丢失的那个查询只有 22 个。穷举开销随关键帧数
量增长，因此机器人已有大致位姿后，应改用先验窗口或检索候选。

检查障碍物通行
--------------

通行检查在前进前回答一个问题：地图上正前方的地面是否畅通？它先定位当前相
机帧，然后测量一条走廊——走廊从相机中心开始，沿相机光轴投影到地面的方向延
伸，再逆时针转过 ``heading_deg``。走廊长 ``length``、宽 ``2 * half_width``，
只有高于地面 ``min_height`` 且低于 ``max_height`` 的点才算障碍物。它不规划路
径，也不移动机器人：它返回测量结果，由 agent 决定下一步动作。

检查需要地图提供三样东西：世界点云、估计的重力方向和拟合的地面平面。缺少其
中任何一样时，结果报告 ``unusable``，而不是猜测。

.. list-table::
   :header-rows: 1

   * - 工具参数
     - 默认值
     - 作用
   * - ``heading_deg``
     - 0
     - 相对相机视线方向逆时针转动走廊。相机不朝行驶方向时可以这样修正。
   * - ``length``
     - 2.0
     - 检查多远。
   * - ``half_width``
     - 0.75
     - 走廊半宽；应设为底盘半宽加安全余量。
   * - ``unit``
     - ``camera_heights``
     - 上述三个距离的单位：``camera_heights``、``map_units`` 或 ``m``。

``min_height`` 默认 0.05、``max_height`` 默认 4.0，二者都不由工具暴露；RPC 与
命令行接受它们。它们与 ``length`` 使用同一单位。

使用 agent 工具时，``check_clearance`` 只读，且仅在接入定位服务时注册。第一
次调用做穷举定位；之后的调用把上一次的位姿与关键帧作为先验，缩小搜索范围，
并让连续的检查保持在同一个估计上。

.. code-block:: text

   check_clearance(heading_deg=0, length=2.0, half_width=0.75,
                   unit="camera_heights")

结果只有以下状态之一：

.. list-table::
   :header-rows: 1

   * - 状态
     - 含义
   * - ``clear``
     - 沿路每个分箱都有已建图几何，且进入走廊的点少于三个。只有该状态允许
       前进。
   * - ``blocked``
     - 至少三个已建图的点落在走廊内。``nearest`` 给出最近点的距离、横向偏
       移、离地高度；地图做过标注时还会给出物体标签。
   * - ``unmapped``
     - 走廊有一部分没有已建图几何。``observed_distance`` 是已建图部分的
       终点。这不等于畅通：只有在另有独立证据时才可盲目前进。
   * - ``unlocalized``
     - 当前帧未定位，没有位姿可检查。停下来，把相机转向已建图区域，然后
       重试。
   * - ``unusable``
     - 地图缺少重力、地面平面、``m`` 所需的米制锚点，或视线方向不可用。
       ``reason`` 说明原因。

结果还会报告 ``unit_scale``，即一个地图单位等于多少个所请求的单位；已知时还
给出 ``camera_height`` 与 ``meters_per_unit``，以及 ``coverage``，即沿路分箱
中已有建图几何的比例。结果同时包含使用的位姿，以及定位所依据的关键帧。

使用命令行时，一张归档相机位姿即可完成离线检查：

.. code-block:: bash

   python -m rpent.tools.clearance \
       --map-path /path/to/maps/workspace \
       --frame-index 40 \
       --length 2.0 --half-width 0.5 --unit camera_heights

它打印与 RPC 相同的 JSON；``--output`` 写入文件，``--camera-height-m`` 为没
有米制尺度的地图锚定米。

通过 RPC 调用时，``LocalizeClient.check_path`` 接受一帧 RGB 图像，或一个地
图坐标系中的 4x4 相机到世界位姿，并转发走廊参数：

.. code-block:: python

   from rpent.robots.components.localize_client import LocalizeClient
   from rpent.utils.rpc.http_rpc import HttpRpcClient

   client = LocalizeClient(HttpRpcClient("http://127.0.0.1:8117"))
   result = client.check_path(frame, exhaustive=True, length=2.0)
   print(result.status, result.nearest)

   # 也可以用其他来源得到的位姿检查。
   result = client.check_path(pose=camera_to_world, half_width=0.5)

只能传入一种来源。``prior_index``、``prior_pose`` 与 ``exhaustive`` 只对图
像有效，因为它们影响定位；位姿不需要这些参数。

在一段 120 帧的单向工作空间录像上，通过服务检查归档帧 15 时，它定位到关键
帧 15（1467 个内点，内点比例 0.90，RMS 4.4 像素），并报告 ``clear``，覆盖
完整。首次定位在模型加载期间耗时约 10 秒；紧接着的重复调用约 6 秒。

.. _workspace-mapping-frames:

坐标系与单位
------------

**坐标系。** 相机位姿、点云与定位结果都使用 ``map.json`` 记录的坐标系，即重
建自身的 ``model`` 坐标系。重力方向与地面平面也在该坐标系中。通行检查在内部
把点旋转到与重力对齐的坐标系，但返回它使用的模型坐标系位姿，以便下一次检查
作为先验传回。

每一帧都携带底盘位姿时，建图服务端还会把重建出的相机轨迹与这些位置拟合一个
相似变换，并写入 ``alignment``：``scale`` 为每个模型单位对应的机器人地图单
位，另有 ``rotation``、``translation``、``rms_error_m``，以及使用了多少帧、
多少帧没有位姿。机器人坐标系下的位姿副本与点云单独保存；模型坐标系数组仍是
定位与通行检查的权威来源。拟合后的坐标系把相机轨迹放在 ``z = 0`` 上、``+z``
朝上，因此地面位于 ``z = 0`` 之下。

**尺度。** 只有底盘位姿锚定过的地图才是米制的（``scale.status`` 为
``metric``）。否则距离是任意的模型单位，米制距离没有意义。因此通行检查默认
使用 ``camera_heights``，它不需要外部尺度：

.. list-table::
   :header-rows: 1

   * - 单位
     - 含义
   * - ``camera_heights``
     - 地面到相机距离的倍数。
   * - ``map_units``
     - 重建自身的单位；在同一张地图内可比。
   * - ``m``
     - 米。需要地图记录米制尺度，或在定位服务端传
       ``--camera-height-m``，用真实相机高度锚定尺度。

走廊也以传入的单位报告，因此 2.0 个相机高度的请求大约相当于相机位于胸口高
度时一个底盘的长度。相机高度在不同地图之间可比，模型单位则不可比。

.. _workspace-mapping-limits:

限制与排查
----------

**地图在帧数上限处停止。** 会话在达到 ``--max-frames`` 后拒绝新帧，默认 300，
约为 24 GB 显存能重建的规模。一次 400 帧的构建在 24 GB 设备上耗尽显存。请
放慢走查速度、在更大的设备上调高 ``--max-frames``，或把工作空间分成两张地
图。显存耗尽失败后，下一次构建前先重启建图服务。

**没有重力或地面。** 通行检查需要两者。过短或接近平面的录制可能没有足够的
相机旋转来估计重力；此时地图对通行检查报告 ``unusable``，但定位仍然可用。
帧归档地图（未启用 GCT 权重）完全没有几何，只能用于定位，不能用于通行检查。

**地图未覆盖不等于畅通。** 走廊分箱就是证据。走廊远端没有地图看到的点时，
检查报告 ``unmapped``，并给出覆盖结束的位置。扩大地图，或缩短 ``length``。

**相机附近的噪声可能误报为阻塞。** 通行检查统计体素内的点，而合并点云
带着每一帧的重建噪声。在一段 300 帧的环绕录像和一段 120 帧的单向录像上，大
多数默认检查都报告 ``blocked``：少数点落在相机地面投影周围几毫米内，另一些
则高于 0.05 个相机高度的障碍物下限。默认值换算成绝对长度也很小：``length``
与 ``half_width`` 都是估计相机高度的倍数，相机较低时走廊也随之又短又窄。请
先查看俯视图报告，并把 ``blocked`` 当作保守答案，而不是对最近物体的测量。在
命令行或 RPC 调用中调高 ``min_height``，可以排除地面粗糙度后再判断路线是否
被堵；agent 工具使用共享的默认值。

**定位丢失。** 低纹理或运动模糊的帧、录制区域之外的视角，或与地图帧不同的
分辨率/朝向都会导致 ``lost``。把相机转向已建图区域，让查询帧保持地图的分辨
率，并在第一次检查时使用穷举搜索。工作空间回放中丢的一个查询只有 22 个内
点，因此置信裕度很重要。

**首次调用很慢。** DISK、LightGlue 与感知模型都是懒加载。第一次定位可能耗
时数十秒，之后会快得多。通行检查在每个服务端进程中只加载并去重一次点云。

**距离看起来不对。** 检查结果中的 ``unit``、``camera_height`` 与
``meters_per_unit``。在尺度任意的地图上请求 ``m`` 会被拒绝，而不是猜测；已
知真实相机高度时可传 ``--camera-height-m``。估计的地面是一个平面拟合，地面
倾斜或观测稀疏会使所有高度带有同一偏差。
