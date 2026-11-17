定位与通行检查
==============

定位工具集找出相机帧在地图中的位置，并报告前方地面是否畅通。它提供 RPC 服务
与与传输无关的客户端，并附带用于回放评估和从归档位姿做通行检查的离线命令。
地图来自 :doc:`mapping`；物体标签是可选的，来自 :doc:`grounding`。

准备工作
--------

定位与通行检查需要 CUDA GPU、``localize`` extra，以及一张带深度、重力方向和
地面平面的地图：

.. code-block:: bash

   uv pip install -e '.[localize]'

DISK 与 LightGlue 在首次使用时下载权重。当地图没有米制尺度、而结果需要以米
表示时，传入 ``--camera-height-m``；详见 :ref:`workspace-mapping-frames`。

启动定位服务
------------

``toolkits.localization.server`` 加载一张已构建的地图，估计查询帧的位姿，并
从该位姿检查一条走廊。它的 ``--map-path`` 是构建写出的地图目录，例如
``/path/to/maps/workspace``。

.. code-block:: bash

   python -m toolkits.localization.server \
       --map-path /path/to/maps/workspace \
       --transport http --host 127.0.0.1 --port 8117

服务客户端
----------

``toolkits.localization.client`` 既可用于本地服务，也可用于远程服务：

.. list-table::
   :header-rows: 1

   * - 调用
     - 输入
     - 返回
   * - ``LocalizeClient.localize(frame, ...)``
     - 与地图分辨率一致的 RGB 帧。
     - ``status``、相机到世界位姿、所用关键帧与内点数。
   * - ``LocalizeClient.check_path(frame=... | pose=..., ...)``
     - 只传一个 RGB 帧或一个 4x4 相机到世界位姿，外加走廊参数。
     - 通行状态及其测量结果。

地图路径始终是构建写出的目录，而不是其中的某个文件。

定位一帧
--------

定位先按建图服务端的相同方式预处理查询帧，再用 DISK + LightGlue 与归档关键
帧匹配，用该关键帧的深度、内参与相机位姿把匹配像素反投影到三维，最后通过
PnP RANSAC 解出查询帧在 ``map.json`` 所记录坐标系中的相机到世界位姿。它补上
了 LingBot-Map 留下的缺口：重建本身不会对后续运行做重定位。

把归档帧当作留出查询做回放，并把指标写入 ``localization_eval.json``：

.. code-block:: bash

   python -m toolkits.localization.localize \
       --map-path /path/to/maps/workspace \
       --evaluate --query-stride 10

回放会隐藏查询帧本身，以及距它不足 0.1 个地图单位、朝向相差不足 10 度的归
档相机，避免查询与自身或近乎重复的视角匹配。``--exclusion N`` 改为（或额外）
隐藏查询前后 N 个帧索引内的归档帧；相机在运动时，这也会一并移除有用的重叠
视角，因此它适合作为压力测试，不适合作为主要成功率。

定位单张归档帧或实时帧：

.. code-block:: bash

   python -m toolkits.localization.localize \
       --map-path /path/to/maps/workspace \
       --image /path/to/live_frame.png

   # 也可以按索引定位归档帧。
   python -m toolkits.localization.localize \
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

   python -m toolkits.localization.server \
       --map-path /path/to/maps/workspace \
       --transport http --host 0.0.0.0 --port 8117

.. code-block:: python

   import imageio
   from rpent.utils.rpc.http_rpc import HttpRpcClient
   from toolkits.localization.client import LocalizeClient

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
径，也不移动机器人：它返回测量结果，由调用方决定下一步动作。

检查需要地图提供三样东西：世界点云、估计的重力方向和拟合的地面平面。缺少其
中任何一样时，结果报告 ``unusable``，而不是猜测。

.. list-table::
   :header-rows: 1

   * - 参数
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

``min_height`` 默认 0.05、``max_height`` 默认 4.0，二者与 ``length`` 使用同
一单位。

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

**使用命令行时**，一张归档相机位姿即可完成离线检查：

.. code-block:: bash

   python -m toolkits.localization.clearance \
       --map-path /path/to/maps/workspace \
       --frame-index 40 \
       --length 2.0 --half-width 0.5 --unit camera_heights

它打印与 RPC 相同的 JSON；``--output`` 写入文件，``--camera-height-m`` 为没
有米制尺度的地图锚定米。

**通过 RPC 调用时**，``LocalizeClient.check_path`` 接受一帧 RGB 图像，或一个
地图坐标系中的 4x4 相机到世界位姿，并转发走廊参数：

.. code-block:: python

   from rpent.utils.rpc.http_rpc import HttpRpcClient
   from toolkits.localization.client import LocalizeClient

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

单位
----

通行检查默认使用 ``camera_heights``，它不需要外部尺度：

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
度时一个底盘的长度。相机高度在不同地图之间可比，模型单位则不可比。地图一侧
的尺度契约见 :ref:`workspace-mapping-frames`。

限制与排查
----------

**没有重力或地面。** 通行检查需要两者。过短或接近平面的录制可能没有足够的
相机旋转来估计重力；此时地图对通行检查报告 ``unusable``，但定位仍然可用。

**地图未覆盖不等于畅通。** 走廊分箱就是证据。走廊远端没有地图看到的点时，
检查报告 ``unmapped``，并给出覆盖结束的位置。扩大地图，或缩短 ``length``。

**相机附近的噪声可能误报为阻塞。** 通行检查统计体素内的点，而合并点云带着
每一帧的重建噪声。在一段 300 帧的环绕录像和一段 120 帧的单向录像上，大多数
默认检查都报告 ``blocked``：少数点落在相机地面投影周围几毫米内，另一些则高
于 0.05 个相机高度的障碍物下限。默认值换算成绝对长度也很小：``length`` 与
``half_width`` 都是估计相机高度的倍数，相机较低时走廊也随之又短又窄。请先
查看俯视图报告，并把 ``blocked`` 当作保守答案，而不是对最近物体的测量。在
命令行或 RPC 调用中调高 ``min_height``，可以排除地面粗糙度后再判断路线是否
被堵。

**定位丢失。** 低纹理或运动模糊的帧、录制区域之外的视角，或与地图帧不同的
分辨率/朝向都会导致 ``lost``。把相机转向已建图区域，让查询帧保持地图的分辨
率，并在第一次检查时使用穷举搜索。工作空间回放中丢的一个查询只有 22 个内
点，因此置信裕度很重要。

**首次调用很慢。** DISK 与 LightGlue 都是懒加载。第一次定位可能耗时数十秒，
之后会快得多。通行检查在每个服务端进程中只加载并去重一次点云。

**距离看起来不对。** 检查结果中的 ``unit``、``camera_height`` 与
``meters_per_unit``。在尺度任意的地图上请求 ``m`` 会被拒绝，而不是猜测；已
知真实相机高度时可传 ``--camera-height-m``。估计的地面是一个平面拟合，地面
倾斜或观测稀疏会使所有高度带有同一偏差。
