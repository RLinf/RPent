.. _generic-toolkit:

具身 RGB-D 工具包
=================

RPent 提供的只读 RGB-D 感知工具，可以从已记录的相机观测中查询像素对应的三维表面位置、分割目标物体或获取物体指定部位的几何信息。

以杯子为例，选中杯壁上的一个像素后，``back_project`` 可返回对应表面点的三维坐标。规划器也可以把这个像素或“白色杯子”这样的描述传给 ``segment``，获取候选掩膜。需要杯口中心时，可用 ``geometry`` 拟合杯口内沿，证据充分时返回中心坐标。杯壁表面点和杯口中心都以米为单位，所在的世界坐标系由 ``world_frame_id`` 标明。

选择工具
--------

.. list-table::
   :header-rows: 1
   :widths: 17 40 43

   * - 工具
     - 功能
     - 结果
   * - ``back_project``
     - 读取指定像素或矩形框内可见表面的三维位置。可以只保留世界坐标系中某个高度范围内的点。
     - 选点时按顺序返回每个点的三维坐标，深度无效或被筛掉的点为 ``null``。框选时返回一个实测表面点的坐标，以及框内可见表面的三维范围。
   * - ``segment``
     - 根据文字描述，或目标上的一个像素点，分割出目标的候选掩膜。
     - 找到目标时，返回掩膜和叠加图的文件名，并附上叠加图供查看。掩膜内有足够多的有效深度点时，还会给出这些点 X、Y、Z 坐标的中位数 ``world_xyz``。
   * - ``geometry``
     - 目前是一个实验性工具。先定位整个物体，再拟合它的指定部位，返回杯口中心、盘面中心等几何量。
     - 返回通过质量检查的几何量，未通过的列在 ``missing_geometry`` 中。

geometry 支持的测量
~~~~~~~~~~~~~~~~~~~

``geometry`` 用 ``required_geometry`` 指定要测的几何量，用 ``feature`` 指定看物体的哪个部位。目前已实现的几何量如下：

.. list-table::
   :header-rows: 1
   :widths: 32 28 40

   * - 可测量的几何量
     - ``required_geometry``
     - ``feature``
   * - 圆形开口内沿的圆心
     - ``ring_center``
     - ``upper_inner_ring``
   * - 圆形外沿的圆心（也适用于实心圆盘）
     - ``ring_center``
     - ``upper_outer_ring``
   * - 圆面的圆心
     - ``disk_center``
     - ``upper_face``
   * - 矩形面的中心
     - ``rectangle_center``
     - ``upper_face``
   * - 直立圆形物体的底部中心
     - ``base_center``
     - ``supported_round_base``

geometry 测试效果
~~~~~~~~~~~~~~~~~

这轮 LIBERO 固定帧测试使用 RPent ``a38d777``、SAM3 和 Molmo2-8B。16 个任务各取种子 1 至 5，共 80 个场景。每个场景都在第 0 步的 ``agentview`` RGB-D 帧上测量两个目标，共 160 条请求。

评测从冻结的仿真场景中独立确定目标部位的参考中心，只用于评分。误差是工具返回的中心与参考中心在世界坐标系中的三维欧氏距离。酒瓶底面以瓶身中轴与支撑面的交点为准，微波炉顶面以机身主矩形顶板的中心为准。

.. list-table::
   :header-rows: 1

   * - 请求数
     - 返回坐标的请求数
     - 误差均值
     - 误差中位数
     - 最大误差
   * - 160
     - 160
     - 1.359 mm
     - 1.184 mm
     - 2.947 mm

所有结果的状态均为 ``usable``，误差均小于 3 mm。参见\ `请求清单和汇总结果 <https://github.com/nanji17/RPent-geometry-test/tree/2f982351e23cde92ddc99cc163de1e8f0500d74a#libero-fixed-frame-tests-r1>`_。

在仿真器中使用
--------------

LIBERO
~~~~~~

LIBERO 的适配器按相机名和步号读取 RGB 图像、世界坐标图和标定数据。通用工具 ``back_project`` 和 ``segment`` 支持 ``agentview`` 和 ``wrist`` 两个视角，实验工具 ``geometry`` 只支持主视角 ``agentview``。

LIBERO 默认仍使用自身的 ``back_project`` 和 ``segment``。加入 ``--rgbd-tools`` 后，改用通用版 ``back_project`` 和 ``segment`` 及其参数格式。加入 ``--geometry`` 后，会启用 ``geometry``，同时启用通用版 ``back_project`` 和 ``segment``。

三个工具都接受观测步号 ``step``，``step=-1`` 表示最近一次记录。在图像上选点时，``image_shape`` 填该图像的高和宽，像素按 ``[行, 列]`` 传入。例如在 1024×1024 的 ``agentview_high.png`` 上选点时，``image_shape`` 为 ``[1024, 1024]``。

用 ``back_project`` 读取某个像素的三维表面位置：

.. code-block:: json

   {"camera": "agentview", "step": 0,
    "selection": {"image_shape": [1024, 1024], "points_rc": [[200, 240]]}}

结果中的 ``samples_xyz`` 按输入顺序给出每个点的三维坐标。要读取一个区域，把 ``points_rc`` 换成矩形框 ``box_rc``，格式为 ``[上, 左, 下, 右]``，下边界和右边界不含在框内。只想保留某个高度范围内的点时，加入 ``height_m``，格式为 ``[最低, 最高]``。

用 ``segment`` 分割目标物体时，``prompt`` 和 ``point`` 二选一。用文字描述目标示例如下：

.. code-block:: json

   {"camera": "agentview", "step": 0, "prompt": "white cup"}

或指定目标上的一个像素点，此时须同时给出 ``image_shape``：

.. code-block:: json

   {"camera": "agentview", "step": 0, "point": [200, 240],
    "image_shape": [1024, 1024]}

启用 ``--geometry`` 时，必须用 ``--molmo-endpoint`` 指定 Molmo 服务地址。Molmo 的启动方法见\ `配置 Molmo <../guides/flash.html#molmo>`_。LIBERO 默认在本地启动 SAM3，也可用 ``--sam3-endpoint`` 连接已有服务。

使用 ``--planner api --geometry`` 时，选用支持图像输入的模型，不加 ``--no-images``，以便规划器核对叠加图中标出的物体。

同时指定 ``--geometry`` 和 ``--sam3-endpoint`` 后，LIBERO 会在启动时检查外部服务能否按文本提示返回候选掩膜。只能返回最高分掩膜的旧服务仍可用于 ``segment``。要用这类服务运行 ``geometry``，先在运行 SAM3 的机器上升级 RPent，再重启服务。

测杯口圆心时，``object_prompt`` 写整个物体的简短名称，如“白杯子”，供 SAM3 分割。画面里有两只白杯子，要找盘子上的那只，就在 ``query`` 中写“盘子上的白杯子”。规划器根据这段描述核对叠加图中标出的杯子。

.. code-block:: json

   {"camera": "agentview", "step": 0, "query": "the white cup on the plate",
    "object_prompt": "white cup", "required_geometry": ["ring_center"],
    "feature": "upper_inner_ring"}

请求的几何量全部通过检查时返回 ``usable``；全部通过但有选物或拟合警告时返回 ``warning``。部分通过时返回 ``partial``，全部未通过时返回 ``insufficient_evidence``。

.. _generic-toolkit-registration:

为机器人接入工具
----------------

在真实机器人或新仿真器上启用这些工具时，需要实现观测适配器，按步号读取 RGB-D 帧与世界坐标系下的标定数据（若相机随机械臂移动，使用采集时刻的相机位姿），并在机器人的 ``toolkit`` 中注册工具。

.. dropdown:: 观测适配器接口与工具注册示例

   接入仿真器或真机时，由该平台的观测适配器实现 ``get_observation(camera=..., step=...)``，通过一个 ``toolkits.observation.GeometryObservation`` 对象返回指定相机在该步的 RGB 图像和对应的三维数据。适配器可以直接提供与图像对齐的世界坐标图，给出每个像素对应的三维坐标（单位为米），也可以提供与图像对齐的光轴深度图（单位为米），并附上相机内参和相机到世界坐标系的变换。

   用 ``back_project`` 按高度筛选采样点时，适配器还需提供世界坐标系的向上方向 ``up_direction``。用 ``geometry`` 测量目标物体时，适配器需要同时提供 ``up_direction`` 和相机位姿 ``camera_to_world``。要限定主视角，可在下方示例创建几何工具时通过 ``primary_camera`` 指定相机名称。

   在平台的 ``toolkit`` 中注册工具前，先准备观测适配器 ``observations``、当前会话的状态对象 ``state``\ （``EnvState`` 实例）和已初始化的分割客户端 ``segmenter``。

   如需同时注册 ``geometry``，将示例中的 ``grounder = None`` 改为能在原图上选点的客户端。分割客户端须能用 ``segment_candidates(image, text_prompt=..., min_score=...)`` 按文字描述返回带分数的候选掩膜，也能用 ``segment(image, point=...)`` 按选点分割。

   .. code-block:: python

      from functools import partial

      from toolkits.back_project import back_project
      from toolkits.segment import SegmentTool

      grounder = None
      sample = back_project.with_handler(partial(back_project, observations=observations))
      toolkit.add_tool(sample)
      masks = SegmentTool(observations=observations, state=state, segmenter=segmenter)
      toolkit.add_tool(masks.segment)

      if grounder is not None:
          from toolkits.geometry.geometry_tool import GeometryTool

          geometry = GeometryTool(
              observations=observations, state=state, segmenter=segmenter,
              grounder=grounder, check_cancelled=toolkit.raise_if_cancelled,
          )
          toolkit.add_tool(geometry.geometry)

   ``geometry`` 会缓存当前观测步的测量结果。如果新 episode 会从相同的步号重新计数，且沿用同一个几何工具实例，请在新 episode 开始时调用 ``geometry.reset_cache()`` 清除旧缓存。
