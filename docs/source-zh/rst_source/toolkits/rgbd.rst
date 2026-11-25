.. _generic-toolkit:

具身 RGB-D 工具包
=================

使用 RPent 的只读 RGB-D 感知工具放大已记录画面的局部细节、查询像素对应的三维表面位置、分割目标物体或测量物体指定部位。

以杯子为例，选中杯壁上的一个像素后，``back_project`` 可返回对应表面点的三维坐标。规划器也可以把这个像素或“白色杯子”这样的描述传给 ``segment``，获取候选掩膜，并在叠加图上核对掩膜是否覆盖目标杯子的可见区域。观测充分时，``geometry`` 可拟合杯口，给出杯口中心和半径，也可根据杯口和杯底的位置估算杯高。结果中的 ``world_frame_id`` 标明杯壁表面点和杯口中心所在的世界坐标系。点的坐标分量、杯口半径和杯高都以米为单位。

选择工具
--------

四个工具都接受相机名 ``camera`` 和观测步号 ``step``。传入 ``step=-1`` 表示最近一次记录。在相机的整幅预览图上选点时，用该图的高和宽填写 ``image_shape``，像素按行、列顺序传入。

.. list-table::
   :header-rows: 1
   :widths: 17 39 44

   * - 工具
     - 规划器提供的输入
     - 如何使用结果
   * - ``crop_zoom``
     - 传入整幅图的 ``image_shape`` 和矩形框 ``box_rc`` [上，左，下，右]。下、右两项分别填写框外紧邻的行号和列号。用 ``grid_step`` 设置网格间隔（原图像素），或用 ``marks_rc`` 按输入顺序给框内的点编号。每个标记须对应不同的原图像素。
     - 工具返回放大后的 PNG，``source_box_rc`` 给出裁剪框在原图中的位置。如果输出边长超过 1024 像素，缩小选框或调低 ``zoom``。
   * - ``back_project``
     - 在 ``selection`` 中传入 ``image_shape``，并给出点列 ``points_rc`` 或矩形框 ``box_rc``。只想保留某个高度范围内的点时，可以设置 ``height_m``，高度按世界坐标系计算。
     - 选点时按顺序返回三维坐标，深度无效或被筛掉的点记为 ``null``。框选时返回一个实测表面点的坐标和框内可见表面的三维范围。
   * - ``segment``
     - 用 ``prompt`` 描述目标，或用 ``point`` 指定目标上的一个像素点。两者都提供时，工具先按文字描述生成候选掩膜，确认所选点落在掩膜内后才返回该掩膜。
     - 结果包含候选掩膜和叠加图的保存路径。当掩膜内有足够多的有效深度点时，``world_xyz`` 的三个分量分别取这些点的 X、Y、Z 坐标的中位数。
   * - ``geometry``
     - 用 ``query`` 描述目标，可以查询其可见表面上一个点的三维坐标，或目标可见部分的范围。要测量特定部位的位置或尺寸，可在 ``required_geometry`` 中指定测量项。
     - 工具在内部选点并核验是否选中了目标物体。必要时，还会查看同一步已记录的其他相机画面。只有通过质量检查的拟合量才返回数值。

在仿真器中使用
--------------

LIBERO
~~~~~~

LIBERO 的适配器按相机名（``agentview`` 或 ``wrist``）和步号读取 RGB 图像、世界坐标图和标定数据。LIBERO 默认使用自身的 ``back_project`` 和 ``segment``。加入 ``--rgbd-tools`` 后，可调用 ``crop_zoom``，并改用这两个工具的通用版本。

``geometry`` 是一项实验功能，需要 Molmo 服务支持原生选点和目标验证（``molmo.verify``）。服务启动步骤见\ `配置 Molmo <../guides/flash.html#molmo>`_。

将 ``MOLMO_ENDPOINT`` 设为服务地址，在 LIBERO 运行命令中加入 ``--geometry --molmo-endpoint "$MOLMO_ENDPOINT"``。该选项同时启用 ``--rgbd-tools``。

例如，规划器可以用一次 ``geometry`` 调用测量同一个杯子的杯口半径和完整高度。

.. code-block:: json

   {"camera": "agentview", "step": 0, "query": "the white cup on the plate",
    "required_geometry": ["rim_radius", "height"]}

两项拟合都通过时返回 ``usable``。如果杯口清晰，但所有已记录视角都看不到杯底，则返回 ``partial``，给出以米为单位的杯口半径，并在 ``missing_geometry`` 中列出 ``height``。如果两项都能估算，但其中一项触发质量警告，则返回 ``warning``，只给出通过检查的那一项的值。两项都缺乏可靠证据时返回 ``insufficient_evidence``，由规划器选择下一步。

查看局部细节
------------

可以用下面的提示词引导规划器查看按钮或接触点。

   在已记录的整幅图上框住细节，调用 ``crop_zoom`` 后查看放大的 PNG。选中编号点时，直接取对应的 ``source_point_rc``。若 ``label_boxes_rc`` 中有一项为 ``null``，说明放大图上放不下对应的编号。可调高 ``zoom`` 或减少标记。如果选的是放大图上的其他像素，先将行号和列号分别除以 ``zoom``，再向下取整。行号加上 ``source_box_rc`` 的第一项，列号加上第二项，即可得到原图坐标。

   需要三维坐标时，把原图坐标填入 ``selection.points_rc``，把返回的 ``image_shape`` 填入 ``selection.image_shape``。用该选区和返回的 ``camera``、``step`` 调用 ``back_project``。

.. _generic-toolkit-registration:

为机器人接入工具
----------------

在真实机器人或新仿真器上启用这些工具时，需要实现观测适配器，按步号读取 RGB-D 帧与世界坐标系下的标定数据（若相机随机械臂移动，使用采集时刻的相机位姿），并在机器人的 ``toolkit`` 中注册工具。

.. dropdown:: 观测适配器接口与工具注册示例

   接入仿真器或真机时，由该平台的观测适配器实现 ``get_observation(camera=..., step=...)``，通过一个 ``toolkits.observation.GeometryObservation`` 对象返回指定相机在该步的 RGB 图像和对应的三维数据。适配器可以直接提供与图像对齐的世界坐标图，给出每个像素对应的三维坐标（单位为米），也可以提供与图像对齐的光轴深度图（单位为米），并附上相机内参和相机到世界坐标系的变换。

   用 ``back_project`` 按高度筛选采样点时，适配器还需提供世界坐标系的向上方向 ``up_direction``。用 ``geometry`` 测量 ``query`` 中描述的物体时，观测适配器需要同时提供 ``up_direction`` 和相机位姿 ``camera_to_world``。若有多台标定相机，可实现 ``additional_observations(primary)``，提供同一步、同一世界坐标系的其他视角。

   在平台的 ``toolkit`` 中注册工具前，先准备观测适配器 ``observations``、当前会话的状态对象 ``state``\ （``EnvState`` 实例）和已初始化的分割客户端 ``segmenter``。

   示例会注册 ``back_project``、``crop_zoom`` 和 ``segment``。需要 ``geometry`` 时传入 ``grounder``，否则设为 ``None``。

   .. code-block:: python

      from functools import partial

      from toolkits.back_project import back_project
      from toolkits.crop_zoom import crop_zoom
      from toolkits.segment import SegmentTool

      sample = back_project.with_handler(partial(back_project, observations=observations))
      toolkit.add_tool(sample)
      crop = crop_zoom.with_handler(partial(crop_zoom, observations=observations))
      toolkit.add_tool(crop)
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
