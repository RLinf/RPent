.. _generic-toolkit:

通用 RGB-D 工具包
=================

机器人注册 ``toolkits`` 中的只读工具后，规划器就可以从已记录的 RGB-D 观测中读取像素的三维坐标、分割物体或测量指定部位。各机器人提供自己的相机和标定数据。位置与长度以米为单位，坐标所属的世界坐标系由 ``world_frame_id`` 标明。

选择工具
--------

三个工具都接受相机名 ``camera`` 和观测步号 ``step``。传入 ``step=-1`` 表示最近一次记录。在预览图上选点时，用预览图的高和宽填写 ``image_shape``，像素按 [行, 列] 传入。

.. list-table::
   :header-rows: 1
   :widths: 17 39 44

   * - 工具
     - 规划器提供的输入
     - 如何使用结果
   * - ``back_project``
     - 在 ``selection`` 中传入 ``image_shape``，并给出点列 ``points_rc`` 或半开矩形框 ``box_rc``，边界按上、左、下、右排列，下边界和右边界不计入选区。可用 ``height_m`` 沿世界坐标系的向上方向筛选高度。
     - 点列按输入顺序返回表面坐标，深度无效或被过滤的点保留为 ``null``。矩形框返回一个实际采样的表面点和可见范围。物体身份由调用方确认。
   * - ``segment``
     - 提供文字描述 ``prompt``、一个正提示点 ``point``，也可以同时提供两者。两者同时使用时，文字分割得到的掩膜须覆盖该点。
     - 结果包含候选掩膜和叠加图的保存位置。有效深度充足时，``world_xyz`` 给出掩膜内各坐标分量的中位数。用叠加图确认选中的物体。
   * - ``geometry``
     - 用 ``query`` 描述目标。可以请求可见表面点、可见区域，或在 ``required_geometry`` 中指定想测量的部位和尺寸。
     - 工具在内部选点并核验目标身份，必要时检查同一步已有的其他视角。只有通过质量检查的拟合量才返回数值。

在 LIBERO 中使用
----------------

LIBERO 已接入这套工具。其 API 规划器默认开放 ``back_project`` 和使用 SAM3 的 ``segment``。如需启用 ``geometry``，将 ``MOLMO_ENDPOINT`` 设为已启动的本地 Molmo 服务地址，并在 :doc:`LIBERO 运行命令 <../simulators/libero>` 中加入 ``--geometry --molmo-endpoint "$MOLMO_ENDPOINT"``。

例如，规划器可以用一次 ``geometry`` 调用测量同一个杯子的杯口半径和完整高度。

.. code-block:: json

   {"camera": "agentview", "step": 0, "query": "the white cup on the plate",
    "required_geometry": ["rim_radius", "height"]}

两项拟合都通过时返回 ``usable``。如果杯口清晰，但所有已记录视角都看不到杯底，则返回 ``partial``，给出以米为单位的杯口半径，并在 ``missing_geometry`` 中列出 ``height``。如果两项都能估算，但其中一项触发质量警告，则返回 ``warning``，只给出通过检查的那一项的值。两项都缺乏可靠证据时返回 ``insufficient_evidence``，由规划器选择下一步。

为机器人注册工具
----------------

在机器人观测适配器中实现 ``get_observation(camera=..., step=...)``，返回 ``toolkits.observation`` 中的 ``GeometryObservation``。观测包含 RGB 图像。可直接提供与图像对齐的米制世界坐标图，也可提供米制光轴深度、相机内参和相机到世界坐标系的变换。测量具名目标还需要相机位姿和世界坐标系的向上方向。若有多台标定相机，可实现 ``additional_observations(primary)``，提供同一步、同一世界坐标系的其他视角。

准备好适配器 ``observations``、当前会话的 ``EnvState`` 实例 ``state`` 和已初始化的分割客户端 ``segmenter``，即可在机器人的 ``toolkit`` 上注册工具。要启用具名目标测量，再传入能选点和核验目标的 ``grounder``。仅注册前两个工具时将其设为 ``None``。

.. code-block:: python

   from functools import partial

   from toolkits.back_project import back_project
   from toolkits.segment import SegmentTool

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

如果新回合会从相同的步号重新计数，请保留几何工具实例，并在开始新回合时调用 ``geometry.reset_cache()``。
