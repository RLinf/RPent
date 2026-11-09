.. _geometry-toolkit:

几何工具包
==========

``GeometryTool`` 用于复杂的几何感知任务。主规划器通过 ``geometry`` 描述要测的物体和部位，工具从指定的一帧相机观测中计算位置、方向或尺寸，并说明哪些可见部位支持这个结果。

设计目的
--------

要把物体对准灰杯左侧那只黄白色杯子的杯口，首先得知道杯口中心在哪里。图中可以指出杯沿上的一点，但杯口中心可能落在空处。工具用杯沿上一组可见的三维点拟合一个圆，由此估计圆心。要测杯高，还要看到杯底，并检查两端是否共轴，即杯口和杯底的圆面是否都与两圆心的连线近似垂直。

视觉模型根据颜色、部位和相对位置在图像中选点。几何计算使用与图像逐像素对齐、以米为单位的三维坐标。主规划器决定测什么，也决定测量结果是否足以支持下一步动作。

.. _geometry-module-design:

模块设计
--------

测量黄白色杯子
~~~~~~~~~~~~~~

主规划器想测杯口中心、杯身中心、杯底中心、轴线、杯口半径和杯高。下面的请求指定了相机 ``inspection_rgb`` 在第 0 步记录的观测。相机名 ``camera`` 必填，由接入工具的程序定义。省略 ``step`` 时，默认值 ``-1`` 表示该相机最新的已记录观测。

.. code-block:: json

   {
     "query": "the yellow and white mug left of the gray cup",
     "camera": "inspection_rgb",
     "step": 0,
     "required_geometry": [
       "rim_center", "body_center", "bottom_center", "axis", "radius", "height"
     ]
   }

``required_geometry`` 一次可列 1 到 10 项。这个请求只需分别查看口沿、外侧杯身和杯底，同一部位的观测可用于计算多个量。表中的中心坐标都位于观测指定的参考坐标系中，长度单位为米。

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * - 请求量
     - 这一帧需要看到什么
     - 通过检查后返回什么
   * - ``rim_center``
     - 有足够长的一段圆形杯沿，边缘上有有效的三维点。
     - 拟合圆的圆心坐标 ``[x, y, z]``。
   * - ``radius``
     - 与杯口中心共用同一段可见杯沿。
     - 同一个圆的半径。也可用 ``rim_radius`` 请求。
   * - ``bottom_center``
     - 可见的圆形杯底或其边缘，能从中提取足够的边界点。
     - 根据杯底边缘拟合出的圆心坐标。
   * - ``body_center``
     - 杯口和杯底都能拟合出圆，两个圆心分开且两端共轴。
     - 两个圆心的中点，即沿杯身轴线定义的中心。
   * - ``height``
     - 与杯身中心共用通过检查的杯口和杯底。
     - 两个圆心之间的直线距离。
   * - ``axis``
     - 单独请求时看杯身外侧的曲面。同一请求若还找到了两端，就检查两端是否共轴。
     - 轴线上一点 ``point``、单位方向 ``direction`` 和方向约定 ``sign``。

计算 ``body_center`` 和 ``height`` 时，程序也会尝试拟合杯身。如果侧壁拟合成功，它的轴线必须与两端吻合，可见杯身也必须位于两端之间。对于 ``axis``，两端通过检查时优先返回从杯底指向杯口的方向（``bottom_to_rim``）。从侧壁单独确定的轴线可以沿两个相反方向表示，``canonical_undirected`` 表示程序按固定规则选了其中一个方向。

离线示例用一圈三维点模拟杯沿，并让定位器只认出这个部位。工具读取一帧观测，分别尝试定位口沿、杯身和杯底，只对找到的口沿做了一次分割。下面摘录这次调用返回的字段，数值经过四舍五入。

.. code-block:: json

   {
     "status": "partial",
     "world_frame_id": "workcell/base",
     "geometry": {
       "rim_center": {"value": [0.0, 0.8, 0.0]},
       "radius": {"value": 0.04004, "unit": "m", "evidence_status": "inferred"},
       "bottom_center": {"value": null},
       "body_center": {"value": null},
       "axis": {"value": null},
       "height": {"value": null, "reason": "both_consistent_end_boundaries_required"}
     },
     "observation_request": {"missing_parts": ["body", "bottom"]}
   }

先看整次请求的 ``status``。``usable`` 表示所需结果全部可用，``partial`` 表示只有部分拟合量可用，``need_observation`` 表示这帧观测不足以给出所需结果。再看各个 ``geometry[field]`` 的 ``definition`` 和 ``value``。上例给出了杯口圆心和约 4 厘米的杯口半径。杯高需要两端的圆心，因此仍为 ``null``。

拟合得到的值标为 ``inferred``，缺少依据的字段标为 ``unknown``。每个字段还记录了单位、形状假设、质量信息和证据引用，可用 ``world_frame_id`` 和 ``source_step`` 确认它属于哪个坐标系、哪一步观测。观测读取、模型调用或证据保存等环节出错时，工具返回 ``status="error"``。若请求漏传必填的 ``camera``，工具集会在调用前返回含 ``error`` 和 ``errors`` 的参数错误。

.. dropdown:: 其他可请求的几何量

   .. list-table::
      :header-rows: 1
      :widths: 30 70

      * - 字段
        - 需要的观测和返回值
      * - ``inner_rim_center``
        - 能从图像或深度中分辨开口的内侧上沿时，返回内沿拟合圆的圆心。只有整个物体的外轮廓时，内沿仍然未知。
      * - ``body_radii``
        - 从可见外侧曲面拟合圆柱或圆台，返回沿轴线可见范围两端的半径和对应的轴向范围。
      * - ``plane``、``plane_normal``
        - 一片平面上分布足够广的三维点可以确定平面。前者返回平面上的点、法向等参数，后者只返回单位法向。
      * - ``plate_center``
        - 根据盘沿以内平坦的圆形承托面及其边界，返回这个平面上的圆心。
      * - ``lid_center``
        - 根据圆形盖子的平坦封口面及其边界，返回盖面上的圆心。凸起的卷边排除在外。
      * - ``rectangle_center``
        - 观察主矩形顶面的四条实际边缘，并利用相机位姿区分边缘与遮挡，返回顶面中心。
      * - ``body_segment_center``
        - 观察近似圆柱形的主瓶身，用半径变化或可见的物体端部确定这段瓶身的两端，返回该段轴线上的中点。

处理流程
~~~~~~~~

每次调用只从指定相机读取一帧观测。接下来的步骤取决于主规划器要测什么。

- **杯壁上的一个点。** 把目标描述为黄白杯子的外壁，默认的 ``measurement="surface_point"`` 就会让定位器在可见杯壁上选一个像素。程序从对齐的三维坐标图中读取该点的位置，通过 ``world_xyz`` 返回。
- **杯身的可见范围。** 使用 ``measurement="visible_region"`` 时，程序还会分割目标区域。分割得到的掩码标出哪些像素属于杯身。程序取这些像素对应的有效三维点，用 ``world_xyz`` 返回各坐标轴上的中位数，用 ``visible_bounds_xyz`` 返回各轴的最小值和最大值。中位数可能落在空处。
- **部位的中心、方向或尺寸。** 给出 ``required_geometry`` 时，工具按这份列表执行拟合。它从各部位的掩码中提取三维点，选出相连的点、过滤离群点，再为口沿和底边拟合圆，或为杯身比较圆柱和圆台。程序检查支持点数、覆盖范围、拟合残差和部位之间是否一致，再写入 ``geometry[field]``。

前两类请求直接使用可见表面的观测点，成功时标为 ``measured``。工具把观测以及用到的掩码、点云和拟合依据保存到共享 ``EnvState`` 的原步骤下。拟合字段的 ``source_refs`` 指向对应记录，测量本身不会推进观测步骤。

观测接口
~~~~~~~~

接入工具的程序负责准备相机观测和标定数据。它通过 ``ObservationProvider`` 的 ``get_observation(*, camera, step)`` 解析相机名和步骤，返回一份 ``GeometryObservation``。其中 ``step`` 必须对应共享 ``EnvState`` 中已有的记录，``world_frame_id`` 要能稳定标识所用的参考坐标系。主规划器在请求中只需引用相机和步骤。

每帧都需要原始分辨率的 RGB 图像（``rgb``，形状为 H × W × 3，类型为 ``uint8``），并至少通过下列一种方式提供与图像对齐的三维数据。

- 有三维坐标图时，提供 ``world_xyz``。它的形状为 H × W × 3，每个像素保存参考坐标系中的三维位置，单位为米。
- 有深度图时，提供沿相机光轴的米制深度 ``depth_m``、3 × 3 针孔相机内参 ``intrinsics`` 和 4 × 4 刚体变换 ``camera_to_world``。工具据此将像素转换为三维坐标。相机的正 Z 轴朝前，参考坐标系的方向由接入程序定义。

观测数组会复制为只读快照。有效点的坐标必须有限。如果提供了深度，或者可以根据相机位姿求出光轴深度，深度还必须有限且为正。同时提供深度图和坐标图时也会检查深度。可选的布尔掩码 ``valid_mask`` 可进一步排除无效像素，位于参考坐标系原点的点也按相同规则判断。

提供相机位姿有助于确定平面法向的朝向。清理口沿或杯身点云时，只有显式给出参考坐标系中的上方向 ``up_direction``，程序才会尝试识别并移除背景平面。可用 ``source_refs`` 记录这帧观测的来源。

职责与初始化
~~~~~~~~~~~~

在机器人程序中接入时，先准备观测提供方、共享的 ``EnvState``、视觉定位器 ``grounder`` 和分割器 ``segmenter``。只测表面点时可省略分割器。分割可以使用已有的 SAM3 客户端，相关服务由机器人程序连接。也可以传入 ``model`` 和可选的 ``base_url`` 创建已有的视觉定位器 ``GroundingAgent``，用 ``fit_options`` 调整几何拟合的条件。

下面的示例在工具集外创建 ``GeometryTool``，然后将实例的 ``geometry`` 注册到已有工具集，并在分发时绑定取消检查。``task_instructions`` 是主规划器原有的任务提示词。

.. code-block:: python

   from functools import partial

   from toolkits.geometry.geometry_tool import GeometryTool
   from toolkits.geometry.prompts import GEOMETRY_INSTRUCTIONS

   geometry_tool = GeometryTool(
       state=state,
       observations=observations,
       grounder=grounder,
       segmenter=segmenter,
   )
   geometry = geometry_tool.geometry
   toolkit.add_tool(
       geometry.with_handler(
           partial(geometry, check_cancelled=toolkit.raise_if_cancelled)
       )
   )
   system_prompt = task_instructions + "\n\n" + GEOMETRY_INSTRUCTIONS

工具集的注册和分发机制见 :doc:`interfaces`。

.. _geometry-prompt-design:

提示词设计
----------

主规划器怎样提出测量请求
~~~~~~~~~~~~~~~~~~~~~~~~

``GEOMETRY_INSTRUCTIONS`` 指导主规划器在 ``query`` 中写清是哪一个物体，用 ``measurement`` 或 ``required_geometry`` 指定想要的结果。杯子例子中，“黄白色”和“在灰杯左侧”共同指明目标，“杯口中心”则说明要测什么。像素和三维坐标都由工具确定。

提示也说明怎样使用答案。如果任务只需要杯口半径，上例的 ``partial`` 结果已经给出了这个量。如果任务还需要杯高，主规划器可以查看 ``height`` 的 ``reason`` 和缺失部位，再决定是否在任务预算内换个角度，让同一帧里能同时看到杯口和杯底。移动相机或机器人由主规划器安排。

怎样定位和分割每个部位
~~~~~~~~~~~~~~~~~~~~~~

程序用 ``PART_DESCRIPTIONS`` 把部位要求加在原始目标描述前。测量示例中的六个量时，定位器实际收到下面三条描述，每次一条，使用同一张 RGB 图像。

.. code-block:: text

   the visible circular rim boundary of
   the yellow and white mug left of the gray cup

   the visible curved outer body, excluding handle, rim and interior, of
   the yellow and white mug left of the gray cup

   the visible circular bottom face or its boundary of
   the yellow and white mug left of the gray cup

共享的 ``GroundingAgent`` 在调用视觉模型时附上原图宽高，要求在指定部位清楚可见的实物表面上选点，避开遮挡、物体轮廓和图像边缘。模型只依据 RGB 图像和文字，将原图像素坐标按 ``[column, row]`` 的顺序填入 ``point_xy``，无法辨认时该字段为 ``null``。程序检查图像尺寸和像素范围，再按行、列读取坐标。上面合成示例中的杯沿选点 ``[104, 64]`` 对应坐标图的第 64 行、第 104 列（从 0 开始）。

需要分割时，分割器先按部位文字生成掩码，程序再检查掩码是否包含独立选出的像素。如果找不到掩码或掩码漏掉该点，最多再用这个点提示分割一次，分数阈值保持不变。每个部位最多定位一次、分割两次，所以杯子请求的上限是三次定位、六次分割。

测量 ``body_segment_center`` 时，定位器要在主瓶身上选点，分割器则接到整瓶描述，以便程序利用瓶肩、瓶颈和底部判断主瓶身的范围。

使用边界与运行验证
------------------

遮挡、深度缺失或一个掩码中混入多个表面，都可能使几何量保持未知。拟合残差表示观测点与所选形状吻合的程度。它与支持点数、分割分数一起用于诊断拟合质量，真实位置误差需要另外测量。

工具每次处理单个视角。主规划器比较两次观测时，需要确认目标、部位定义和参考坐标系都相同，物体移动后应重新测量。多视角融合、完整盒体或内腔重建超出当前工具的范围。

安装测试依赖后，在仓库根目录运行离线检查。

.. code-block:: bash

   python -m pytest tests/unit_tests/toolkits/geometry -q

这些检查使用合成观测和替代模型服务，覆盖工具集分发、坐标转换、分割重试、取消及部位拟合。真实场景中的测量精度和机器人任务成功率需要单独验证。
