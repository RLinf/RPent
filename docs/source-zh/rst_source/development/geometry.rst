.. _geometry-toolkit:

几何工具包
==========

``GeometryTool`` 是一个通用的几何感知工具。主规划器通过 ``geometry`` 描述要测的物体和部位，工具从指定的一帧相机观测中计算位置、方向或尺寸，并标明每个测量值是根据哪些可见部位计算得到。

设计目的
--------

规划机器人动作时，主规划器可能需要知道目标部位的位置、朝向或尺寸。表面上一点的位置可以从三维观测中直接读取，而部位的中心、轴线或半径通常需要根据边缘或曲面上的多个点来估计。`GeometryTool` 为不同机器人程序提供统一的测量接口。主规划器描述要测的目标和所需的几何量，工具根据指定的一帧相机观测给出测量结果。如果当前视角只能支持部分测量，工具会返回可用的值，并说明其余量缺少哪些依据，供主规划器决定是否需要换个角度观察。

.. _geometry-module-design:

模块设计
--------

主规划器调用 ``geometry`` 时，用 ``query`` 描述要测的物体或部位，用 ``camera`` 指定相机。工具从这台相机读取一帧已记录的观测。视觉模型在彩色图像中定位目标，工具再用同一帧里对应的三维点进行测量。

.. _processing-flow:

选择测量方式
~~~~~~~~~~~~

``measurement`` 默认为 ``surface_point``，工具通过 ``world_xyz`` 返回目标表面一个可见点的三维坐标。设为 ``visible_region`` 时，``visible_bounds_xyz`` 给出可见区域内有效三维点在各轴上的取值范围，``world_xyz`` 给出各轴的中位数。要测量某个部位的中心、朝向或尺寸，可以用 ``required_geometry`` 指定 1 到 10 个几何量，工具根据三维点拟合形状，并将结果逐项写入 ``geometry``。

.. _request-and-result-contract:

测一只杯子的尺寸
~~~~~~~~~~~~~~~~

假设桌上放着一只圆口杯，我们想知道杯口有多宽、杯子有多高。杯口直径可以由半径算出，因此主规划器请求杯口半径 ``rim_radius`` 和杯高 ``height``。本例中的相机名为 ``table_camera``。省略 ``step`` 时，默认值 ``-1`` 表示读取这台相机最近记录的一帧。

.. code-block:: json

   {
     "query": "the cup on the table",
     "camera": "table_camera",
     "required_geometry": [
       "rim_radius",
       "height"
     ]
   }

.. list-table::
   :header-rows: 1
   :widths: 20 43 37

   * - 请求量
     - 这一帧需要看到什么
     - 能返回什么
   * - ``rim_radius``
     - 一段足够长的圆形杯沿，边缘上有有效的三维点。
     - 拟合圆的半径，单位为米。``radius`` 是同一个量的另一个名称。
   * - ``height``
     - 杯口和杯底的边缘各能拟合出一个圆，两端分开且共轴。若杯壁也能拟合，它须与两端共轴，且可见部分位于两端之间。
     - 两个圆心之间的距离，单位为米。

假设从上方只能看清杯沿，拟合出的半径约为 4 厘米。下面是这个情形下的示例返回片段。

.. code-block:: json

   {
     "status": "partial",
     "geometry": {
       "rim_radius": {
         "value": 0.04,
         "unit": "m",
         "evidence_status": "inferred"
       },
       "height": {
         "value": null,
         "evidence_status": "unknown",
         "reason": "both_consistent_end_boundaries_required"
       }
     }
   }

这里的 ``partial`` 表示两项测量中只有一项有值。半径乘以 2，得到杯口直径约 8 厘米。杯高仍是 ``null``，原因是这一帧缺少可与杯口配对的杯底边缘。所有请求的量都有值时，状态为 ``usable``。当前观测不足以给出任何请求结果时，状态为 ``need_observation``。

直接从表面点或可见区域得到的结果标为 ``measured``，拟合值标为 ``inferred``。读取拟合结果时，``definition`` 和 ``unit`` 说明数值的含义和单位，``world_frame_id`` 和 ``source_step`` 标明参考坐标系和观测步骤。``assumptions`` 和 ``quality`` 记录拟合所用的假设及检查结果。``source_parts`` 列出用到的部位，``source_refs`` 可找到保存在共享 ``EnvState`` 原步骤下的掩码、三维点和拟合记录。

.. dropdown:: 其他可请求的几何量

   - ``rim_center`` 用杯沿上的可见三维点拟合圆，返回杯口中心的坐标。
   - ``axis`` 根据杯身外侧的曲面估计轴线，返回轴线上一点和轴的方向。
   - ``plane_normal`` 用平坦表面上的三维点拟合平面，返回垂直于该平面的单位向量。

   还可以用同样的方式请求杯底中心、圆盖中心等几何量。

.. _observation-contract:

相机需要提供什么数据
~~~~~~~~~~~~~~~~~~~~

机器人程序通过 ``ObservationProvider`` 的 ``get_observation(*, camera, step)`` 提供一份 ``GeometryObservation``。相机名由接入程序定义，解析后的 ``step`` 对应共享 ``EnvState`` 中已有的一条记录，``world_frame_id`` 应始终指向同一个参考坐标系。

每帧都需要原始分辨率的 RGB 图像（``rgb``，形状为 H × W × 3，类型为 ``uint8``），并至少通过下列一种方式提供与图像对齐的三维数据。

- 有三维坐标图时，提供 ``world_xyz``。它的形状为 H × W × 3，每个像素保存参考坐标系中的三维位置，单位为米。
- 有深度图时，提供沿相机光轴的米制深度 ``depth_m``、3 × 3 针孔相机内参 ``intrinsics`` 和 4 × 4 刚体变换 ``camera_to_world``。工具据此将像素转换为三维坐标。相机的正 Z 轴朝前，参考坐标系的方向由接入程序定义。

.. dropdown:: 有效点和可选的观测数据

   观测数组会复制为只读快照。有效点的坐标必须有限。如果提供了深度，或者可以根据相机位姿求出光轴深度，深度还必须有限且为正。同时提供深度图和坐标图时也会检查深度。可选的布尔掩码 ``valid_mask`` 可进一步排除无效像素，参考坐标系原点处的点也按相同规则判断。

   相机位姿有助于确定平面法向的朝向。清理口沿或侧壁的三维点时，提供参考坐标系中的上方向 ``up_direction`` 后，工具才会尝试识别并移除背景平面。可用 ``source_refs`` 记录这帧观测的来源。

.. _responsibilities-and-initialization:

在机器人程序中注册工具
~~~~~~~~~~~~~~~~~~~~~~

先准备共享的 ``EnvState``、观测提供方和视觉定位器 ``grounder``。只测一个表面点时，这些就够了。测可见区域或拟合几何量时，还需提供分割器 ``segmenter``，例如机器人程序已连接的 SAM3 客户端。也可以传入 ``model`` 和可选的 ``base_url`` 来创建视觉定位器 ``GroundingAgent``，用 ``fit_options`` 调整拟合条件。

下面的代码在已有的 ``toolkit`` 中注册实例的 ``geometry`` 方法，并在分发请求时绑定取消检查。

.. code-block:: python

   from functools import partial

   from toolkits.geometry.geometry_tool import GeometryTool

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

工具集的注册和分发机制见 :doc:`interfaces`。

.. _geometry-prompt-design:

提示词设计
----------

主规划器怎样提出测量请求
~~~~~~~~~~~~~~~~~~~~~~~~

接入程序将 ``GEOMETRY_INSTRUCTIONS`` 添加到主规划器已有的任务提示词中，配合已注册的 ``geometry`` 工具使用。

.. code-block:: python

   from toolkits.geometry.prompts import GEOMETRY_INSTRUCTIONS

   system_prompt = task_instructions + "\n\n" + GEOMETRY_INSTRUCTIONS

提示词要求主规划器用文字指明目标和所需的测量。例子里桌上只有一只杯子，用“桌上的杯子”就足以指明目标。如果画面中有相似物体，可以补充颜色或相对位置。像素由工具在图像中选取，三维坐标由相机观测提供。

收到 ``partial`` 时，主规划器先看缺少哪一项以及对应的 ``reason``。如果任务只需要杯口直径，已有的半径就够用。如果还需要杯高，主规划器可以在任务预算内安排换个视角，让新的一帧能同时看到杯口和杯底。

.. _expanding-part-requirements:

怎样定位和分割每个部位
~~~~~~~~~~~~~~~~~~~~~~

工具用 ``PART_DESCRIPTIONS`` 把要看的部位加在原始 ``query`` 前。测杯口半径要看杯沿，测杯高还会查看外壁和杯底。两项测量共用同一次杯沿定位，传给定位器的杯沿描述如下。

.. code-block:: text

   the visible circular rim boundary of the cup on the table

.. _internal-visual-grounding-prompt:

``GroundingAgent`` 将这段文字、原始 RGB 图像和图像尺寸交给视觉模型，让它在指定部位清楚可见的实物表面上选点。例如可以选杯沿上的一点。模型通过 ``point_xy`` 按列、行的顺序返回原图中的像素坐标，无法辨认部位时返回 ``null``。

.. _segmentation-prompts-and-call-bounds:

分割器先根据部位描述圈出图像区域。工具检查这个区域是否包含独立选出的像素。如果没有找到区域，或区域漏掉了该点，工具会用这个点再提示分割一次。每个部位最多定位一次、分割两次。

使用边界与运行验证
------------------

遮挡、深度缺失或分割区域混入其他表面，都可能使测量值保持未知。比较两帧的结果前，先确认目标、部位定义和参考坐标系一致。物体移动后应重新测量。把多个视角合成完整物体或内腔的三维模型需要额外的重建步骤。拟合残差反映观测点与所选形状的吻合程度，真实测量误差需要用独立的测量结果验证。

读取观测、调用模型或保存记录时发生错误，工具返回 ``status="error"``。如果请求缺少必填参数 ``camera``，工具集会在调用前返回含 ``error`` 和 ``errors`` 的参数错误。

安装测试依赖后，在仓库根目录运行离线检查。

.. code-block:: bash

   python -m pytest tests/unit_tests/toolkits/geometry -q

这些检查使用合成观测和替代模型服务，覆盖工具分发、坐标转换、分割重试、取消和部位拟合。
