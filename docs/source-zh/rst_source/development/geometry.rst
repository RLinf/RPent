.. _geometry-toolkit:

几何工具包
==========

几何工具将物体或可见部位的描述转换为有明确定义的空间测量结果。主规划器通过统一的 ``geometry`` 接口提交目标、所需几何量和观测引用，GeometryAgent 协调视觉定位、分割、点云清理和形状拟合，返回位置、方向、尺寸及对应证据。

设计目的
--------

抓取、放置或对齐需要具体部位的几何信息。“杯口中心”“杯身中心”和“可见表面上的一点”具有不同含义，不能用一个选定像素相互替代。几何工具为每个测量量规定定义，并将结果与源观测和形状假设一起返回。

主规划器决定测量什么、执行什么动作，GeometryAgent 判断哪些观测部位能够支持该测量。视觉模型负责语义定位，度量观测与几何算法负责空间计算。这样，不同环境可以使用相同的请求与结果约定，任务决策与几何测量分别处理。

.. _geometry-module-design:

模块设计
--------

GeometryAgent 使用固定的程序流程。视觉语言模型选择可见像素，注入的分割器生成掩码；部位展开、调用顺序与预算、点云处理、模型比较和结果验收由程序控制。

职责与初始化
~~~~~~~~~~~~

宿主先准备观测访问、共享证据存储和感知客户端，在 toolkit 外构造 GeometryAgent，再注册该实例的 ``geometry`` 工具声明。子代理负责处理测量请求；宿主负责传感器采集、标定、坐标系定义和动作执行。

构造时传入 ``ObservationProvider``、共享的 ``EnvState``、视觉定位器 ``grounder`` 和可选的 ``segmenter``。也可以用 ``model`` 与 ``base_url`` 创建已有的视觉定位适配器，用 ``fit_options`` 配置几何算法条件。表面点请求需要定位器，区域与拟合请求还需要分割器；兼容的 SAM3 客户端可以直接作为分割器传入，模块自身不启动服务。

下面演示普通 toolkit 的注册方式，其中 ``state``、``observations``、感知客户端和 ``toolkit`` 由宿主提供：

.. code-block:: python

   from functools import partial

   from toolkits.geometry.geometry_agent import GeometryAgent

   agent = GeometryAgent(
       state=state,
       observations=observations,
       grounder=grounder,
       segmenter=segmenter,
   )
   geometry = agent.geometry
   toolkit.add_tool(
       geometry.with_handler(
           partial(geometry, check_cancelled=toolkit.raise_if_cancelled)
       )
   )

取消检查在工具分发时绑定。GeometryAgent 没有动作工具，也不推进已记录的观测步骤。证据不足时返回缺失原因，由主规划器决定获取新视角或执行恢复动作。

观测接口
~~~~~~~~

观测提供方实现 ``get_observation(*, camera: str, step: int) -> GeometryObservation``，自行解析相机名称与 ``step=-1``，返回一份对齐的观测快照。返回的步骤必须对应共享 ``EnvState`` 中已有的记录。每次请求只调用一次提供方，同一请求中的全部部位使用该快照。

观测显式提供以下数据及坐标语义：

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - 字段
     - 约定
   * - ``camera``、``step``、``world_frame_id``
     - 调用方定义的相机名称、已解析的记录步骤，以及稳定的度量坐标系标识。
   * - ``rgb``
     - 原图分辨率的 H × W × 3 RGB 数组，类型为 uint8。
   * - ``world_xyz``
     - 与图像对齐的 H × W × 3 三维坐标图，单位为米，属于提供方指定的参考坐标系。
   * - ``depth_m``、``intrinsics``、``camera_to_world``
     - 可替代坐标图：沿光轴的米制深度、针孔相机的 3 × 3 内参，以及相机到参考坐标系的 4 × 4 刚体变换。相机正 Z 轴朝前，参考坐标系方向由调用方定义。
   * - ``valid_mask``
     - 可选的对齐布尔有效掩码。非有限坐标、提供的非正深度无效；参考坐标原点处的有限点是有效点。
   * - ``up_direction``、``source_refs``
     - 可选的参考坐标系上方向，以及标识源证据的引用。

数组会被复制为只读快照。同时提供深度和坐标图时，深度也参与有效性判断。相机位姿可用于确定平面法向符号和光轴深度信息。接口不要求固定的传感器文件名或相机名称枚举，也不自行生成世界坐标系标识。

背景平面移除使用提供的 ``up_direction``；未提供时，清理流程跳过自动背景平面移除，不假定参考坐标系的 Z 轴朝上。提供方应先把传感器深度转换为沿光轴的米制深度，再构造观测。

请求与结果约定
~~~~~~~~~~~~~~

主规划器传入目标描述、提供方支持的相机名称、记录步骤及所需几何量，工具调用中无需提供标定参数：

.. code-block:: json

   {
     "query": "the yellow and white mug left of the gray cup",
     "camera": "inspection_rgb",
     "step": 0,
     "required_geometry": [
       "rim_center", "body_center", "bottom_center", "axis", "radius", "height"
     ]
   }

设置 ``required_geometry`` 时，读取 ``geometry[field]``。每个字段包含数值、定义、单位、``world_frame_id``、源步骤、形状假设、质量信息和证据引用。``radius`` 表示杯口半径，``body_center`` 表示已观测到的共轴杯口和杯底中心的中点，``height`` 表示两者的间距。

只测量可见材质点时，不设置 ``required_geometry``，使用 ``measurement="surface_point"``；需要可见分割样本的逐轴中位数和边界时，使用 ``measurement="visible_region"``。这些结果通过 ``world_xyz`` 等字段返回，属于提供方指定的参考坐标系。区域统计不是拟合中心，中位数也可能落在空处。

主规划器使用数值规划动作前，应检查状态：

- ``usable``：全部请求量通过几何检查。
- ``partial``：部分量可用，其余字段保持未知。
- ``need_observation``：当前证据不足，原因和缺失部位用于指导主规划器的下一次观测。
- ``error``：输入、观测提供方、模型服务或证据保存失败。

拟合值在对应假设下标为 ``inferred``，证据不足的字段标为 ``unknown``，值为 null。残差、支持点数和分割分数是诊断信息，不是经过标定的位置误差。结果引用已保存的观测、掩码、清理点云和拟合支撑，完整数组单独存储，不放入工具返回文本。

处理流程
~~~~~~~~

程序按所需测量执行以下步骤：

1. **固定观测**：获取一份源快照，保留相机、步骤、坐标系和证据元数据。
2. **定位并分割部位**：视觉模型选择可见材质点；区域请求使用与独立选点一致的分割掩码。
3. **提取并清理点云**：使用对齐的度量坐标及有效性信息，保留连通的部位证据，过滤离群点。背景移除需要显式上方向。
4. **拟合所需形状**：拟合圆形边缘和平面，比较圆柱与圆台曲面，或推导有足够支撑的圆形面、矩形顶面及主瓶身段中心。
5. **检查并返回测量**：检查支持点、覆盖范围、残差、退化情况和部位一致性，保存证据，报告缺失量。

完整物体掩码不能直接确定内口中心，局部杯身的可见长度不能直接确定完整高度。低残差说明候选点与模型相符，还需要观测部位支持该测量的定义。

.. _geometry-prompt-design:

提示词设计
----------

主规划器调用约定
~~~~~~~~~~~~~~~~

``GEOMETRY_INSTRUCTIONS`` 提供可复用的主规划器提示。宿主将这段文字与已注册的工具描述一起用于规划器：

.. code-block:: python

   from toolkits.geometry.prompts import GEOMETRY_INSTRUCTIONS

   system_prompt = task_instructions + "\n\n" + GEOMETRY_INSTRUCTIONS

提示要求主规划器保留外观、部位身份和空间关系，选择表面测量或拟合模式，读取状态、定义、坐标系标识、源证据及质量信息。主规划器描述目标，不自行提供猜测的像素或 XYZ。

结果不完整或需要新证据时，主规划器先检查原因，再在任务预算内选择有用的观测；跨视角保持目标身份，比较相同坐标系中的结果，物体移动后重新测量。提示词和工具声明不向 GeometryAgent 提供动作能力。

部位需求展开
~~~~~~~~~~~~

程序将 ``required_geometry`` 映射为需要观测的部位，在原始目标描述前添加部位要求。每次视觉调用只接收一项展开后的描述，并保留目标原有的外观与空间关系。例如：

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 请求量
     - 展开后的观测需求
   * - ``rim_center``、``radius``
     - 可见圆形口沿边界。
   * - ``axis``
     - 可见外侧曲面，排除把手、口沿和内部表面。
   * - ``body_center``、``height``
     - 分别定位口沿、外侧杯身和杯底，再检查各部位的一致性。
   * - ``inner_rim_center``
     - 内口上沿边缘，排除外唇、把手和内部底面。

内部视觉定位提示
~~~~~~~~~~~~~~~~

已有的 API/Codex 视觉定位适配器只接收 RGB 和目标文字。提示中的 ``{query}`` 是展开后的部位描述，尺寸来自原始图像：

.. code-block:: text

   Locate {query} in this robot camera image. Preserve the named object
   part and requested spatial relation. Select a clearly visible material
   point away from silhouettes, occlusion and image borders. Do not
   substitute another object or part. Return null if it cannot be identified.
   The image has {width} columns and {height} rows. Return point_xy as
   [column, row] in ORIGINAL pixel units, not normalized coordinates.
   Only select a point; do not use tools or execute robot actions.

结构化结果中的 ``point_xy`` 在可见时为原图像素单位的 ``[column, row]``，无法识别时为 null。程序检查图像尺寸和像素范围，转换为内部行列顺序，再从固定的观测中获得度量坐标。视觉模型不接收深度、物体真值或机器人工具。

分割提示与调用边界
~~~~~~~~~~~~~~~~~~

区域请求先用部位文字请求分割，要求掩码包含独立视觉选点；候选未找到或未包含该点时，最多再用该点作为正提示调用一次。主瓶身段测量会定位瓶身，但分割整瓶，以保留瓶肩、瓶颈和底部的上下文。

每个部位最多一次视觉定位、两次分割，不进行同义词搜索或降低阈值重试。分割限定候选证据，部位中心仍由程序通过清理与拟合计算。提示词约束语义任务，代码执行调用预算和几何验收条件。

使用边界与运行验证
------------------

遮挡、深度缺失、混合表面或目标定位错误可能使部分几何量保持未知。当前不支持完整盒体或内腔恢复，也不进行多视角融合。只比较相同参考坐标系和点定义的结果，物体移动后重新请求测量。

安装测试依赖后，可运行离线检查：

.. code-block:: bash

   python -m pytest tests/unit_tests/toolkits/geometry -q

这些检查使用合成数据和替代模型服务，覆盖通用观测输入、toolkit 分发、度量反投影、感知预算、取消和拟合流程。真实场景的位置精度与机器人任务成功率需要另外进行实验。
