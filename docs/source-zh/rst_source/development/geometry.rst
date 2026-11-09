.. _geometry-toolkit:

几何工具包
==========

使用 ``GeometryAgent.geometry``，可以从已记录的 LIBERO RGB-D 观测帧中获取可见表面位置，或拟合物体部位的几何量。主规划器提供目标描述、所需几何量和观测引用；子代理通过内部感知工具选择图像点并计算世界坐标。请求本身不会移动机器人。当前实现接入 LIBERO 观测，环境、SAM3 和规划器的配置见 :doc:`../simulators/libero`。

模块分工
--------

工具包将几何测量、机器人执行和独立评估分开。``toolkits/geometry/`` 下各模块的职责如下：

``geometry_agent.py``
   视觉选点、分割、坐标查询和部位拟合。

``geometry.py``
   点云清理，以及平面、圆、圆柱和圆台拟合。

``reference_geometry.py``
   内口、承托面、盖面、矩形顶面和主瓶身段的参考几何。

``candidate_geometry.py``
   为宽松误差评估提供带有估计方式标记的近似中心。

``evaluation/checklist.py``
   固定任务、随机种子和参考点定义。

``evaluation/experiment.py``
   初始帧采集和几何推理的命令行入口。

``evaluation/metrics.py``
   独立真值校验、位置误差和覆盖率统计。

``evaluation/evaluation_scene.py``
   仅供评估器使用的编译后网格与同一帧的物体刚体位姿。

LIBERO 环境、动作原语和工具注册位于 ``robots/libero/``。共享的视觉模型适配位于 ``rpent/robots/components/grounding_agent.py``。可直接导入子代理：

.. code-block:: python

   from toolkits.geometry.geometry_agent import GeometryAgent

请求几何量
----------

先用 ``view_env_state`` 识别与任务相关的物体，再将目标描述、所需几何量和观测引用传给 ``geometry``：

.. code-block:: json

   {
     "query": "the yellow and white mug left of the gray cup",
     "camera": "agentview",
     "step": 0,
     "required_geometry": [
       "rim_center", "body_center", "bottom_center", "axis", "radius", "height"
     ]
   }

主规划器无需提供标定参数。GeometryAgent 从记录的观测中读取相互匹配的图像、世界坐标、深度和相机元数据。设置 ``required_geometry`` 后，请求使用几何拟合，取代 ``measurement`` 对应的表面测量模式。每个部位调用一次视觉定位模型、最多两次 SAM，不进行同义词搜索或降低阈值重试。杯子请求最多涉及三个部位；单独请求 ``plane`` 或 ``plane_normal`` 时，可增加第四个部位。所有部位共用一个固定的观测帧。子代理没有动作工具。

部位清理与拟合
--------------

每个请求部位都使用对应的观测候选和几何模型。下表说明支持的证据及清理方法：

.. list-table::
   :header-rows: 1
   :widths: 18 38 44

   * - 部位
     - 观测候选
     - 清理与模型
   * - 杯口
     - SAM 部位掩码，以及视觉杯口选点附近的 RGB／深度前景边缘。
     - 与目标连通的点云；三维圆一致性估计和稳健精化。
   * - 杯身
     - 排除把手、杯口和内部表面后的外侧曲面。
     - 去除背景平面、保留连通点云、过滤邻域离群点；稳健比较圆柱和圆台模型。
   * - 杯底
     - 可见底面的边界。
     - 保留底部平面，在观测到的边界上拟合圆。
   * - 平面
     - 请求中的可见平面区域。
     - 保留目标平面，使用 RANSAC 和 SVD。
   * - 内口
     - 内侧掩码孔边缘、内部前景深度跳变，或视觉选定内缘处的连通 RGB 轮廓。
     - 排除外轮廓和内部底面；清理最大的候选连通分量，拟合三维圆。
   * - 盘面／封口盖
     - 中央承托面或封口盖平面，以及该面的可见物理边界。
     - 拟合表面平面，将观测边界投影到该平面上，再拟合圆。
   * - 矩形顶面
     - 主要顶面平面及四条物理边。
     - 拒绝共面标签裁剪、前景覆盖和图像截断；有边缘及角点证据后才拟合矩形。
   * - 主瓶身段
     - 整瓶掩码，以及位于主瓶身上的视觉锚点。
     - 在锚点附近初始化主导圆柱；通过瓶肩半径过渡或真实端缘确定瓶身段的两个界限。

体素采样保留实际输入样本。杯口和杯身请求仅在掩码外观测到水平背景平面，且该平面与视觉锚点分离时，才去除这个背景平面。把手和内部表面的混入点通过杯身模型残差排除；混合点云不保证能得到被接受的模型。点云过小或不连通、圆弧过短、区域共线、支持不足或拟合病态时，都需要更多证据。低残差本身不足以证明几何量可观测。

内口候选保留最大的连通分量，不要求该分量靠近视觉材质点。SAM 掩码仍须包含这个点，它还用于背景平面分离和圆拟合一致性检查。体素采样、离群点过滤和全部圆拟合验收条件仍然生效。

初始算法门槛记录在 ``quality.fit_options`` 中：内点距离阈值为 3 mm，体素大小为 1 mm，至少需要 30 个点、60% 的支持比例、180 度的圆周边界覆盖和 120 度的杯身覆盖。圆和杯身拟合还要求表面残差的第 95 百分位不超过拟合参考半径的 5%。这个相对形状门槛用于防止小型非圆边界仅因绝对阈值较粗而通过检查。优化和采样都有次数或数量上限。这些是实验性验收设置，不是实测精度或经校准的传感器不确定度；float16 世界坐标仍有量化限制。

读取拟合结果
------------

拟合结果应读取 ``geometry[field]``，而不是表面测量模式的 ``world_xyz``。每个字段都提供 ``value``、定义、单位、``world_frame_id``、源步骤、证据引用、假设、质量信息和 ``evidence_status``：

- ``rim_center``、``rim_radius`` 和 ``radius`` 对应拟合出的可见杯口圆。``radius`` 是杯口半径的别名，不表示杯身在所有高度处的半径。
- ``bottom_center`` 需要可见的圆形杯底边界，不能用底部平面区域的中位数或杯内的选点代替。
- ``axis`` 来自拟合的圆形截面杯身，或相互一致的已观测端部圆。两个端部不完整时，轴线的正负方向采用约定；它不是把手朝向或开门方向。
- ``body_radii`` 仅描述可见轴向采样范围两端的拟合半径。
- ``body_center`` 是已观测到的共轴杯口圆心和杯底圆心的中点；``height`` 是共轴端部假设下两者的间距。局部杯身的可见长度不足以确定这两个量。这里的中心不是质量中心或体积中心。
- ``plane`` 和 ``plane_normal`` 描述可见平面；有相机元数据时，法向符号朝向相机。它们不足以确定完整表面的尺寸。
- ``inner_rim_center`` 是内口上沿边缘所在平面内的三维中心。无法辨认内缘时，结果保持未知，不能用外口沿代替。
- ``plate_center`` 是圆形中央承托面在该面高度处的中心；``lid_center`` 是圆形封口盖的中心，并随倾斜的盖面平面变化。
- ``rectangle_center`` 是主要矩形顶面的中心，需要四条物理边及角点邻域共同支持。可见片段的外接框不足以确定这个中心，也不能据此恢复完整实心盒体。
- ``body_segment_center`` 是近似圆柱形主瓶身段的轴向中点，排除瓶颈和瓶嘴。两个界限都需要观测证据，局部点云的最小值／最大值不能定义主瓶身段中心。瓶身轴线可以是水平的。

拟合量在对应形状假设下标为 ``inferred``。没有足够支持的字段保持 ``unknown``，值为 null。残差、支持点数和 SAM 分数是分别记录的诊断量；``calibrated_error_m=null`` 表示定位误差仍然未知。

``status="usable"`` 表示全部请求量通过了实现中的检查；``partial`` 表示仅部分通过。``need_observation`` 提供原因以及缺失的字段或部位，由主规划器决定并执行新的观测移动。``error`` 表示输入、服务或证据保存失败。请求的几何拟合被拒绝时，不应悄悄改用表面中位数。

``geometry_XXX.json`` 保存部位请求、调用预算、清理统计、模型比较和一致性检查。掩码、候选点云、清理后点云、清理索引和拟合内点数组分别作为当前步骤的产物保存。``fit_points_artifact`` 标识拟合内点索引所引用的数组；圆形面使用显式推导的平面投影点，``observed_support_artifact`` 另行保留原始观测边界。数组不进入工具返回文本。当前实现不支持完整盒体／内腔恢复和多视角融合。

.. _geometry-permissive-evaluation:

宽松误差评估
~~~~~~~~~~~~

评估清单中心的定位误差时，可以用 ``permissive=True`` 构造 ``GeometryAgent``，或在实验的 ``infer`` 命令中添加 ``--permissive``。严格拟合被拒绝后，系统会给出带标记的、仅基于已有观测的候选估计，不再请求补充视角。应检查 ``strict_accepted``、``estimate_method`` 和 ``evaluation_only``：返回坐标可能只是不完整可见范围的估计。默认模式仍采用严格检查。放宽后的门槛和回退方法见 `评估说明 <https://github.com/RLinf/RPent/blob/main/toolkits/geometry/evaluation/README.md#宽松误差评估>`_。

80 组场景、160 个参考点的协议见 :ref:`固定参考点清单实验 <libero-fixed-reference-experiment>`，环境准备见 :doc:`LIBERO 文档 <../simulators/libero>`。该实验只使用初始 agentview 帧，记录失败时不改变目标，也不移动到其他视角。它与交互式主规划器的恢复流程分别定义。

表面测量
--------

接触点使用 ``measurement="surface_point"``；可见样本统计使用 ``measurement="visible_region"``，并且不设置 ``required_geometry``。前者返回一个选定的可见材质点，后者返回包含视觉选点的分割区域中各坐标轴的中位数和边界。这些统计量不是拟合中心，中位数可能落在空处。

先用 agentview 确认目标身份；需要更近的几何观测时，在新的 wrist 观测中请求同一部位。只比较世界坐标系和点定义都一致的结果。目标移动后需要重新请求几何测量；使用记忆中的偏移量时，其参考点定义也须与返回的测量一致。

模型配置与验证
--------------

API 和 Codex 规划器默认继承主规划器的模型及服务地址。可用 ``--geometry-model provider:model`` 或 ``--geometry-model codex:model`` 指定模型，并用 ``--geometry-base-url`` 覆盖继承的 ``--base-url`` 服务地址。单独使用 ``codex`` 时读取其配置中的默认模型。Claude Code 规划器需要显式指定 ``--geometry-model``。视觉模型只接收 RGB 和目标文字，不接收机器人动作工具或物体真值。

``tests/unit_tests/toolkits/geometry/`` 下的离线冒烟测试使用解析点云和固定的感知输出。真实场景精度需要独立标注或仅供评估器使用的仿真真值；任务操作成功率和几何测量误差须分别测量。采样一致性方法参考标准 `PCL 模型 <https://pointclouds.org/documentation/group__sample__consensus.html>`_，稳健精化使用 `SciPy least_squares <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html>`_。
