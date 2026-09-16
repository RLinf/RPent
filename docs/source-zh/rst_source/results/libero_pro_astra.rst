:html_theme.sidebar_secondary.remove:

.. _libero-pro-astra-results:

LIBERO-PRO: GPT-6 Astra
==========================================================================================

.. raw:: html

   <link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/RLinf/misc@c980b2da0b4b4fa425b9d0a4a283ff3ee23661c1/rpent/benchmarks/docs.css">

:doc:`返回排行榜 <../benchmarks>`

**Codex / GPT-6 Astra / low / reasoning.**

2026-09-15 09:32:29 UTC 核验：**741 成功、59 失败，800 回合；Overall 92.63%（741/800）。**

套件结果
------------------------------------------------------------------------------------------

.. _astra-suite-progress:

.. csv-table::
   :header: "套件", "成功 / 评测回合", "失败", "成功率"
   :class: table-sm

   "Object Task", "100/100", "0", "100.00%"
   "Spatial Task", "100/100", "0", "100.00%"
   "Goal Swap", "99/100", "1", "99.00%"
   "Object Swap", "99/100", "1", "99.00%"
   "Spatial Swap", "98/100", "2", "98.00%"
   "Goal Task", "88/100", "12", "88.00%"
   "Long Task", "85/100", "15", "85.00%"
   "Long Swap", "72/100", "28", "72.00%"
   "Overall", "741/800", "59", "92.63%"

每套件 10 个任务（ID 0–9），各评测 seed 1–10；seed 0 仅用于探索。S 为成功，F 为失败。800 个位置各计分一次，有效失败保留。任务明细按原始 ID 升序排列，seed 列顺序不变；方法榜仍按成功率排名。

GPT-6 Astra · low · reasoning · Object Task · 100/100
----------------------------------------------------------------------------------------------------------

.. csv-table::
   :header: "ID", "任务", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "成功率"
   :class: table-sm

   "0", "Pick the cream cheese and place it in the basket", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "1", "Pick the alphabet soup and place it in the basket", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "2", "Pick the tomato sauce and place it in the basket", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "3", "Pick the ketchup and place it in the basket", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "4", "Pick the milk and place it in the basket", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "5", "Pick the bbq sauce and place it in the basket", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "6", "Pick the orange juice and place it in the basket", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "7", "Pick the butter and place it in the basket", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "8", "Pick the salad dressing and place it in the basket", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "9", "Pick the chocolate pudding and place it in the basket", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"

GPT-6 Astra · low · reasoning · Spatial Task · 100/100
------------------------------------------------------------------------------------------------------------

.. csv-table::
   :header: "ID", "任务", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "成功率"
   :class: table-sm

   "0", "Pick the akita black bowl not between the plate and the ramekin and place it on the plate", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "1", "Pick the akita black bowl next to the cookie box and place it on the plate", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "2", "Pick the akita black bowl next to the plate and place it on the plate", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "3", "Pick the akita black bowl on the top of the cabinet and place it on the plate", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "4", "Pick the akita black bowl on the top of the wooden cabinet and place it on the plate", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "5", "Pick the akita black bowl on the cookie box and place it on the plate", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "6", "Pick the akita black bowl on the stove and place it on the plate", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "7", "Pick the akita black bowl on the top of the cabinet and place it on the plate", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "8", "Pick the akita black bowl next to the ramekin and place it on the plate", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "9", "Pick the akita black bowl on the stove and place it on the plate", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"

GPT-6 Astra · low · reasoning · Goal Swap · 99/100
----------------------------------------------------------------------------------------------------

.. csv-table::
   :header: "ID", "任务", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "成功率"
   :class: table-sm

   "0", "Open the middle layer of the drawer", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "1", "Put the bowl on the stove", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "2", "Put the wine bottle on the top of the drawer", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "3", "Open the top layer of the drawer and put the bowl inside", "F", "S", "S", "S", "S", "S", "S", "S", "S", "S", "90%"
   "4", "Put the bowl on the top of the drawer", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "5", "Push the plate to the front of the stove", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "6", "Put the cream cheese on the bowl", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "7", "Turn on the stove", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "8", "Put the bowl on the plate", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "9", "Put the wine bottle on the rack", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"

GPT-6 Astra · low · reasoning · Object Swap · 99/100
--------------------------------------------------------------------------------------------------------

.. csv-table::
   :header: "ID", "任务", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "成功率"
   :class: table-sm

   "0", "Pick the alphabet soup and place it in the basket", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "1", "Pick the cream cheese and place it in the basket", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "2", "Pick the salad dressing and place it in the basket", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "3", "Pick the bbq sauce and place it in the basket", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "4", "Pick the ketchup and place it in the basket", "S", "S", "S", "S", "S", "S", "S", "S", "F", "S", "90%"
   "5", "Pick the tomato sauce and place it in the basket", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "6", "Pick the butter and place it in the basket", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "7", "Pick the milk and place it in the basket", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "8", "Pick the chocolate pudding and place it in the basket", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "9", "Pick the orange juice and place it in the basket", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"

GPT-6 Astra · low · reasoning · Spatial Swap · 98/100
----------------------------------------------------------------------------------------------------------

.. csv-table::
   :header: "ID", "任务", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "成功率"
   :class: table-sm

   "0", "Pick the akita black bowl between the plate and the ramekin and place it on the plate", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "1", "Pick the akita black bowl next to the ramekin and place it on the plate", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "2", "Pick the akita black bowl from table center and place it on the plate", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "3", "Pick the akita black bowl on the cookies box and place it on the plate", "S", "F", "F", "S", "S", "S", "S", "S", "S", "S", "80%"
   "4", "Pick the akita black bowl in the top layer of the wooden cabinet and place it on the plate", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "5", "Pick the akita black bowl on the ramekin and place it on the plate", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "6", "Pick the akita black bowl next to the cookies box and place it on the plate", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "7", "Pick the akita black bowl on the stove and place it on the plate", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "8", "Pick the akita black bowl next to the plate and place it on the plate", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "9", "Pick the akita black bowl on the wooden cabinet and place it on the plate", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"

GPT-6 Astra · low · reasoning · Goal Task · 88/100
----------------------------------------------------------------------------------------------------

.. csv-table::
   :header: "ID", "任务", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "成功率"
   :class: table-sm

   "0", "open the bottom drawer of the cabinet", "F", "F", "F", "S", "F", "F", "F", "F", "F", "F", "10%"
   "1", "Put the plate on the stove", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "2", "put the wine bottle in the bowl", "F", "S", "S", "S", "S", "F", "S", "S", "S", "S", "80%"
   "3", "Open the top layer of the drawer and put the cream cheese inside", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "4", "Put the plate on the top of the drawer", "S", "S", "S", "S", "S", "F", "S", "S", "S", "S", "90%"
   "5", "Push the cream cheese to the front of the stove", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "6", "put the wine bottle in the bowl", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "7", "Turn off the stove", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "8", "Put the wine bottle on the plate", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "9", "Put the cream cheese on the rack", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"

GPT-6 Astra · low · reasoning · Long Task · 85/100
----------------------------------------------------------------------------------------------------

.. csv-table::
   :header: "ID", "任务", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "成功率"
   :class: table-sm

   "0", "put both the cream cheese and the tomato sauce in the basket", "S", "F", "S", "S", "S", "S", "S", "S", "F", "S", "80%"
   "1", "put both the alphabet soup and the butter in the basket", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "2", "turn on the stove and put the pan on it", "S", "S", "F", "S", "S", "S", "S", "S", "S", "S", "90%"
   "3", "put the bottle in the bottom drawer of the cabinet and close it", "F", "S", "F", "F", "S", "F", "S", "S", "S", "F", "50%"
   "4", "put the yellow and white mug on the left plate and put the white mug on the right plate", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "5", "pick up the cup and place it in the back compartment of the caddy", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "6", "put the red mug on the plate and put the chocolate pudding to the right of the plate", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "7", "put both the ketchup and the cream cheese box in the basket", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "8", "put the left moka pot on the stove", "F", "F", "S", "S", "S", "S", "S", "S", "S", "S", "80%"
   "9", "put the white mug in the microwave and close it", "S", "F", "S", "S", "F", "F", "S", "S", "F", "F", "50%"

GPT-6 Astra · low · reasoning · Long Swap · 72/100
----------------------------------------------------------------------------------------------------

.. csv-table::
   :header: "ID", "任务", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "成功率"
   :class: table-sm

   "0", "put both the alphabet soup and the tomato sauce in the basket", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "1", "put both the cream cheese box and the butter in the basket", "F", "S", "S", "S", "S", "F", "S", "S", "F", "S", "70%"
   "2", "turn on the stove and put the moka pot on it", "S", "S", "S", "S", "S", "S", "S", "F", "S", "S", "90%"
   "3", "put the black bowl in the bottom drawer of the cabinet and close it", "F", "S", "S", "S", "S", "S", "F", "S", "S", "S", "80%"
   "4", "put the white mug on the left plate and put the yellow and white mug on the right plate", "S", "S", "S", "S", "S", "S", "S", "F", "S", "S", "90%"
   "5", "pick up the book and place it in the back compartment of the caddy", "S", "F", "S", "S", "S", "S", "S", "S", "S", "F", "80%"
   "6", "put the white mug on the plate and put the chocolate pudding to the right of the plate", "F", "S", "F", "S", "F", "S", "S", "F", "F", "F", "40%"
   "7", "put both the alphabet soup and the cream cheese box in the basket", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "100%"
   "8", "put both moka pots on the stove", "S", "S", "F", "S", "F", "S", "S", "F", "S", "F", "60%"
   "9", "put the yellow and white mug in the microwave and close it", "F", "F", "S", "F", "F", "F", "F", "F", "F", "F", "10%"

协议与来源
------------------------------------------------------------------------------------------

成功只认原始环境轨迹的 ``terminated = true``，不是 planner 消息。运行版本为 ``014a0fa97f69c991ee5e0f62f14e1d6f89c3dcd7``，planner 时限 5000 秒，环境上限 10000 步。这些是实验记录，不是新增运行时默认值。

Long 属于 ``libero_long_gpt6_astra_20260907``；Spatial/Object/Goal 属于 ``libero_pro_remaining_gpt6_astra_20260913``。两批分别使用评测前冻结的 seed-0 memory，不是同一个快照。

800 个唯一回合与任务说明均已对照贡献者报告和核验 JSON 检查，此前 600 个结果不变。百分比来自精确计数；741/800 按四舍五入显示为 92.63%，不是从一位小数汇总反推。

Report SHA-256: ``627d6d8a95fb1e10357961694867cd0bf0a2aa9011c96c8a33504b904cb96c80``.

Verified JSON SHA-256: ``60b675da35557900f921befd84db306514723ebe5a9db73af2c72ae38bc46578``.

`公开脱敏快照 <https://raw.githubusercontent.com/RLinf/misc/c980b2da0b4b4fa425b9d0a4a283ff3ee23661c1/rpent/benchmarks/astra-pro-20260915.json>`_.
包含任务说明、seed 成败、计数与公开协议元数据，不含服务器路径、凭据或原始轨迹。本轮发布未重跑实验。
