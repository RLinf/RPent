:html_theme.sidebar_secondary.remove:

.. _libero-pro-astra-results:

LIBERO-PRO: GPT-6 Astra
=======================

:doc:`返回排行榜 <../benchmarks>`

**Codex / GPT-6 Astra / Reasoning 开启 / effort low。**
核验快照：**2026-09-14 14:00:55 UTC**。

.. note::

   这是尚未完成的评测：已完成 **619/800 回合**，其中 **564 成功、55 失败、
   181 待完成**。八套中已有六套完成，Goal Task、Goal Swap 与完整 Overall
   成功率仍为 **未报告**。待完成回合不计作失败，也不以已完成子集的成功率
   替代完整 800 回合的 Overall。

套件结果
--------

.. csv-table::
   :name: astra-suite-progress
   :header: "套件", "已完成", "成功", "失败", "成功率", "状态"
   :widths: 25 15 10 10 15 25
   :class: table-sm

   "Spatial Task", "100/100", "100", "0", "100.00%", "已完成"
   "Spatial Swap", "100/100", "98", "2", "98.00%", "已完成"
   "Object Task", "100/100", "100", "0", "100.00%", "已完成"
   "Object Swap", "100/100", "99", "1", "99.00%", "已完成"
   "Goal Task", "19/100", "10", "9", "未报告", "部分完成"
   "Goal Swap", "0/100", "0", "0", "未报告", "待完成"
   "Long Task", "100/100", "85", "15", "85.00%", "已完成"
   "Long Swap", "100/100", "72", "28", "72.00%", "已完成"
   "Overall", "619/800", "564", "55", "未报告", "181 待完成"

每套有 10 个任务（ID 0-9），评测 seed 为 1-10，共 100 回合。
Seed 0 仅用于探索，不计入评测分母。

逐任务与 seed 结果
------------------

``S`` 表示成功，``F`` 表示失败，``-`` 表示尚无可计分结果。仅当某任务的
十个评测 seed 全部完成时计算任务成功率。任务说明保留原实验的
``task_language`` 及任务排序。

Spatial Task
~~~~~~~~~~~~

**已完成 100/100；100 成功、0 失败、0 待完成。**

.. csv-table::
   :name: astra-seeds-spatial-task
   :header: "任务", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "任务结果"
   :class: table-sm

   "0", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "1", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "2", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "3", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "4", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "5", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "6", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "7", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "8", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "9", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"

.. dropdown:: 任务说明

   .. csv-table::
      :header: "任务", "task_language"
      :widths: 10 90

      "0", "Pick the akita black bowl not between the plate and the ramekin and place it on the plate"
      "1", "Pick the akita black bowl next to the cookie box and place it on the plate"
      "2", "Pick the akita black bowl next to the plate and place it on the plate"
      "3", "Pick the akita black bowl on the top of the cabinet and place it on the plate"
      "4", "Pick the akita black bowl on the top of the wooden cabinet and place it on the plate"
      "5", "Pick the akita black bowl on the cookie box and place it on the plate"
      "6", "Pick the akita black bowl on the stove and place it on the plate"
      "7", "Pick the akita black bowl on the top of the cabinet and place it on the plate"
      "8", "Pick the akita black bowl next to the ramekin and place it on the plate"
      "9", "Pick the akita black bowl on the stove and place it on the plate"

Spatial Swap
~~~~~~~~~~~~

**已完成 100/100；98 成功、2 失败、0 待完成。**

.. csv-table::
   :name: astra-seeds-spatial-swap
   :header: "任务", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "任务结果"
   :class: table-sm

   "0", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "1", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "2", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "3", "S", "F", "F", "S", "S", "S", "S", "S", "S", "S", "8/10 (80%)"
   "4", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "5", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "6", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "7", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "8", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "9", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"

.. dropdown:: 任务说明

   .. csv-table::
      :header: "任务", "task_language"
      :widths: 10 90

      "0", "Pick the akita black bowl between the plate and the ramekin and place it on the plate"
      "1", "Pick the akita black bowl next to the ramekin and place it on the plate"
      "2", "Pick the akita black bowl from table center and place it on the plate"
      "3", "Pick the akita black bowl on the cookies box and place it on the plate"
      "4", "Pick the akita black bowl in the top layer of the wooden cabinet and place it on the plate"
      "5", "Pick the akita black bowl on the ramekin and place it on the plate"
      "6", "Pick the akita black bowl next to the cookies box and place it on the plate"
      "7", "Pick the akita black bowl on the stove and place it on the plate"
      "8", "Pick the akita black bowl next to the plate and place it on the plate"
      "9", "Pick the akita black bowl on the wooden cabinet and place it on the plate"

Object Task
~~~~~~~~~~~

**已完成 100/100；100 成功、0 失败、0 待完成。**

.. csv-table::
   :name: astra-seeds-object-task
   :header: "任务", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "任务结果"
   :class: table-sm

   "0", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "1", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "2", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "3", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "4", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "5", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "6", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "7", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "8", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "9", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"

.. dropdown:: 任务说明

   .. csv-table::
      :header: "任务", "task_language"
      :widths: 10 90

      "0", "Pick the cream cheese and place it in the basket"
      "1", "Pick the alphabet soup and place it in the basket"
      "2", "Pick the tomato sauce and place it in the basket"
      "3", "Pick the ketchup and place it in the basket"
      "4", "Pick the milk and place it in the basket"
      "5", "Pick the bbq sauce and place it in the basket"
      "6", "Pick the orange juice and place it in the basket"
      "7", "Pick the butter and place it in the basket"
      "8", "Pick the salad dressing and place it in the basket"
      "9", "Pick the chocolate pudding and place it in the basket"

Object Swap
~~~~~~~~~~~

**已完成 100/100；99 成功、1 失败、0 待完成。**

.. csv-table::
   :name: astra-seeds-object-swap
   :header: "任务", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "任务结果"
   :class: table-sm

   "0", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "1", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "2", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "3", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "4", "S", "S", "S", "S", "S", "S", "S", "S", "F", "S", "9/10 (90%)"
   "5", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "6", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "7", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "8", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "9", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"

.. dropdown:: 任务说明

   .. csv-table::
      :header: "任务", "task_language"
      :widths: 10 90

      "0", "Pick the alphabet soup and place it in the basket"
      "1", "Pick the cream cheese and place it in the basket"
      "2", "Pick the salad dressing and place it in the basket"
      "3", "Pick the bbq sauce and place it in the basket"
      "4", "Pick the ketchup and place it in the basket"
      "5", "Pick the tomato sauce and place it in the basket"
      "6", "Pick the butter and place it in the basket"
      "7", "Pick the milk and place it in the basket"
      "8", "Pick the chocolate pudding and place it in the basket"
      "9", "Pick the orange juice and place it in the basket"

Goal Task
~~~~~~~~~

**已完成 19/100；10 成功、9 失败、81 待完成。**

.. csv-table::
   :name: astra-seeds-goal-task
   :header: "任务", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "任务结果"
   :class: table-sm

   "0", "F", "F", "F", "S", "F", "F", "F", "F", "F", "F", "1/10 (10%)"
   "1", "S", "S", "S", "S", "S", "S", "S", "S", "S", "``-``", "完成 9/10"
   "2", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "完成 0/10"
   "3", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "完成 0/10"
   "4", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "完成 0/10"
   "5", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "完成 0/10"
   "6", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "完成 0/10"
   "7", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "完成 0/10"
   "8", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "完成 0/10"
   "9", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "完成 0/10"

.. dropdown:: 任务说明

   .. csv-table::
      :header: "任务", "task_language"
      :widths: 10 90

      "0", "open the bottom drawer of the cabinet"
      "1", "Put the plate on the stove"
      "2", "put the wine bottle in the bowl"
      "3", "Open the top layer of the drawer and put the cream cheese inside"
      "4", "Put the plate on the top of the drawer"
      "5", "Push the cream cheese to the front of the stove"
      "6", "put the wine bottle in the bowl"
      "7", "Turn off the stove"
      "8", "Put the wine bottle on the plate"
      "9", "Put the cream cheese on the rack"

Goal Swap
~~~~~~~~~

**已完成 0/100；0 成功、0 失败、100 待完成。**

.. csv-table::
   :name: astra-seeds-goal-swap
   :header: "任务", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "任务结果"
   :class: table-sm

   "0", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "完成 0/10"
   "1", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "完成 0/10"
   "2", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "完成 0/10"
   "3", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "完成 0/10"
   "4", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "完成 0/10"
   "5", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "完成 0/10"
   "6", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "完成 0/10"
   "7", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "完成 0/10"
   "8", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "完成 0/10"
   "9", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "``-``", "完成 0/10"

.. dropdown:: 任务说明

   .. csv-table::
      :header: "任务", "task_language"
      :widths: 10 90

      "0", "Open the middle layer of the drawer"
      "1", "Put the bowl on the stove"
      "2", "Put the wine bottle on the top of the drawer"
      "3", "Open the top layer of the drawer and put the bowl inside"
      "4", "Put the bowl on the top of the drawer"
      "5", "Push the plate to the front of the stove"
      "6", "Put the cream cheese on the bowl"
      "7", "Turn on the stove"
      "8", "Put the bowl on the plate"
      "9", "Put the wine bottle on the rack"

Long Task
~~~~~~~~~

**已完成 100/100；85 成功、15 失败、0 待完成。**

.. csv-table::
   :name: astra-seeds-long-task
   :header: "任务", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "任务结果"
   :class: table-sm

   "0", "S", "F", "S", "S", "S", "S", "S", "S", "F", "S", "8/10 (80%)"
   "1", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "2", "S", "S", "F", "S", "S", "S", "S", "S", "S", "S", "9/10 (90%)"
   "3", "F", "S", "F", "F", "S", "F", "S", "S", "S", "F", "5/10 (50%)"
   "4", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "5", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "6", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "7", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "8", "F", "F", "S", "S", "S", "S", "S", "S", "S", "S", "8/10 (80%)"
   "9", "S", "F", "S", "S", "F", "F", "S", "S", "F", "F", "5/10 (50%)"

.. dropdown:: 任务说明

   .. csv-table::
      :header: "任务", "task_language"
      :widths: 10 90

      "0", "put both the cream cheese and the tomato sauce in the basket"
      "1", "put both the alphabet soup and the butter in the basket"
      "2", "turn on the stove and put the pan on it"
      "3", "put the bottle in the bottom drawer of the cabinet and close it"
      "4", "put the yellow and white mug on the left plate and put the white mug on the right plate"
      "5", "pick up the cup and place it in the back compartment of the caddy"
      "6", "put the red mug on the plate and put the chocolate pudding to the right of the plate"
      "7", "put both the ketchup and the cream cheese box in the basket"
      "8", "put the left moka pot on the stove"
      "9", "put the white mug in the microwave and close it"

Long Swap
~~~~~~~~~

**已完成 100/100；72 成功、28 失败、0 待完成。**

.. csv-table::
   :name: astra-seeds-long-swap
   :header: "任务", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "任务结果"
   :class: table-sm

   "0", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "1", "F", "S", "S", "S", "S", "F", "S", "S", "F", "S", "7/10 (70%)"
   "2", "S", "S", "S", "S", "S", "S", "S", "F", "S", "S", "9/10 (90%)"
   "3", "F", "S", "S", "S", "S", "S", "F", "S", "S", "S", "8/10 (80%)"
   "4", "S", "S", "S", "S", "S", "S", "S", "F", "S", "S", "9/10 (90%)"
   "5", "S", "F", "S", "S", "S", "S", "S", "S", "S", "F", "8/10 (80%)"
   "6", "F", "S", "F", "S", "F", "S", "S", "F", "F", "F", "4/10 (40%)"
   "7", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "10/10 (100%)"
   "8", "S", "S", "F", "S", "F", "S", "S", "F", "S", "F", "6/10 (60%)"
   "9", "F", "F", "S", "F", "F", "F", "F", "F", "F", "F", "1/10 (10%)"

.. dropdown:: 任务说明

   .. csv-table::
      :header: "任务", "task_language"
      :widths: 10 90

      "0", "put both the alphabet soup and the tomato sauce in the basket"
      "1", "put both the cream cheese box and the butter in the basket"
      "2", "turn on the stove and put the moka pot on it"
      "3", "put the black bowl in the bottom drawer of the cabinet and close it"
      "4", "put the white mug on the left plate and put the yellow and white mug on the right plate"
      "5", "pick up the book and place it in the back compartment of the caddy"
      "6", "put the white mug on the plate and put the chocolate pudding to the right of the plate"
      "7", "put both the alphabet soup and the cream cheese box in the basket"
      "8", "put both moka pots on the stove"
      "9", "put the yellow and white mug in the microwave and close it"

协议与来源
----------

成功只认原始环境 ``states.json`` 中的 ``terminated = true``，不以 planner
最终文本判断。每个 task/seed 最多保留一个最终计分结果，已有有效失败保持不变。
本报告保留全部 800 个计划位置，包括尚无结果的位置。

运行版本：`014a0fa <https://github.com/RLinf/RPent/commit/014a0fa97f69c991ee5e0f62f14e1d6f89c3dcd7>`_。
评测使用 5000 秒 planner timeout 与 10000 步环境上限。
Long 来自 ``libero_long_gpt6_astra_20260907``；Spatial/Object/Goal 来自
``libero_pro_remaining_gpt6_astra_20260913``。各批次使用各自评测前冻结的
seed-0 memory，八套并非共用同一份 memory 快照。这些结果不是只改变 planner
模型的受控对照实验；上述参数是实验来源记录，不是新的 RPent 默认值。

发布前已将贡献者提供的报告与套件合计、619 个已计分 task/seed 位置交叉核对。
原报告 SHA-256 为 ``e9880e58953af964edbcdd50d586b54c22a66f3a7fab54d919a4205be9436fdf``。
公开的`清理后结果快照 <https://raw.githubusercontent.com/RLinf/misc/1f93c83cbbef749ce55af0399b1bd405e8e9c7a7/rpent/benchmarks/astra-pro-20260914.json>`_
包含逐 task/seed 结果、待完成位置、统计及协议元数据，不包含服务器路径、
凭据或原始轨迹。本页整理已有实验证据，不代表文档更新期间重新执行了策略评测。
