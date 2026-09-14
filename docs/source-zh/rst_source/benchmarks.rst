:html_theme.sidebar_secondary.remove:

.. _benchmark-results:
.. _benchmark-leaderboard:
.. _leaderboard:

RPent 排行榜
============

仿真基准上的已发布成功率。

.. tab-set::

   .. tab-item:: LIBERO
      :class-content: sd-border-0 sd-px-0

      **Standard LIBERO**

      Spatial、Object、Goal、Long 四个标准套件的 Overall 成功率。

      .. list-table::
         :name: ranking-standard-libero
         :header-rows: 1
         :widths: 8 67 25
         :class: table-sm

         * - #
           - 方法 / 规划模型
           - 成功率
         * - 1
           - AtomVLA
           - **97.0%**
         * - 2
           - **RPent / Opus-4.8** ``max``
           - 96.0%
         * - 3
           - π_RLinf
           - 95.3%
         * - 4
           - π0
           - 94.2%
         * - 5
           - NORA
           - 79.5%
         * - 6
           - OpenVLA
           - 76.5%

      未报告: GPT-5.5, GPT-6 Astra.

      :ref:`协议与来源 <benchmark-source-p2>`

      .. dropdown:: 参考图表

         .. image:: https://raw.githubusercontent.com/RLinf/misc/1f93c83cbbef749ce55af0399b1bd405e8e9c7a7/rpent/benchmarks/standard-libero-zh-light.png
            :alt: Standard LIBERO
            :class: only-light
            :width: 100%

         .. image:: https://raw.githubusercontent.com/RLinf/misc/1f93c83cbbef749ce55af0399b1bd405e8e9c7a7/rpent/benchmarks/standard-libero-zh-dark.png
            :alt: Standard LIBERO
            :class: only-dark
            :width: 100%

         本设置中已收录的代表性方法；百分比保留来源精度。


   .. tab-item:: LIBERO-PRO
      :class-content: sd-border-0 sd-px-0

      **GPT-6 Astra：已完成 6/8 套件、619/800 回合。**
      :doc:`已核验结果与待完成范围 <results/libero_pro_astra>`

      .. tab-set::

         .. tab-item:: Overall
            :class-content: sd-border-0 sd-px-0

            **LIBERO-PRO Overall**

            覆盖 Spatial / Object / Goal / Long 的全部八个 Task / Swap 设置。

            .. list-table::
               :name: ranking-libero-pro
               :header-rows: 1
               :widths: 8 67 25
               :class: table-sm

               * - #
                 - 方法 / 规划模型
                 - 成功率
               * - 1
                 - **RPent / Opus-4.8** ``max``
                 - **82.4%**
               * - 2
                 - **RPent / GPT-5.5** ``xhigh``
                 - 72.1%
               * - 3
                 - π_RLinf
                 - 50.0%
               * - 4
                 - π0.5
                 - 11.0%
               * - 5
                 - AtomVLA
                 - 6.3%
               * - 6
                 - X-VLA
                 - 3.8%
               * - 7
                 - MolmoAct
                 - 1.5%
               * - 8
                 - π0
                 - 0.3%
               * - 9
                 - OpenVLA
                 - 0.0%
               * - 9
                 - NORA
                 - 0.0%

            GPT-6 Astra Overall：**未报告** （已完成 619/800 回合）。
            Cap-X 和 RATS 只覆盖六个设置，不参与此八项 Overall 排名。

            :ref:`协议与来源 <benchmark-source-astra-pro>` |
            :doc:`Astra 逐任务与 seed 结果 <results/libero_pro_astra>`

            .. dropdown:: 参考图表

               .. image:: https://raw.githubusercontent.com/RLinf/misc/1f93c83cbbef749ce55af0399b1bd405e8e9c7a7/rpent/benchmarks/libero-pro-zh-light.png
                  :alt: LIBERO-PRO
                  :class: only-light
                  :width: 100%

               .. image:: https://raw.githubusercontent.com/RLinf/misc/1f93c83cbbef749ce55af0399b1bd405e8e9c7a7/rpent/benchmarks/libero-pro-zh-dark.png
                  :alt: LIBERO-PRO
                  :class: only-dark
                  :width: 100%

               仅展示已报告成绩；未完成的 Astra 套件不绘制成绩柱。


         .. tab-item:: Spatial
            :class-content: sd-border-0 sd-px-0

            .. tab-set::

               .. tab-item:: Task
                  :class-content: sd-border-0 sd-px-0

                  **Spatial Task**

                  RPent 规划模型；每个配置评测 100 回合。

                  .. list-table::
                     :name: ranking-spatial-task
                     :header-rows: 1
                     :widths: 8 67 25
                     :class: table-sm

                     * - #
                       - 方法 / 规划模型
                       - 成功率
                     * - 1
                       - **RPent / GPT-6 Astra** ``low``
                       - **100%**
                     * - 2
                       - **RPent / Opus-4.8** ``max``
                       - 94.0%
                     * - 3
                       - **RPent / GPT-5.5** ``xhigh``
                       - 81.0%

                  GPT-6 Astra：成功 100/100，套件已完成。单套件成绩不替代完整 PRO Overall。

                  :ref:`协议与来源 <benchmark-source-astra-pro>` |
                  :doc:`Astra 逐任务与 seed 结果 <results/libero_pro_astra>`

                  .. dropdown:: 参考图表

                     .. image:: https://raw.githubusercontent.com/RLinf/misc/1f93c83cbbef749ce55af0399b1bd405e8e9c7a7/rpent/benchmarks/spatial-task-zh-light.png
                        :alt: PRO Spatial Task
                        :class: only-light
                        :width: 100%

                     .. image:: https://raw.githubusercontent.com/RLinf/misc/1f93c83cbbef749ce55af0399b1bd405e8e9c7a7/rpent/benchmarks/spatial-task-zh-dark.png
                        :alt: PRO Spatial Task
                        :class: only-dark
                        :width: 100%

                     仅展示已报告成绩；未完成的 Astra 套件不绘制成绩柱。


               .. tab-item:: Swap
                  :class-content: sd-border-0 sd-px-0

                  **Spatial Swap**

                  RPent 规划模型；每个配置评测 100 回合。

                  .. list-table::
                     :name: ranking-spatial-swap
                     :header-rows: 1
                     :widths: 8 67 25
                     :class: table-sm

                     * - #
                       - 方法 / 规划模型
                       - 成功率
                     * - 1
                       - **RPent / GPT-6 Astra** ``low``
                       - **98%**
                     * - 2
                       - **RPent / Opus-4.8** ``max``
                       - 80.0%
                     * - 3
                       - **RPent / GPT-5.5** ``xhigh``
                       - 69.0%

                  GPT-6 Astra：成功 98/100，套件已完成。单套件成绩不替代完整 PRO Overall。

                  :ref:`协议与来源 <benchmark-source-astra-pro>` |
                  :doc:`Astra 逐任务与 seed 结果 <results/libero_pro_astra>`

                  .. dropdown:: 参考图表

                     .. image:: https://raw.githubusercontent.com/RLinf/misc/1f93c83cbbef749ce55af0399b1bd405e8e9c7a7/rpent/benchmarks/spatial-swap-zh-light.png
                        :alt: PRO Spatial Swap
                        :class: only-light
                        :width: 100%

                     .. image:: https://raw.githubusercontent.com/RLinf/misc/1f93c83cbbef749ce55af0399b1bd405e8e9c7a7/rpent/benchmarks/spatial-swap-zh-dark.png
                        :alt: PRO Spatial Swap
                        :class: only-dark
                        :width: 100%

                     仅展示已报告成绩；未完成的 Astra 套件不绘制成绩柱。


         .. tab-item:: Object
            :class-content: sd-border-0 sd-px-0

            .. tab-set::

               .. tab-item:: Task
                  :class-content: sd-border-0 sd-px-0

                  **Object Task**

                  RPent 规划模型；每个配置评测 100 回合。

                  .. list-table::
                     :name: ranking-object-task
                     :header-rows: 1
                     :widths: 8 67 25
                     :class: table-sm

                     * - #
                       - 方法 / 规划模型
                       - 成功率
                     * - 1
                       - **RPent / GPT-6 Astra** ``low``
                       - **100%**
                     * - 2
                       - **RPent / GPT-5.5** ``xhigh``
                       - 94.0%
                     * - 3
                       - **RPent / Opus-4.8** ``max``
                       - 88.0%

                  GPT-6 Astra：成功 100/100，套件已完成。单套件成绩不替代完整 PRO Overall。

                  :ref:`协议与来源 <benchmark-source-astra-pro>` |
                  :doc:`Astra 逐任务与 seed 结果 <results/libero_pro_astra>`

                  .. dropdown:: 参考图表

                     .. image:: https://raw.githubusercontent.com/RLinf/misc/1f93c83cbbef749ce55af0399b1bd405e8e9c7a7/rpent/benchmarks/object-task-zh-light.png
                        :alt: PRO Object Task
                        :class: only-light
                        :width: 100%

                     .. image:: https://raw.githubusercontent.com/RLinf/misc/1f93c83cbbef749ce55af0399b1bd405e8e9c7a7/rpent/benchmarks/object-task-zh-dark.png
                        :alt: PRO Object Task
                        :class: only-dark
                        :width: 100%

                     仅展示已报告成绩；未完成的 Astra 套件不绘制成绩柱。


               .. tab-item:: Swap
                  :class-content: sd-border-0 sd-px-0

                  **Object Swap**

                  RPent 规划模型；每个配置评测 100 回合。

                  .. list-table::
                     :name: ranking-object-swap
                     :header-rows: 1
                     :widths: 8 67 25
                     :class: table-sm

                     * - #
                       - 方法 / 规划模型
                       - 成功率
                     * - 1
                       - **RPent / GPT-6 Astra** ``low``
                       - **99%**
                     * - 2
                       - **RPent / GPT-5.5** ``xhigh``
                       - 91.0%
                     * - 3
                       - **RPent / Opus-4.8** ``max``
                       - 90.0%

                  GPT-6 Astra：成功 99/100，套件已完成。单套件成绩不替代完整 PRO Overall。

                  :ref:`协议与来源 <benchmark-source-astra-pro>` |
                  :doc:`Astra 逐任务与 seed 结果 <results/libero_pro_astra>`

                  .. dropdown:: 参考图表

                     .. image:: https://raw.githubusercontent.com/RLinf/misc/1f93c83cbbef749ce55af0399b1bd405e8e9c7a7/rpent/benchmarks/object-swap-zh-light.png
                        :alt: PRO Object Swap
                        :class: only-light
                        :width: 100%

                     .. image:: https://raw.githubusercontent.com/RLinf/misc/1f93c83cbbef749ce55af0399b1bd405e8e9c7a7/rpent/benchmarks/object-swap-zh-dark.png
                        :alt: PRO Object Swap
                        :class: only-dark
                        :width: 100%

                     仅展示已报告成绩；未完成的 Astra 套件不绘制成绩柱。


         .. tab-item:: Goal
            :class-content: sd-border-0 sd-px-0

            .. tab-set::

               .. tab-item:: Task
                  :class-content: sd-border-0 sd-px-0

                  **Goal Task**

                  RPent 规划模型；每个配置评测 100 回合。

                  .. list-table::
                     :name: ranking-goal-task
                     :header-rows: 1
                     :widths: 8 67 25
                     :class: table-sm

                     * - #
                       - 方法 / 规划模型
                       - 成功率
                     * - 1
                       - **RPent / Opus-4.8** ``max``
                       - **87.0%**
                     * - 2
                       - **RPent / GPT-5.5** ``xhigh``
                       - 75.0%

                  GPT-6 Astra：**未报告**。已完成 19/100 回合，10 成功、9 失败，
                  剩余 81 回合。未完成套件不参与排名。

                  :ref:`协议与来源 <benchmark-source-astra-pro>` |
                  :doc:`Astra 逐任务与 seed 结果 <results/libero_pro_astra>`

                  .. dropdown:: 参考图表

                     .. image:: https://raw.githubusercontent.com/RLinf/misc/1f93c83cbbef749ce55af0399b1bd405e8e9c7a7/rpent/benchmarks/goal-task-zh-light.png
                        :alt: PRO Goal Task
                        :class: only-light
                        :width: 100%

                     .. image:: https://raw.githubusercontent.com/RLinf/misc/1f93c83cbbef749ce55af0399b1bd405e8e9c7a7/rpent/benchmarks/goal-task-zh-dark.png
                        :alt: PRO Goal Task
                        :class: only-dark
                        :width: 100%

                     仅展示已报告成绩；未完成的 Astra 套件不绘制成绩柱。


               .. tab-item:: Swap
                  :class-content: sd-border-0 sd-px-0

                  **Goal Swap**

                  RPent 规划模型；每个配置评测 100 回合。

                  .. list-table::
                     :name: ranking-goal-swap
                     :header-rows: 1
                     :widths: 8 67 25
                     :class: table-sm

                     * - #
                       - 方法 / 规划模型
                       - 成功率
                     * - 1
                       - **RPent / Opus-4.8** ``max``
                       - **87.0%**
                     * - 2
                       - **RPent / GPT-5.5** ``xhigh``
                       - 66.0%

                  GPT-6 Astra：**未报告**。已完成 0/100 回合，0 成功、0 失败，
                  剩余 100 回合。未完成套件不参与排名。

                  :ref:`协议与来源 <benchmark-source-astra-pro>` |
                  :doc:`Astra 逐任务与 seed 结果 <results/libero_pro_astra>`

                  .. dropdown:: 参考图表

                     .. image:: https://raw.githubusercontent.com/RLinf/misc/1f93c83cbbef749ce55af0399b1bd405e8e9c7a7/rpent/benchmarks/goal-swap-zh-light.png
                        :alt: PRO Goal Swap
                        :class: only-light
                        :width: 100%

                     .. image:: https://raw.githubusercontent.com/RLinf/misc/1f93c83cbbef749ce55af0399b1bd405e8e9c7a7/rpent/benchmarks/goal-swap-zh-dark.png
                        :alt: PRO Goal Swap
                        :class: only-dark
                        :width: 100%

                     仅展示已报告成绩；未完成的 Astra 套件不绘制成绩柱。


         .. tab-item:: Long
            :class-content: sd-border-0 sd-px-0

            .. tab-set::

               .. tab-item:: Task
                  :class-content: sd-border-0 sd-px-0

                  **Long Task**

                  RPent 规划模型；每个配置评测 100 回合。

                  .. list-table::
                     :name: ranking-long-task
                     :header-rows: 1
                     :widths: 8 67 25
                     :class: table-sm

                     * - #
                       - 方法 / 规划模型
                       - 成功率
                     * - 1
                       - **RPent / GPT-6 Astra** ``low``
                       - **85%**
                     * - 2
                       - **RPent / Opus-4.8** ``max``
                       - 71.0%
                     * - 3
                       - **RPent / GPT-5.5** ``xhigh``
                       - 52.0%

                  GPT-6 Astra：成功 85/100，套件已完成。单套件成绩不替代完整 PRO Overall。

                  :ref:`协议与来源 <benchmark-source-astra-pro>` |
                  :doc:`Astra 逐任务与 seed 结果 <results/libero_pro_astra>`

                  .. dropdown:: 参考图表

                     .. image:: https://raw.githubusercontent.com/RLinf/misc/1f93c83cbbef749ce55af0399b1bd405e8e9c7a7/rpent/benchmarks/long-task-zh-light.png
                        :alt: PRO Long Task
                        :class: only-light
                        :width: 100%

                     .. image:: https://raw.githubusercontent.com/RLinf/misc/1f93c83cbbef749ce55af0399b1bd405e8e9c7a7/rpent/benchmarks/long-task-zh-dark.png
                        :alt: PRO Long Task
                        :class: only-dark
                        :width: 100%

                     仅展示已报告成绩；未完成的 Astra 套件不绘制成绩柱。


               .. tab-item:: Swap
                  :class-content: sd-border-0 sd-px-0

                  **Long Swap**

                  RPent 规划模型；每个配置评测 100 回合。

                  .. list-table::
                     :name: ranking-long-swap
                     :header-rows: 1
                     :widths: 8 67 25
                     :class: table-sm

                     * - #
                       - 方法 / 规划模型
                       - 成功率
                     * - 1
                       - **RPent / GPT-6 Astra** ``low``
                       - **72%**
                     * - 2
                       - **RPent / Opus-4.8** ``max``
                       - 62.0%
                     * - 3
                       - **RPent / GPT-5.5** ``xhigh``
                       - 49.0%

                  GPT-6 Astra：成功 72/100，套件已完成。单套件成绩不替代完整 PRO Overall。

                  :ref:`协议与来源 <benchmark-source-astra-pro>` |
                  :doc:`Astra 逐任务与 seed 结果 <results/libero_pro_astra>`

                  .. dropdown:: 参考图表

                     .. image:: https://raw.githubusercontent.com/RLinf/misc/1f93c83cbbef749ce55af0399b1bd405e8e9c7a7/rpent/benchmarks/long-swap-zh-light.png
                        :alt: PRO Long Swap
                        :class: only-light
                        :width: 100%

                     .. image:: https://raw.githubusercontent.com/RLinf/misc/1f93c83cbbef749ce55af0399b1bd405e8e9c7a7/rpent/benchmarks/long-swap-zh-dark.png
                        :alt: PRO Long Swap
                        :class: only-dark
                        :width: 100%

                     仅展示已报告成绩；未完成的 Astra 套件不绘制成绩柱。


   .. tab-item:: RoboCasa
      :class-content: sd-border-0 sd-px-0

      **RoboCasa365 Target50**

      50 个任务等权的 Overall 成功率；共 340 个评测回合。

      .. list-table::
         :name: ranking-robocasa
         :header-rows: 1
         :widths: 8 67 25
         :class: table-sm

         * - #
           - 方法 / 规划模型
           - 成功率
         * - 1
           - **RPent / GPT-5.5** ``xhigh``
           - **57.1%**
         * - 2
           - **RPent / Opus-4.8** ``max``
           - 48.6%
         * - 3
           - WorldDreamer
           - 35.3%
         * - 4
           - RLDX-1
           - 30.0%
         * - 5
           - π0.5
           - 16.9%
         * - 6
           - π0
           - 14.8%

      未报告: GPT-6 Astra.

      :ref:`协议与来源 <benchmark-source-p4>`

      .. dropdown:: 参考图表

         .. image:: https://raw.githubusercontent.com/RLinf/misc/1f93c83cbbef749ce55af0399b1bd405e8e9c7a7/rpent/benchmarks/robocasa-zh-light.png
            :alt: RoboCasa365 Target50
            :class: only-light
            :width: 100%

         .. image:: https://raw.githubusercontent.com/RLinf/misc/1f93c83cbbef749ce55af0399b1bd405e8e9c7a7/rpent/benchmarks/robocasa-zh-dark.png
            :alt: RoboCasa365 Target50
            :class: only-dark
            :width: 100%

         本设置中已收录的代表性方法；百分比保留来源精度。


   .. tab-item:: RoboTwin
      :class-content: sd-border-0 sd-px-0

      **RoboTwin C2R**

      Clean 到 randomized 的迁移；50 个任务，RPent 共评测 250 回合。

      .. list-table::
         :name: ranking-robotwin
         :header-rows: 1
         :widths: 8 67 25
         :class: table-sm

         * - #
           - 方法 / 规划模型
           - 成功率
         * - 1
           - **RPent / Opus-4.8** ``max``
           - **58.4%**
         * - 2
           - **RPent / GPT-5.5** ``xhigh``
           - 58.0%
         * - 3
           - LingBot-VLA
           - 50.4%
         * - 4
           - π0.5
           - 47.9%
         * - 5
           - GR00T-N1.7
           - 20.7%
         * - 6
           - StarVLA
           - 10.6%

      未报告: GPT-6 Astra.

      :ref:`协议与来源 <benchmark-source-p6>`

      .. dropdown:: 参考图表

         .. image:: https://raw.githubusercontent.com/RLinf/misc/1f93c83cbbef749ce55af0399b1bd405e8e9c7a7/rpent/benchmarks/robotwin-zh-light.png
            :alt: RoboTwin C2R
            :class: only-light
            :width: 100%

         .. image:: https://raw.githubusercontent.com/RLinf/misc/1f93c83cbbef749ce55af0399b1bd405e8e9c7a7/rpent/benchmarks/robotwin-zh-dark.png
            :alt: RoboTwin C2R
            :class: only-dark
            :width: 100%

         本设置中已收录的代表性方法；百分比保留来源精度。


排名仅适用于各自评测范围；同分并列，未报告不记零分、不参与排名。
外部方法保留各自协议和样本量。参考图表展示已选取的方法，不代表全部排名项。

:ref:`模型配置 <id2>` | :ref:`完整结果 <libero-series>` |
:ref:`演示视频 <benchmark-demo>`

.. _id2:

模型配置
------------


.. list-table:: RPent 规划模型配置
   :header-rows: 1
   :widths: 25 30 25 20

   * - 后端
     - 模型
     - Reasoning
     - Effort
   * - Codex
     - GPT-5.5
     - 开启
     - ``xhigh``
   * - Claude Code
     - Opus-4.8
     - 开启
     - ``max``
   * - Codex
     - GPT-6 Astra
     - 开启
     - ``low``


``Reasoning`` 指模型的原生推理模式，``Effort`` 为实际配置的推理强度。
``xhigh`` 和 ``max`` 属于不同提供方的设置，不表示相同计算预算。
模型身份、后端对应关系及推理设置已经实验贡献者确认。

.. _libero-series:

LIBERO 系列
----------------

Standard LIBERO
~~~~~~~~~~~~~~~

标准 LIBERO 使用未施加 PRO 扰动的 Spatial、Object、Goal、Long 套件。
这里的 Long 为标准 LIBERO-10，与下表的 PRO Long Task/Swap 分开评测。


.. list-table:: RPent 成功率
   :header-rows: 1
   :widths: 31 23 23 23

   * - 评测项
     - GPT-5.5 / ``xhigh``
     - Opus-4.8 / ``max``
     - GPT-6 Astra / ``low``
   * - Spatial
     - 未报告
     - 97.0%
     - 未报告
   * - Object
     - 未报告
     - 100.0%
     - 未报告
   * - Goal
     - 未报告
     - 94.0%
     - 未报告
   * - Long
     - 未报告
     - 93.0%
     - 未报告
   * - 总体
     - 未报告
     - 96.0% (384/400)
     - 未报告


.. _libero-pro-long:

.. _libero-pro-across-task-families:

LIBERO-PRO
~~~~~~~~~~

Task 为指令重定向，Swap 为位置交换。Overall 覆盖 Spatial、Object、Goal、Long 的全部
八个 Task/Swap 单元；只完成 Long 的结果不构成该总体指标。


.. list-table:: RPent 成功率
   :header-rows: 1
   :widths: 31 23 23 23

   * - 评测项
     - GPT-5.5 / ``xhigh``
     - Opus-4.8 / ``max``
     - GPT-6 Astra / ``low``
   * - Spatial Task
     - 81.0%
     - 94.0%
     - 100% (100/100)
   * - Spatial Swap
     - 69.0%
     - 80.0%
     - 98% (98/100)
   * - Object Task
     - 94.0%
     - 88.0%
     - 100% (100/100)
   * - Object Swap
     - 91.0%
     - 90.0%
     - 99% (99/100)
   * - Goal Task
     - 75.0%
     - 87.0%
     - 未报告
   * - Goal Swap
     - 66.0%
     - 87.0%
     - 未报告
   * - Long Task
     - 52.0%
     - 71.0%
     - 85% (85/100)
   * - Long Swap
     - 49.0%
     - 62.0%
     - 72% (72/100)
   * - 总体
     - 72.1%
     - 82.4%
     - 未报告


.. _libero-pro-goal:


GPT-6 Astra 当前快照已完成 **619/800 回合** （564 成功、55 失败），
仍有 181 回合待完成。六套成绩已完整；Goal Task、Goal Swap 与 Overall
保持未报告。分套件、逐任务及逐 seed 统计见 :doc:`results/libero_pro_astra`。

LIBERO-PRO Goal：零样本
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

此消融不使用目标设置的 Task Specific Memory 和 Global Memory，
与上方使用记忆的 PRO 主结果分别统计。


.. list-table:: RPent 成功率
   :header-rows: 1
   :widths: 31 23 23 23

   * - 评测项
     - GPT-5.5 / ``xhigh``
     - Opus-4.8 / ``max``
     - GPT-6 Astra / ``low``
   * - Goal Task
     - 未报告
     - 79.0%
     - 未报告
   * - Goal Swap
     - 未报告
     - 31.0%
     - 未报告


RoboCasa365 Target50
--------------------

列出全部三个划分及 Overall。Overall 按 50 个任务均权；各划分的回合数不同，
不能直接把所有成功回合合并求比率。


.. list-table:: RPent 成功率
   :header-rows: 1
   :widths: 31 23 23 23

   * - 评测项
     - GPT-5.5 / ``xhigh``
     - Opus-4.8 / ``max``
     - GPT-6 Astra / ``low``
   * - Atomic-Seen
     - 92.0%
     - 79.4%
     - 未报告
   * - Composite-Seen
     - 61.0%
     - 47.5%
     - 未报告
   * - Composite-Unseen
     - 13.8%
     - 15.0%
     - 未报告
   * - 总体（任务均权）
     - 57.1%
     - 48.6%
     - 未报告


RoboTwin C2R
------------

C2R 表示从干净设置到随机设置的迁移评测。


.. list-table:: RPent 成功率
   :header-rows: 1
   :widths: 31 23 23 23

   * - 评测项
     - GPT-5.5 / ``xhigh``
     - Opus-4.8 / ``max``
     - GPT-6 Astra / ``low``
   * - C2R
     - 58.0%
     - 58.4%
     - 未报告


.. _id3:

外部基线参考
------------------

以下附表保留来源中的其他方法，供查看相关评测覆盖范围。它们不作为 RPent 的规划模型列，
也不与主表合并计分。直接执行动作的 VLA 方法没有单独规划器的 Reasoning/Effort 设置；
Cap-X 和 RATS 的具体规划模型与推理配置在这些来源中未报告。

.. dropdown:: Standard LIBERO


   .. list-table:: 参考方法
      :header-rows: 1
      :widths: 24 36 20 20

      * - 方法
        - 覆盖范围
        - 成功率
        - 来源
      * - OpenVLA
        - 四个标准套件
        - 76.5%
        - :ref:`表 2 <benchmark-source-p2>`
      * - NORA
        - 四个标准套件
        - 79.5%
        - :ref:`表 2 <benchmark-source-p2>`
      * - π0
        - 四个标准套件
        - 94.2%
        - :ref:`表 2 <benchmark-source-p2>`
      * - π_RLinf
        - 四个标准套件
        - 95.3%
        - :ref:`表 2 <benchmark-source-p2>`
      * - AtomVLA
        - 四个标准套件
        - 97.0%
        - :ref:`表 2 <benchmark-source-p2>`


   外部基线的样本量沿用各自来源，不统一套用 RPent 的每套件 100 回合。

.. dropdown:: LIBERO-PRO


   .. list-table:: 参考方法
      :header-rows: 1
      :widths: 24 36 20 20

      * - 方法
        - 覆盖范围
        - 成功率
        - 来源
      * - OpenVLA
        - 八个 Task/Swap 单元
        - 0.0%
        - :ref:`表 3 <benchmark-source-p3>`
      * - π0
        - 八个 Task/Swap 单元
        - 0.3%
        - :ref:`表 3 <benchmark-source-p3>`
      * - π0.5
        - 八个 Task/Swap 单元
        - 11.0%
        - :ref:`表 3 <benchmark-source-p3>`
      * - MolmoAct
        - 八个 Task/Swap 单元
        - 1.5%
        - :ref:`表 3 <benchmark-source-p3>`
      * - NORA
        - 八个 Task/Swap 单元
        - 0.0%
        - :ref:`表 3 <benchmark-source-p3>`
      * - X-VLA
        - 八个 Task/Swap 单元
        - 3.8%
        - :ref:`表 3 <benchmark-source-p3>`
      * - AtomVLA
        - 八个 Task/Swap 单元
        - 6.3%
        - :ref:`表 3 <benchmark-source-p3>`
      * - π_RLinf
        - 八个 Task/Swap 单元
        - 50.0%
        - :ref:`表 3 <benchmark-source-p3>`
      * - Cap-X
        - 六个非 Long 单元
        - 18.2%
        - :ref:`表 3 <benchmark-source-p3>`
      * - RATS
        - 六个非 Long 单元
        - 43.8%
        - :ref:`表 3 <benchmark-source-p3>`


   Cap-X 和 RATS 仅涵盖 Spatial、Object、Goal 的 Task/Swap 六个单元，不与涵盖八个单元的 Overall 直接排名。

.. dropdown:: RoboCasa365 Target50


   .. list-table:: 参考方法
      :header-rows: 1
      :widths: 24 36 20 20

      * - 方法
        - 覆盖范围
        - 成功率
        - 来源
      * - RLDX-1
        - 全部三个划分，任务均权
        - 30.0%
        - :ref:`表 4 <benchmark-source-p4>`
      * - WorldDreamer
        - 全部三个划分，任务均权
        - 35.3%
        - :ref:`表 4 <benchmark-source-p4>`
      * - π0.5
        - 全部三个划分，任务均权
        - 16.9%
        - :ref:`表 4 <benchmark-source-p4>`
      * - π0
        - 全部三个划分，任务均权
        - 14.8%
        - :ref:`表 4 <benchmark-source-p4>`


   RLDX-1 是冻结 VLA 的直接评测基线；其余方法为外部报告。Overall 保留来源报告的聚合口径。

.. dropdown:: RoboTwin C2R


   .. list-table:: 参考方法
      :header-rows: 1
      :widths: 24 36 20 20

      * - 方法
        - 覆盖范围
        - 成功率
        - 来源
      * - GR00T-N1.7
        - C2R
        - 20.7%
        - :ref:`表 6 <benchmark-source-p6>`
      * - π0.5
        - C2R
        - 47.9%
        - :ref:`表 6 <benchmark-source-p6>`
      * - StarVLA
        - C2R
        - 10.6%
        - :ref:`表 6 <benchmark-source-p6>`
      * - LingBot-VLA
        - C2R
        - 50.4%
        - :ref:`表 6 <benchmark-source-p6>`


   LingBot-VLA 既是 RPent 的冻结接触策略后端，也是直接评测基线；不把 RPent 的 250 回合样本量套用到外部报告。

.. dropdown:: LIBERO-PRO Goal 零样本


   .. list-table:: 参考方法
      :header-rows: 1
      :widths: 24 36 20 20

      * - 方法
        - 覆盖范围
        - 成功率
        - 来源
      * - Cap-X
        - Goal Task
        - 16.8%
        - :ref:`表 5 <benchmark-source-p5>`
      * - Cap-X
        - Goal Swap
        - 25.6%
        - :ref:`表 5 <benchmark-source-p5>`


   这些 Goal Task/Swap 数值对应不使用目标设置记忆的消融。

.. _id4:

指标、协议与来源
------------------------

结果快照更新于 2026-09-14。RPent 的 GPT-5.5 与 Opus-4.8 已报告成绩与
Harness VLA 论文 v4（2026-09-02）对齐；GPT-6 Astra 为新增模型评测结果。
论文使用 Codex 和 CC（Claude Code）标记后端，精确模型与推理设置由贡献者提供。

任务成功依据基准判定条件：LIBERO 环境轨迹中的 ``terminated``、RoboCasa 的
``state.success``，或 RoboTwin 的 ``TASK_ENV.eval_success``。规划器调用 ``finish`` 或原语
返回局部成功，本身不构成任务成功标签。用于构建记忆的探索回合不计入评测成绩。

LIBERO 系列使用冻结的 RLinf π0.5 full-shot LIBERO 检查点；RoboCasa 使用冻结的
RLDX-1；RoboTwin 使用后训练后冻结的 LingBot-VLA 检查点。规划模型、VLA 后端、
记忆库与评测协议分别描述系统的不同组成部分，成绩差异不等同于只改变规划模型的受控实验。

.. _benchmark-source-p2:

**Standard LIBERO。** `Harness VLA，表 2 <https://arxiv.org/html/2607.08448v4#S3.T2>`_。
四个标准套件，每个套件 100 回合，Overall 共 400 回合。

.. _benchmark-source-r2:
.. _benchmark-source-p3:

**LIBERO-PRO。** `Harness VLA，表 3 <https://arxiv.org/html/2607.08448v4#S3.T3>`_。
八个 Task/Swap 单元，每个单元 10 个任务、每个任务 10 个评测 seed，共 100 回合；Overall 共 800 回合。seed 0 用于构建记忆。Long Task 对应 ``libero_10_task``，Long Swap 对应 ``libero_10_swap``。

.. _benchmark-source-r3:
.. _benchmark-source-p4:

**RoboCasa365 Target50。** `Harness VLA，表 4 <https://arxiv.org/html/2607.08448v4#S3.T4>`_。
Atomic-Seen 有 18 个任务、每任务 10 个 seed；Composite-Seen 和 Composite-Unseen 各有 16 个任务、每任务 5 个 seed，分别为 180、80、80 回合。Seen/Unseen 表示预训练中的任务模板覆盖情况。Overall 对 50 个任务等权平均。百分比保留来源精度，不从四舍五入后的比例反推成功次数。

.. _benchmark-source-p5:

**LIBERO-PRO Goal 零样本。** `Harness VLA，表 5 <https://arxiv.org/html/2607.08448v4#S3.T5>`_。
Goal Task 和 Goal Swap 各有 10 个任务、每任务 10 个 seed，共 100 回合；不使用目标设置的 Task Specific Memory 和 Global Memory。

.. _benchmark-source-r4:
.. _benchmark-source-p6:

**RoboTwin C2R。** `Harness VLA，表 6 <https://arxiv.org/html/2607.08448v4#S3.T6>`_。
50 个任务，每个任务 5 个经专家验证的官方随机 seed，共 250 回合。任务记忆来自已验证的 ``demo_clean`` 实例，并迁移到 ``demo_randomized``，不在随机设置中探索。

.. _benchmark-source-r1:
.. _benchmark-protocol-r1:

**GPT-6 Astra 评测。** 实验 ``libero_long_gpt6_astra_20260907`` 于 2026-09-07 开始，
使用运行时版本 `014a0fa <https://github.com/RLinf/RPent/commit/014a0fa97f69c991ee5e0f62f14e1d6f89c3dcd7>`_。
Long Task 与 Long Swap 各有 10 个任务；任务记忆在 seed 0 上独立构建并冻结，
随后对每个任务使用 seed 1–10 评测。使用本地记忆库，规划器时间上限为 5000 秒，
环境步数上限为 10000 步；表中记录环境成功率。

新增模型或成绩时，保持同一评测项的行与模型列顺序，记录后端、模型、推理设置、
样本规模及协议；更新来源说明，并将没有结果的单元保持为“未报告”。

.. astra-supplementary-source-begin

.. _benchmark-source-astra-pro:

**GPT-6 Astra 补充 LIBERO-PRO 评测。** 实验
``libero_pro_remaining_gpt6_astra_20260913`` 使用运行版本
`014a0fa <https://github.com/RLinf/RPent/commit/014a0fa97f69c991ee5e0f62f14e1d6f89c3dcd7>`_。Spatial、Object、Goal 分别评测 Task 和 Swap；
每个已报告分项覆盖 10 个任务，各使用 seed 1–10，共 100 回合。
各任务在 seed 0 独立构建记忆，经合并并冻结后开始评测。

GPT-6 Astra 的完整 800 回合 Overall 仅在八个 Task/Swap 分项全部完成后报告。
完整八项范围由 Long 200 回合与补充评测 600 回合组成；各已报告批次使用
对应的预先冻结记忆库，这些批次并非使用同一份记忆快照。

**2026-09-14 14:00:55 UTC** 核验快照中，已完成 619 回合。
Goal Task 已完成 19 回合（10 成功、9 失败），Goal Swap 暂无已完成回合。
这些数字仅表示进度，不是完整套件或 Overall 成功率。
:doc:`逐任务与 seed 结果表 <results/libero_pro_astra>` 保留全部 800 个计划位置，
其中 181 个待完成；不包含服务器路径或原始运行日志。

.. astra-supplementary-source-end


.. _benchmark-demo:

演示视频
--------

**RPent 仿真演示：GPT-6 Astra 与 GPT-5.6 xhigh，4 倍速播放。**
此视频为独立演示，与 GPT-5.5 统计记录分开说明。

.. image:: https://raw.githubusercontent.com/RLinf/misc/1f93c83cbbef749ce55af0399b1bd405e8e9c7a7/rpent/demo/demo-poster.jpg
   :alt: RPent simulation demo
   :width: 100%
   :target: https://raw.githubusercontent.com/RLinf/misc/1f93c83cbbef749ce55af0399b1bd405e8e9c7a7/rpent/demo/demo.mp4

查看或下载`完整 MP4（约 24 秒，5.4 MiB） <https://raw.githubusercontent.com/RLinf/misc/1f93c83cbbef749ce55af0399b1bd405e8e9c7a7/rpent/demo/demo.mp4>`_。

“未报告”表示没有对应成绩，不能视为零。

结果维护
--------

图片和演示媒体统一存放在 `RLinf/misc <https://github.com/RLinf/misc>`_ 的
``rpent/`` 目录。本页引用不可变媒体 commit
``1f93c83cbbef749ce55af0399b1bd405e8e9c7a7``；配套的
`结果快照 <https://raw.githubusercontent.com/RLinf/misc/1f93c83cbbef749ce55af0399b1bd405e8e9c7a7/rpent/benchmarks/results.json>`_ 记录数据来源。
构建文档不需要绘图工具或 benchmark 专用 JavaScript。

更新成绩时同步修改中英文表格，保留评测范围与来源精度，并将对应图片提交到 misc。
确认图片与表格一致后再更新媒体 commit。不要从四舍五入的百分比推算成功次数，
也不要把缺失结果当作零。保持模型列顺序、RoboCasa 任务加权口径，以及完整
LIBERO-PRO 与六项或零样本比较的区别。维护说明集中在本文档页面，不再增加
代码目录中的独立 README。

.. toctree::
   :hidden:

   results/libero_pro_astra
