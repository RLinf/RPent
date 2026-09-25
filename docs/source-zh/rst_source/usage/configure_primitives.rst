动作原语与工具
=====================

规划器通过工具调用执行操作。动作原语负责完成具体的运动，例如抓取、移动末端或打开夹爪；感知和状态工具用于定位目标、读取图像及检查结果。

选择动作工具
------------------

RPent 的动作工具主要分为两类：

- **VLA 动作**：由视觉语言动作模型生成动作序列，例如 LIBERO 的 ``pi0_pick``。模型通常在独立服务中运行。
- **程序化动作**：按参数执行运动，例如 ``move_to``、``rotate_wrist`` 和 ``release``。具体名称与参数由环境定义。

``back_project``、``segment`` 和 ``view_env_state`` 属于感知或状态读取工具，本身不代表机器人运动。``finish`` 用于结束规划器循环，任务是否成功仍由环境或真机操作员判定。

各平台使用的模型
------------------------

.. list-table::
   :header-rows: 1

   * - 环境 / 机器人
     - 动作模型与配置
   * - :doc:`libero`
     - Pi0.5
   * - :doc:`robocasa`
     - RLDX-1
   * - :doc:`robotwin`
     - LingBot-VLA
   * - :doc:`franka`
     - Pi0.5 / openpi
   * - :doc:`dual_franka`
     - Pi0.5 / openpi

各平台页面给出模型路径、工具范围和任务要求。YAM 的部署说明和 SO-101 内容尚待补充，见 :doc:`yam` 和 :doc:`so101`。

服务与扩展
---------------

VLA 服务通过 ``predict`` 提供动作预测，通过 ``healthz`` 提供健康检查。RPent 的 RPC 服务支持 HTTP 和 socket 传输，具体部署方式见 :doc:`advanced_deployment`。

添加工具时，需要同时定义参数、执行逻辑和返回结果，详见 :doc:`../development/add_primitive`。规划器使用同一个 toolkit 调用这些工具。
