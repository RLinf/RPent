.. 仿真环境指南模板：复制本文件，替换所有 <...>，并删除这些注释。
.. 规则见 .agents/skills/docs/rules/simulator.md、rules/chinese.md 和 docs/STYLE_GUIDE.md。

<仿真环境名称>
==============

.. 可选概览图（不需要时删除）：
.. .. figure:: https://raw.githubusercontent.com/<org>/<repo>/<branch>/<path>
..    :alt: <仿真环境名称> 概览
..    :width: 90%
..    :align: center
..
..    <图示说明>。图片来源：`<项目名> <<URL>>`_。

使用 RPent 在 `<仿真环境名称> <<项目地址>>`_ 中运行<任务类型>任务。<一句话说明仿真后端、支持版本与默认动作模型。>

概览
----

先确认动作模型、规划器、任务范围与硬件要求，再按后续步骤安装和运行。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 动作模型

      <默认动作模型及可选模型。>

   .. grid-item-card:: 规划器

      ``api``、``claude_code``、``codex``、``gemini_cli``

   .. grid-item-card:: 任务

      <任务套件或任务类别。>

   .. grid-item-card:: 硬件

      Linux、NVIDIA GPU；Python 3.11；CUDA 与 EGL。

任务
~~~~

<一句话概述支持的任务套件或类别。>

.. list-table::
   :header-rows: 1

   * - 套件 / 类别
     - 范围
     - 说明
   * - ``<suite_name>``
     - <任务数量或范围>
     - <一句话说明>

观测与动作
~~~~~~~~~~

下表区分规划器工具、动作模型输入与环境原生成功判定。

.. list-table::
   :header-rows: 1

   * - 项目
     - 说明
   * - 观测
     - <规划器与动作模型使用的相机图像、深度或机器人状态。>
   * - 动作
     - <规划器可调用的工具与动作原语，以及它们执行的环境动作。>
   * - 奖励与成功判定
     - <环境原生成功判定（如最终状态的 ``terminated``）。达到步数上限或规划器调用 ``finish`` 不代表成功。>
   * - 任务指令
     - <任务语言指令的来源。>

安装与资源准备
--------------

在独立虚拟环境中安装 ``robot-<name>`` 组件：

.. code-block:: bash

   ./rpent-install.sh install robot-<name> --venv .venv-<name>
   source .venv-<name>/bin/activate

下载所需的仿真资源和动作模型权重：

.. code-block:: bash

   <资源与权重下载命令>

运行一个任务
------------

按 :doc:`../guides/configure_planner` 配置规划器后，运行单个任务：

.. code-block:: bash

   rpent --robot <robot_id> --planner <planner> <关键参数>

常用可调参数：

- ``<flag>``：<作用与默认值>。

查看结果
--------

每次运行会在 ``--output-dir`` 下保存每步观测、工具调用记录和最终状态。查看 ``<结果文件及原生成功字段>`` 确认任务是否成功。
