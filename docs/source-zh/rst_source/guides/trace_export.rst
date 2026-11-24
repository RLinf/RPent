.. _code-policy-trace-export:

代码策略轨迹导出
================

``tool-trace_export`` 用于离线校验已记录的代码策略轨迹，并导出经过审计且与具体模型无关的数据集行（``rows.jsonl``）、按内容寻址保存的 PNG 图像以及导出审计报告（``export-audit.json``）。该工具不负责渲染提示词。

.. note::

   RPent 目前尚未提供写出 ``rpent.code_policy_trace.v1`` 格式的在线轨迹记录器。该导出器当前用于读取外部生成或后续代码策略记录钩子写出的该格式轨迹目录。

安装
----

``tool-trace_export`` 作为独立组件安装，仅包含核心 ``rpent`` 与 ``pillow>=10``，不引入机器人、仿真器或模型服务依赖：

.. code-block:: bash

   rpent-install install tool-trace_export

在尚未将 ``rpent-install`` 放入 ``PATH`` 的源码仓库中，可运行：

.. code-block:: bash

   ./rpent-install.sh install tool-trace_export --venv /path/to/venv

输入轨迹格式
------------

每条轨迹目录包含 ``episode.json``、``turns.jsonl`` 和 ``images/`` 三个部分：

.. code-block:: text

   /path/to/traces/<episode_id>/
   ├── episode.json
   ├── turns.jsonl
   └── images/
       └── <image_file>.png

``episode.json`` 必须包含下表前五个必填键，并可额外包含可选键 ``robo_methods``；``turns.jsonl`` 中的每个 JSON 对象必须与下表列出的键集合完全一致。缺少必填字段、多出未知字段或出现 ``NaN``、``Infinity`` 等非有限数值时，整条轨迹都会被拒收。

.. list-table:: ``episode.json`` 记录轨迹标识、可用方法与运行判定
   :header-rows: 1

   * - 字段
     - 含义与约束
   * - ``schema``
     - 必须为 ``rpent.code_policy_trace.v1``。
   * - ``episode_id``
     - 本条轨迹的非空唯一标识。
   * - ``robot``
     - 非空机器人标识。
   * - ``model``
     - 非空模型标识。
   * - ``verdict``
     - 对象，键必须恰好为 ``accepted``、``truth_source`` 和 ``stop_reason``。
   * - ``robo_methods``
     - 可选；非空列表，声明记录代码允许调用的公开 ``robo.<method>`` 方法名（不得重复或包含私有、保留字及非法标识符）。未提供时需通过 ``--robo-method`` 兜底指定。

.. list-table:: ``turns.jsonl`` 每行按序记录一个规划轮次
   :header-rows: 1

   * - 字段
     - 含义与约束
   * - ``turn``
     - 从 1 开始的连续整数，须与所在轮次序号一致。
   * - ``system``
     - 规划器看到的非空系统提示词。
   * - ``user``
     - 非空片段列表，元素为 ``{"type": "text", "text": "..."}`` 或 ``{"type": "image", "image": <index>}``，须按索引恰好覆盖 ``images`` 中的每张图像一次。
   * - ``images``
     - 本轮次引用的图像记录列表。
   * - ``code``
     - 模型提交的非空 Python 代码单元。
   * - ``executed_code``
     - 实际执行的 Python 代码，必须与 ``code`` 完全一致。
   * - ``accepted``
     - 必须为 ``true``。
   * - ``interrupted``、``timed_out``、``namespace_reset``
     - 这三项状态标志必须全部为 ``false``。

.. list-table:: ``turns.jsonl`` 中 ``images`` 列表的每一项
   :header-rows: 1

   * - 字段
     - 含义与约束
   * - ``path``
     - 相对于 ``images/`` 的非空路径，不能逃逸出 ``images/`` 目录。
   * - ``png_sha256``
     - PNG 文件原始字节的 SHA-256 十六进制摘要。
   * - ``rgb_sha256``
     - 解码后 RGB 像素字节的 SHA-256 十六进制摘要。
   * - ``width``、``height``
     - 正整数，必须与解码后的 RGB PNG 图像尺寸一致。

运行导出
--------

在已安装 ``tool-trace_export`` 的环境中，直接传入包含多条轨迹的父目录（或一个或多个单条轨迹目录）以及新的 ``-o`` / ``--output`` 输出目录：

.. code-block:: bash

   python -m components.tools.trace_export /path/to/traces -o /path/to/trace-dataset

若输入轨迹的 ``episode.json`` 未记录 ``robo_methods``，可通过 ``--robo-method`` 兜底指定（支持逗号分隔或重复传入）；也可以直接列出多个轨迹路径：

.. code-block:: bash

   python -m components.tools.trace_export \
     /path/to/traces/episode_001 \
     /path/to/traces/episode_002 \
     -o /path/to/trace-dataset \
     --robo-method move_pose,set_gripper,finish

- ``SOURCE``：一个或多个轨迹目录或上级父目录，程序会自动递归查找其中的 trace v1 轨迹目录（必填）。
- ``-o``、``--output``：新建的输出目录，不能位于任何 ``SOURCE`` 目录内，且运行前不能已存在（必填）。
- ``--robo-method``：可选兜底参数；当 ``episode.json`` 未记录 ``robo_methods`` 时，指定允许调用的公开 ``robo`` 方法名，支持逗号分隔或重复传参。若 ``episode.json`` 已提供 ``robo_methods``，优先以轨迹自身记录为准。
- ``--include-unsuccessful``：可选开关；同时导出未获环境确认成功的轨迹，并在输出行中标记 ``environment_success: false``，而非直接隔离。

导出器在写出结果前会执行严格拒收校验与 AST 投影：

- 轨迹原子性：任一轮次校验失败，整条轨迹都会被隔离，不会写出该轨迹的任何数据行或图像；重复的 ``episode_id`` 也会被隔离。
- 成功判定：默认情况下仅导出 ``verdict.accepted`` 为 ``true`` 且 ``verdict.truth_source`` 为 ``"environment"`` 的轨迹（标记为 ``environment_success: true``）；其余轨迹默认一律隔离，仅当传入 ``--include-unsuccessful`` 时导出（标记为 ``environment_success: false``）。
- AST 注解剥离（``robo-explanation-strip.v1``）：剥离未别名的公开 ``robo.<method>`` 调用上的 ``explanation`` 关键字参数（1–800 个字符的字符串字面量、未重绑定的转发形参，以及保留求值副作用的 f-string 注解）和 ``robo.parallel`` 子动作字典中的字面量 ``explanation`` 键。若代码中存在 ``robo`` 遮蔽或别名、未列出的 ``robo`` 方法、未知 ``**kwargs``、动态调用（``exec``、``eval``、``globals``、``locals``、``compile``、``__import__``）或下标读取 ``["explanation"]``，则整条轨迹拒收。
- 持久命名空间检查：按轮次检查所有被引用的名称是否已在前置轮次、当前轮次、Python 内置名称或注入变量 ``robo`` 中定义。
- 退出码：仅当至少导出 1 行数据且 ``quarantined_episodes`` 为空时返回 ``0``；若发生参数或目录错误、导出行数为 0 或存在任何被隔离轨迹，均返回 ``1``。

输出产物
--------

导出完成后，``--output`` 目录下会生成 ``rows.jsonl``、``export-audit.json`` 以及按内容去重保存的 PNG 图像：

.. code-block:: text

   /path/to/trace-dataset/
   ├── rows.jsonl
   ├── export-audit.json
   └── images/
       └── <png_sha256>.png

- ``rows.jsonl``：每个轮次对应一行 ``rpent.trace_export.v1`` JSON 对象，包含 ``schema``、``id`` （格式为 ``<episode_id>:turn_NNNNNN``）、``episode_id``、``split_group``、``environment_success``、``messages`` （含 ``system`` 以及路径已改写为 ``images/<png_sha256>.png`` 的 ``user``）、``target`` （调用 ``python`` 工具执行投影后 ``code`` 的 assistant 目标）、``images``、``source`` （含 ``episode_dir``、``robot``、``model``、``turn`` 和 ``original_code``）以及 ``audit`` （含投影统计与校验和、``namespace_prefix_cells`` 和 ``native_verdict``）。
- ``images/<png_sha256>.png``：按内容寻址且去重的 PNG 图像；仅当至少导出一张图像时创建该目录，若同名目标文件字节不一致则拒收。
- ``export-audit.json``：记录本次导出的统计与审计信息，包含 ``schema``、``trace_schema``、``projection``、``accepted_episodes``、``rows``、``successful_rows``、``unsuccessful_rows``、``quarantined_episodes``、``rows_sha256`` 以及 ``split_policy`` （固定为 ``"episode-level only; no independent heldout implied"``）。
