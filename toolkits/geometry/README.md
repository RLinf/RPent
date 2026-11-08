# 几何定位工具

GeometryAgent、点云清理和形状拟合位于 `toolkits/geometry/`，固定清单实验及独立评估位于 `evaluation/`。
LIBERO 的环境、动作原语和工具注册位于 `robots/libero/`，通过直接导入调用这里的实现。
共享的视觉模型适配位于 `rpent/robots/components/grounding_agent.py`。

| 文件 | 职责 |
| --- | --- |
| [geometry_agent.py](geometry_agent.py) | 接收目标和部位描述，协调视觉选点、分割、反投影与几何计算 |
| [geometry.py](geometry.py) | 点云清理、平面、圆、圆柱和圆台拟合，以及验收参数 |
| [reference_geometry.py](reference_geometry.py) | 内口沿、盘面、封口盖、矩形顶面和瓶身主段的候选提取与中心拟合 |
| [candidate_geometry.py](candidate_geometry.py) | 宽松评估模式：严格质量检查未通过时，从当前观测估计候选中心并保留近似方法 |
| [evaluation/localization_checklist.py](evaluation/localization_checklist.py) | 固定清单及 80 组场景、160 个参考点的展开与校验 |
| [evaluation/localization_experiment.py](evaluation/localization_experiment.py) | 初始帧采集、定位推理及评估命令 |
| [evaluation/localization_evaluation.py](evaluation/localization_evaluation.py) | 独立参考点校验、坐标误差和返回率统计 |
| [evaluation/evaluation_scene.py](evaluation/evaluation_scene.py) | 评估专用的完整模型网格和同帧位姿快照 |

LIBERO 工具集直接导入定位类：

```python
from toolkits.geometry.geometry_agent import GeometryAgent
```

检查评估入口：

```bash
python -m toolkits.geometry.evaluation.localization_experiment --help
python -m toolkits.geometry.evaluation.localization_experiment validate-manifest
```

固定任务表见 [评估说明](evaluation/README.md)。默认使用包内定义的清单；需要独立 JSON 文件时用 `write-manifest --manifest <path>` 导出，后续通过 `--manifest <path>` 显式读取并校验。

实验推理添加 `--permissive` 可比较严格拟合拒绝点的坐标误差。
候选中心带有 `strict_accepted=false`、`estimate_method` 和原拒绝原因，
仅用于误差评估；有效观测不足仍记录错误。默认使用严格拟合。

离线单元测试位于 `tests/unit_tests/toolkits/geometry/`，评估模块的单元测试位于其 `evaluation/` 子目录。
使用方式见 [GeometryAgent 指南](../../robots/libero/guides/geometry_agent.md)和
[定位实验指南](../../robots/libero/guides/localization_experiment.md)。

旧实验分支的 `tools.libero` 和 `robots.libero` 中转模块已移除。已有脚本需更新导入路径及 `python -m` 命令。
