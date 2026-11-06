# 项目工具

LIBERO 的 GeometryAgent 和定位实验实现集中在 `tools/libero/`。
机器人环境、动作原语和工具注册位于 `robots/libero/`，由工具入口调用这里的实现。
共享的视觉模型适配仍位于 `rpent/robots/components/grounding_agent.py`。

| 文件 | 职责 |
| --- | --- |
| [geometry_agent.py](libero/geometry_agent.py) | 接收目标和部位描述，协调视觉选点、分割、反投影与几何计算 |
| [geometry.py](libero/geometry.py) | 点云清理、平面、圆、圆柱和圆台拟合，以及验收参数 |
| [reference_geometry.py](libero/reference_geometry.py) | 内口沿、盘面、封口盖、矩形顶面和瓶身主段的候选提取与中心拟合 |
| [candidate_geometry.py](libero/candidate_geometry.py) | 宽松评估模式：质量检查未通过时，从当前观测估计候选中心并保留近似方法 |
| [localization_checklist.py](libero/localization_checklist.py) | 固定清单及 80 组场景、160 个参考点的展开与校验 |
| [localization_experiment.py](libero/localization_experiment.py) | 初始帧采集、定位推理及评估命令 |
| [localization_evaluation.py](libero/localization_evaluation.py) | 独立参考点校验、坐标误差和返回率统计 |
| [evaluation_scene.py](libero/evaluation_scene.py) | 评估专用的完整模型网格和同帧位姿快照 |

从项目根目录检查实验入口：

```bash
python -m tools.libero.localization_experiment --help
python -m tools.libero.localization_experiment validate-manifest
```

实验推理命令添加 `--permissive` 可比较被严格拟合拒绝的点的坐标误差。
候选中心带有 `strict_accepted=false`、`estimate_method` 和原拒绝原因，
仅用于误差评估；有效观测不足仍记录错误。默认使用严格拟合。

对应测试位于 `tests/unit_tests/tools/libero/`。
使用方式见 [GeometryAgent 指南](../robots/libero/guides/geometry_agent.md)和
[定位实验指南](../robots/libero/guides/localization_experiment.md)。
原 `robots.libero` 下这七个模块的导入路径和定位实验命令转接到同一实现，历史脚本可继续使用。
