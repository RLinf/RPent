# LIBERO 几何定位评估

评估入口使用固定的 16 项任务、seed 1～5，共 80 组场景、160 个参考点。
每组在初始帧分别定位 A、B 两个目标。任务和部位定义由 `localization_checklist.py` 固定，
失败或遮挡时仍保留原请求并记录原因。表中的每一行都展开为五组固定 seed。

| 任务集 | 任务编号 | Seed | 目标 A | 目标 B |
| --- | --- | --- | --- | --- |
| `libero_spatial_task` | 0 | 1～5 | 盘面中心 | 小烤碗碗口中心 |
| `libero_spatial_task` | 4 | 1～5 | 盘面中心 | 小烤碗碗口中心 |
| `libero_spatial_swap` | 0 | 1～5 | 盘面中心 | 小烤碗碗口中心 |
| `libero_spatial_swap` | 4 | 1～5 | 盘面中心 | 小烤碗碗口中心 |
| `libero_object_task` | 0 | 1～5 | 奶酪盒顶面中心 | 汤罐罐顶中心 |
| `libero_object_task` | 4 | 1～5 | 奶酪盒顶面中心 | 汤罐罐顶中心 |
| `libero_object_swap` | 0 | 1～5 | 奶酪盒顶面中心 | 汤罐罐顶中心 |
| `libero_object_swap` | 4 | 1～5 | 奶酪盒顶面中心 | 汤罐罐顶中心 |
| `libero_goal_task` | 0 | 1～5 | 盘面中心 | 黑碗碗口中心 |
| `libero_goal_task` | 7 | 1～5 | 盘面中心 | 黑碗碗口中心 |
| `libero_goal_swap` | 0 | 1～5 | 盘面中心 | 黑碗碗口中心 |
| `libero_goal_swap` | 7 | 1～5 | 盘面中心 | 黑碗碗口中心 |
| `libero_10_task` | 3 | 1～5 | 黑碗碗口中心 | 酒瓶瓶身主段中心 |
| `libero_10_task` | 9 | 1～5 | 灰白纹理杯杯口中心 | 黄白杯杯口中心 |
| `libero_10_swap` | 3 | 1～5 | 黑碗碗口中心 | 酒瓶瓶身主段中心 |
| `libero_10_swap` | 9 | 1～5 | 黄白杯杯口中心 | 微波炉机身顶面中心 |

## 清单与入口

默认使用内置清单，检查入口无需仿真或模型服务：

```bash
python -m toolkits.geometry.evaluation.localization_experiment validate-manifest
```

若需保存独立清单供实验记录或人工检查，显式导出并读取：

```bash
python -m toolkits.geometry.evaluation.localization_experiment write-manifest \
  --manifest experiments/localization/test-cases.json
python -m toolkits.geometry.evaluation.localization_experiment validate-manifest \
  --manifest experiments/localization/test-cases.json
```

传入的清单必须与固定协议完全一致；缺失或修改过的文件会报错。
`collect` 需要 LIBERO-Pro 资源和渲染环境，`infer` 需要 SAM3 和视觉模型，
`evaluate` 使用独立真值计算误差。完整操作步骤、输出文件和统计口径见
[定位实验指南](../../../robots/libero/guides/localization_experiment.md)。

离线单元测试位于 `tests/unit_tests/toolkits/geometry/evaluation/`，
使用合成几何和替代服务，检查坐标变换、任务分派、输入隔离、资源释放和误差统计。
