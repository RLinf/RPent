# LIBERO 几何评估

本入口执行全部 16 项固定任务、seed 1～5，共 80 组、160 个参考点。
任务由包内的 `checklist.py` 展开，固定目标、参考点定义、step 0、agentview 和米制坐标。
每组分别定位 A、B；遮挡或拟合失败不会替换目标。

评估实现位于 `toolkits/geometry/evaluation/`，几何定位和拟合位于 `toolkits/geometry/`。
几何接口和拟合约定见[工具说明](../README.md)。

## 固定任务

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

## 准备与运行

在仓库根目录、已安装 LIBERO-Pro 的环境中运行。采集需要已安装的仿真资源和可工作的 EGL/CUDA 环境；推理需要 SAM3 checkpoint 和可用的视觉模型。沿用 LIBERO 使用文档的安装及 `SAM3_CHECKPOINT_PATH` 配置。实验不需要 Pi0.5/WAM 权重、任务记忆或主规划器。

```bash
python -m toolkits.geometry.evaluation.experiment validate-manifest

python -m toolkits.geometry.evaluation.experiment collect \
  --output experiments/geometry/observations

python -m toolkits.geometry.evaluation.experiment infer \
  --inputs experiments/geometry/observations \
  --geometry-model codex \
  --output experiments/geometry/predictions
```

这些命令默认使用内置清单。需要保存独立 JSON 时，显式导出：

```bash
python -m toolkits.geometry.evaluation.experiment write-manifest \
  --manifest experiments/geometry/test-cases.json
```

后续命令可添加 `--manifest experiments/geometry/test-cases.json` 读取并严格校验该副本。
显式指定的文件缺失或清单内容被修改时会报错。

`codex` 使用已配置的默认模型，也可写成 `codex:model` 或 API 的 `provider:model`；后者通过供应商规定的环境变量提供凭据，可用 `--geometry-base-url` 指定兼容服务地址。模型必须支持图像输入与结构化选点。这里的 `--geometry-model` 必填，不会启动一个主规划器来继承模型。

`collect` 对每组启动独立 ENV，固定 suite/task/seed，记录初始观测后释放自己启动的进程。它不启动分割或动作模型。高分辨率记录来自原生 1024×1024 配对 RGB-D，保留米制深度、相机元数据和 float32 世界坐标图；不会放大低分辨率深度。标定由环境内部获取，使用者不传标定参数。常规机器人历史中的 float16 坐标图仍保持原有格式。

`infer` 只启动一个 SAM3 服务，并在采集结果的副本上调用 GeometryAgent。视觉模型只看 RGB 和目标描述；候选掩码必须包含独立视觉选点。每个清单请求最多 1 次视觉选点、2 次 SAM、1 次材质点反投影；拟合和清理在本地进行。实验不执行动作、补视角或同义词重试。若服务或观测失败，仍记录对应请求，继续处理其余目标。

输出目录必须为空或尚不存在。已有观测可以反复用于不同模型的推理，每次指定新的推理输出目录。推理会核对观测身份和文件哈希，并只复制允许的相机观测文件，不复制原始 state、extras 或真值标注。

## 先运行少量样本

采集和推理都支持 `--limit N` 或重复传入 `--case <case_id>`。两阶段使用相同选择；省略选择时默认全部 80 组。

```bash
python -m toolkits.geometry.evaluation.experiment collect \
  --case libero_10_swap_t9_s1 \
  --output experiments/geometry/pilot-observations

python -m toolkits.geometry.evaluation.experiment infer \
  --case libero_10_swap_t9_s1 \
  --inputs experiments/geometry/pilot-observations \
  --geometry-model codex \
  --output experiments/geometry/pilot-predictions
```

通过 `--cuda-device` 或分别设置 `--env-cuda-device`、`--sam3-cuda-device` 选择 GPU。推理可用 `--sam3-endpoint http://host:port` 借用现有 SAM3；采集的 `--env-endpoint` 只允许同时选择一组，且服务的 suite/task/seed/max_episode_steps 必须匹配。借用服务不会由实验入口关闭。

## 固定参考点及输出字段

清单对象使用以下拟合量，定义与清单一致：

| 对象 | 数量 | `required_geometry` | 几何定义 |
|---|---:|---|---|
| 盘子 | 40 | `plate_center` | 中央承托圆面圆心，z 在承托面上 |
| 小烤碗 | 20 | `inner_rim_center` | 内口上沿的三维圆心 |
| 奶酪盒 | 20 | `rectangle_center` | 主要矩形顶面的几何中心 |
| 汤罐 | 20 | `lid_center` | 封口圆盖的三维圆心，随盖面倾斜 |
| 黑碗 | 30 | `inner_rim_center` | 内口上沿的三维圆心 |
| 酒瓶 | 10 | `body_segment_center` | 排除瓶颈、瓶嘴后主瓶身段的轴向中点 |
| 灰白纹理杯 | 5 | `inner_rim_center` | 不含把手的内口上沿圆心 |
| 黄白杯 | 10 | `inner_rim_center` | 不含把手的内口上沿圆心 |
| 微波炉 | 5 | `rectangle_center` | 机身外壳完整矩形顶面中心 |

内口拟合不会用洞中心像素的深度，也不会在无法区分内缘时返回外口圆心。RGB 轮廓方案依赖视觉模型正确标识内口材质点，结果仍是有形状假设的推断。盘面和罐盖先确定目标面，再从实际面边界拟合圆心；顶面的四条物理边和角点邻域必须有支撑，单纯标签裁剪、前景遮挡或图像截断会被拒绝。酒瓶使用瓶身附近的圆柱假设，并要求肩部半径过渡或真实端缘来确定两个端部；可见点云长度不能代替主瓶身范围。

这是对所有清单定义的算法及执行覆盖，不承诺单视角中的全部 160 点都能测出。`usable` 表示通过当前拟合检查，`need_observation` 表示证据不足或拟合被拒绝，`error` 表示观测、服务或保存出错。此固定协议只记录补视角请求，不执行移动。定位误差要由独立真值确定，SAM score、拟合残差和算法阈值都不是测量精度。

## 宽松误差评估

仅为比较坐标误差时，在 `infer` 命令添加 `--permissive`。默认仍使用严格模式。
该模式覆盖清单中的五种中心字段：先运行原有检查，未通过时再尝试宽松拟合或可见范围中心估计，
不因拟合质量不足返回补视角请求。缺失有效观测、模型服务失败或保存失败仍记录错误。

宽松拟合保留 3 mm 残差阈值，将最少点数降为 6、内点比例降为 10%、圆和瓶身角覆盖降为 1°、
相对残差上限改为 99%、条件数上限改为 1e12；内口圆拟合不再要求圆经过视觉材质点。
盘面或盖面缺少经验证的物理边界时尝试可见平面掩码轮廓；矩形中心来自可见面最小面积矩形，
瓶身使用拟合轴的可见跨度中点，拟合仍失败时使用对应可见点的范围中心。
这些近似可能受遮挡、错误轮廓、瓶颈或不完整顶面影响，需要通过独立真值检查误差。

`geometry[field]` 和 `results.jsonl` 中的 `strict_accepted`、`estimate_method` 标明是否通过原检查及实际方法。
候选字段另有 `evaluation_only=true` 和 `strict_rejection`，原拟合结果仍保存在 `parts[part].fit`，
候选拟合点另存于 `candidate_fit_points_artifact`。此模式的 `usable` 仅表示返回了可比对坐标，
不表示通过原质量检查。真值仍只由评估器读取，推理不接收模型部位坐标。

## 查看结果

- `run_manifest.json`：清单哈希、源文件哈希、模型、阈值、协议和所选组数。
- `summary.json`：已运行数量、状态分布、返回圆心数量、调用数及耗时；未标注时 `accuracy=null`。
- `results.jsonl`：每个固定请求一行，含 `request_id`、定义、状态、三维中心或 null、失败原因、观测哈希及证据引用。
- `<case_id>/result_A.json`、`result_B.json`：完整 GeometryAgent 返回。
- `<case_id>/episode/states.json` 及对应文件夹：原始图像、分割叠加图、掩码、候选点云、清理后的点云和拟合内点。逻辑文件名通过 EnvState 对应到 `<逻辑文件名>/00.<后缀>`。

拟合中心只读取 `geometry[请求字段].value`，拟合失败不会回退到材质点或分割中位数。`fit_points_artifact` 是拟合内点索引引用的数组；圆形面的拟合输入是投影到该面上的边界点，原始观测边界另存于 `observed_support_artifact`。

## 独立真值和误差统计

推理可以在没有真值的情况下完成。数值评估前生成标注模板：

```bash
python -m toolkits.geometry.evaluation.experiment truth-template \
  --output experiments/geometry/truth.json
```

为全部 160 行填写独立标注。保持模板中的 `request_id`、`reference`、`frame_id` 和 `definition`，所有数值以米为单位：

- 可评：`evaluable=true`，`world_xyz` 为独立参考点，`annotation_source` 写明模型标注或其他独立来源。不能使用 GeometryAgent 输出或它的掩码重新拟合后当真值。
- 不可评：`evaluable=false`、`world_xyz=null`，填写 `annotation_source` 和 `not_evaluable_reason`。应由独立标注判断可见性，不能因为算法拒绝就改成不可评。
- 也可保持 `world_xyz=null`，提供预先固定的 `model_local_xyz` 和同一物体的 4×4 刚体 `model_to_world`。这些输入只由评估器读取。
- 酒瓶可以用 `barrel_profile` 替代 `model_local_xyz`，包含模型局部的单位轴向 `axis`、轴上原点 `axis_origin` 和两个预先固定的轴向界限 `bounds_m`，再提供 `model_to_world`。评估器计算局部轴向中点后变换到世界坐标；该范围不会提供给 GeometryAgent。

模板的 null 是尚未标注，不会被视为零坐标或自动不可评。缺少标注、重复 ID、非米制、错帧或定义不一致时，评估会拒绝生成精度报告。

若按完整 LIBERO 模型建立参考点，采集时添加 `--with-references`，会在每组的 `evaluator_snapshot.json` 中保存同一初始状态的编译后可视网格及物体刚体位姿。该选项需要由本次入口启动 ENV；借用服务须已启用 `--evaluation-references`。先根据清单在完整模型中固定局部参考点，再用对应物体位姿转换到世界坐标，不能把物体原点默认视为部位中心。快照位于 episode 目录外，推理的传感器白名单不会复制它；这些模型数据不进入 GeometryAgent。快照本身不包含部位标注，仍需明确记录参考点定义和模型来源。

```bash
python -m toolkits.geometry.evaluation.experiment evaluate \
  --predictions experiments/geometry/predictions/results.jsonl \
  --truth experiments/geometry/truth.json \
  --tolerance-m 0.01 \
  --output experiments/geometry/evaluation
```

`metrics.json` 和 `errors.csv` 包括逐点误差、按对象/套件分组的返回率及阈值内成功率。成功率的分母是全部独立标注为可评的点，包含算法拒绝、服务失败和未运行项。均值、中位数、P95、RMSE 明确仅统计返回了中心的可评点，须与返回率一起看。默认阈值 1 cm 是可配置的实验判定标准。若评估试跑子集，添加与采集、推理一致的 `--case` 或 `--limit`；真值文件仍需完成全部固定清单的独立标注。

CPU 单元测试使用解析几何场景和替代感知输出，检查清单分派、几何定义、失败条件、调用预算和评估统计。真实 SAM3/视觉模型在仿真 RGB-D 上的定位效果，以及任务操作成功率，需要分别实测。
