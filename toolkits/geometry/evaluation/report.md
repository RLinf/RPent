# LIBERO initial-frame geometry / LIBERO 起始帧几何评估

Coordinates are metres; position errors are millimetres. 坐标单位为米，位置误差单位为毫米。

Planned / 计划项数: **160**; attempted / 已运行: **160**; returned / 返回坐标: **160**.

Tested local revision / 被测本地版本: `89dbdec93e02c580c52888dcee7488696acbd624`; perception / 感知输入: `frozen`; fit profile / 拟合模式: `default_quality_checks_with_warning_estimates`.

Errors below are conditional on an evaluable reference and a returned point. Missing, failed and unrun requests remain in coverage counts. 以下误差只统计有参考点且返回坐标的项；未返回、失败和未运行项保留在返回率统计中。

Frozen initial-frame RGB-D, visual selections and segmentation masks were replayed. Every request was fitted again; no previous predicted coordinates were reused. No fresh visual-model inference or robot actions were performed. 本轮回放冻结的起始帧 RGB-D、视觉选点和分割掩码，逐项重新拟合，未复用旧预测坐标；未重新推理视觉模型或执行机器人动作。

A returned point can fail quality checks. Inspect fit_accepted and warnings; fit residuals are not calibrated position errors. 返回坐标仍可能未通过质量检查，应读取 fit_accepted 和 warnings；拟合残差不是经过标定的位置误差。

## Fit Quality / 拟合质量

| Group / 分组 | Returned / 返回 | Mean mm / 均值 | Median mm / 中位数 | Max mm / 最大值 |
| --- | ---: | ---: | ---: | ---: |
| Accepted / 通过检查 | 18 | 2.233 | 2.342 | 3.143 |
| Warning / 带警告 | 142 | 23.749 | 2.498 | 378.348 |

## Target Parts / 目标部位

| Target / 部位 | Planned / 计划 | Returned / 返回 | Accepted / 通过 | Warning / 警告 | Mean mm / 均值 | Median mm / 中位数 | Max mm / 最大值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| All / 全部 | 160 | 160 | 18 | 142 | 21.329 | 2.453 | 378.348 |
| 盘面中心 | 40 | 40 | 14 | 26 | 1.865 | 1.584 | 3.143 |
| 小烤碗碗口中心 | 20 | 20 | 0 | 20 | 1.732 | 1.657 | 3.204 |
| 奶酪盒顶面中心 | 20 | 20 | 4 | 16 | 1.593 | 1.476 | 4.132 |
| 汤罐罐顶中心 | 20 | 20 | 0 | 20 | 132.222 | 6.641 | 378.348 |
| 黑碗碗口中心 | 30 | 30 | 0 | 30 | 3.422 | 3.250 | 6.533 |
| 酒瓶瓶身主段中心 | 10 | 10 | 0 | 10 | 8.638 | 8.559 | 9.591 |
| 灰白纹理杯杯口中心 | 5 | 5 | 0 | 5 | 69.707 | 69.831 | 70.123 |
| 黄白杯杯口中心 | 10 | 10 | 0 | 10 | 2.517 | 2.456 | 2.913 |
| 微波炉机身顶面中心 | 5 | 5 | 0 | 5 | 12.854 | 11.575 | 18.705 |

## Per-request coordinates / 逐项坐标

| Request | Predicted XYZ m / 测量坐标 | Reference XYZ m / 参考坐标 | Error mm / 误差 | Status / 状态 | Accepted / 通过 | Warnings or reason / 警告或原因 |
| --- | --- | --- | ---: | --- | --- | --- |
| libero_spatial_task_t0_s1/A | 0.0735076, 0.1055144, 0.9136303 | 0.0736415, 0.1060895, 0.9121126 | 1.629 | warning | False | physical_boundary_not_verified |
| libero_spatial_task_t0_s1/B | -0.2059881, 0.1853347, 0.9417370 | -0.2073519, 0.1861381, 0.9419907 | 1.603 | warning | False | insufficient_circle_angular_coverage, circle_inconsistent_with_part_anchor |
| libero_spatial_task_t0_s2/A | 0.0782353, 0.1719255, 0.9081614 | 0.0783293, 0.1724044, 0.9067250 | 1.517 | warning | False | physical_boundary_not_verified |
| libero_spatial_task_t0_s2/B | -0.1960998, 0.2077942, 0.9416631 | -0.1984590, 0.2099368, 0.9419906 | 3.204 | warning | False | insufficient_circle_angular_coverage, circle_inconsistent_with_part_anchor |
| libero_spatial_task_t0_s3/A | 0.0624796, 0.1976707, 0.9081852 | 0.0626770, 0.1983377, 0.9067250 | 1.617 | warning | False | physical_boundary_not_verified |
| libero_spatial_task_t0_s3/B | -0.2073091, 0.2030121, 0.9417104 | -0.2088115, 0.2039570, 0.9419906 | 1.797 | warning | False | insufficient_circle_angular_coverage, circle_inconsistent_with_part_anchor |
| libero_spatial_task_t0_s4/A | 0.0594071, 0.2630788, 0.9087933 | 0.0599429, 0.2618442, 0.9067250 | 2.468 | usable | True | — |
| libero_spatial_task_t0_s4/B | -0.2149383, 0.1855015, 0.9417529 | -0.2161205, 0.1864548, 0.9419906 | 1.537 | warning | False | insufficient_circle_angular_coverage, circle_inconsistent_with_part_anchor |
| libero_spatial_task_t0_s5/A | 0.0524627, 0.2120125, 0.9081927 | 0.0525128, 0.2126940, 0.9067250 | 1.619 | warning | False | physical_boundary_not_verified |
| libero_spatial_task_t0_s5/B | -0.2039882, 0.1932070, 0.9417698 | -0.2049875, 0.1941164, 0.9419906 | 1.369 | warning | False | insufficient_circle_angular_coverage, circle_inconsistent_with_part_anchor |
| libero_spatial_task_t4_s1/A | 0.0492499, 0.1971033, 0.9081941 | 0.0492955, 0.1977607, 0.9067250 | 1.610 | warning | False | physical_boundary_not_verified |
| libero_spatial_task_t4_s1/B | -0.2041566, 0.1984019, 0.9416558 | -0.2057693, 0.1993741, 0.9419906 | 1.913 | warning | False | insufficient_circle_angular_coverage, circle_inconsistent_with_part_anchor |
| libero_spatial_task_t4_s2/A | 0.0578849, 0.2004178, 0.9081836 | 0.0581190, 0.2009242, 0.9067250 | 1.562 | warning | False | physical_boundary_not_verified |
| libero_spatial_task_t4_s2/B | -0.1988181, 0.1976746, 0.9417521 | -0.2002617, 0.1985589, 0.9419906 | 1.710 | warning | False | insufficient_circle_angular_coverage, circle_inconsistent_with_part_anchor |
| libero_spatial_task_t4_s3/A | 0.0562258, 0.2030312, 0.9081764 | 0.0564397, 0.2036681, 0.9067250 | 1.599 | warning | False | physical_boundary_not_verified |
| libero_spatial_task_t4_s3/B | -0.2001077, 0.2015164, 0.9417982 | -0.2014541, 0.2024617, 0.9419906 | 1.656 | warning | False | insufficient_circle_angular_coverage, circle_inconsistent_with_part_anchor |
| libero_spatial_task_t4_s4/A | 0.0569816, 0.2115329, 0.9081894 | 0.0572397, 0.2120792, 0.9067250 | 1.584 | warning | False | physical_boundary_not_verified |
| libero_spatial_task_t4_s4/B | -0.2074268, 0.1983683, 0.9400206 | -0.2074317, 0.1995590, 0.9419906 | 2.302 | warning | False | insufficient_circle_angular_coverage, noncircular_boundary_residual, circle_inconsistent_with_part_anchor |
| libero_spatial_task_t4_s5/A | 0.0694208, 0.2122886, 0.9087811 | 0.0697440, 0.2124956, 0.9067250 | 2.092 | usable | True | — |
| libero_spatial_task_t4_s5/B | -0.2120606, 0.2016619, 0.9416628 | -0.2134435, 0.2026131, 0.9419906 | 1.710 | warning | False | insufficient_circle_angular_coverage, circle_inconsistent_with_part_anchor |
| libero_spatial_swap_t0_s1/A | 0.0570727, 0.0381844, 0.9081835 | 0.0569786, 0.0385363, 0.9067250 | 1.503 | warning | False | physical_boundary_not_verified |
| libero_spatial_swap_t0_s1/B | -0.1971942, 0.1988907, 0.9417383 | -0.1983063, 0.1998101, 0.9419906 | 1.465 | warning | False | insufficient_circle_angular_coverage, circle_inconsistent_with_part_anchor |
| libero_spatial_swap_t0_s2/A | 0.0756527, 0.0402468, 0.9081871 | 0.0757489, 0.0405382, 0.9067250 | 1.494 | warning | False | physical_boundary_not_verified |
| libero_spatial_swap_t0_s2/B | -0.2108356, 0.2090621, 0.9418747 | -0.2119302, 0.2099754, 0.9419906 | 1.430 | warning | False | insufficient_circle_angular_coverage, circle_inconsistent_with_part_anchor |
| libero_spatial_swap_t0_s3/A | 0.0847264, 0.0261758, 0.9081868 | 0.0848353, 0.0265043, 0.9067250 | 1.502 | warning | False | physical_boundary_not_verified |
| libero_spatial_swap_t0_s3/B | -0.1987970, 0.1956940, 0.9418658 | -0.1997568, 0.1966116, 0.9419906 | 1.334 | warning | False | insufficient_circle_angular_coverage, circle_inconsistent_with_part_anchor |
| libero_spatial_swap_t0_s4/A | 0.0586462, 0.0177713, 0.9081860 | 0.0586886, 0.0179944, 0.9067250 | 1.479 | warning | False | physical_boundary_not_verified |
| libero_spatial_swap_t0_s4/B | -0.2047185, 0.1953604, 0.9417548 | -0.2060998, 0.1962470, 0.9419906 | 1.658 | warning | False | insufficient_circle_angular_coverage, circle_inconsistent_with_part_anchor |
| libero_spatial_swap_t0_s5/A | 0.0595635, 0.0313935, 0.9081739 | 0.0596562, 0.0318241, 0.9067250 | 1.514 | warning | False | physical_boundary_not_verified |
| libero_spatial_swap_t0_s5/B | -0.2082070, 0.1964716, 0.9417654 | -0.2096242, 0.1974303, 0.9419906 | 1.726 | warning | False | insufficient_circle_angular_coverage, circle_inconsistent_with_part_anchor |
| libero_spatial_swap_t4_s1/A | 0.0839926, 0.0285897, 0.9081769 | 0.0839610, 0.0288698, 0.9067250 | 1.479 | warning | False | physical_boundary_not_verified |
| libero_spatial_swap_t4_s1/B | -0.2016298, 0.1990318, 0.9418979 | -0.2028182, 0.1999313, 0.9419906 | 1.493 | warning | False | insufficient_circle_angular_coverage, circle_inconsistent_with_part_anchor |
| libero_spatial_swap_t4_s2/A | 0.0812624, 0.0215081, 0.9081545 | 0.0812238, 0.0217722, 0.9067250 | 1.454 | warning | False | physical_boundary_not_verified |
| libero_spatial_swap_t4_s2/B | -0.1858389, 0.1871368, 0.9416546 | -0.1872488, 0.1881678, 0.9419906 | 1.779 | warning | False | insufficient_circle_angular_coverage |
| libero_spatial_swap_t4_s3/A | 0.0663622, 0.0397213, 0.9081702 | 0.0663335, 0.0401441, 0.9067250 | 1.506 | warning | False | physical_boundary_not_verified |
| libero_spatial_swap_t4_s3/B | -0.1850734, 0.2072418, 0.9415992 | -0.1862863, 0.2081210, 0.9419906 | 1.548 | warning | False | insufficient_circle_angular_coverage, circle_inconsistent_with_part_anchor |
| libero_spatial_swap_t4_s4/A | 0.0633754, 0.0436158, 0.9081778 | 0.0636280, 0.0440895, 0.9067250 | 1.549 | warning | False | physical_boundary_not_verified |
| libero_spatial_swap_t4_s4/B | -0.1976358, 0.2044982, 0.9417775 | -0.1990921, 0.2054869, 0.9419906 | 1.773 | warning | False | insufficient_circle_angular_coverage, circle_inconsistent_with_part_anchor |
| libero_spatial_swap_t4_s5/A | 0.0655601, 0.0364927, 0.9081760 | 0.0653911, 0.0370440, 0.9067250 | 1.561 | warning | False | physical_boundary_not_verified |
| libero_spatial_swap_t4_s5/B | -0.2094897, 0.2023706, 0.9417986 | -0.2107882, 0.2033546, 0.9419906 | 1.641 | warning | False | insufficient_circle_angular_coverage |
| libero_object_task_t0_s1/A | -0.1549039, 0.0564487, 0.0188319 | -0.1545503, 0.0567581, 0.0180659 | 0.899 | warning | False | four_complete_physical_edges_not_observed, physical_boundary_not_verified |
| libero_object_task_t0_s1/B | -0.1257882, -0.2444875, 0.0728620 | -0.1184758, -0.2433067, 0.0717673 | 7.488 | warning | False | noncircular_boundary_residual |
| libero_object_task_t0_s2/A | -0.1508713, 0.0643108, 0.0187813 | -0.1508688, 0.0628983, 0.0180659 | 1.583 | warning | False | four_complete_physical_edges_not_observed, physical_boundary_not_verified |
| libero_object_task_t0_s2/B | 0.1454194, 0.0278081, 0.0725921 | -0.1184758, -0.2433067, 0.0717673 | 378.345 | warning | False | noncircular_boundary_residual |
| libero_object_task_t0_s3/A | -0.1463017, 0.0612317, 0.0187991 | -0.1464570, 0.0605456, 0.0180659 | 1.016 | warning | False | four_complete_physical_edges_not_observed, physical_boundary_not_verified |
| libero_object_task_t0_s3/B | 0.1454210, 0.0278111, 0.0725961 | -0.1184758, -0.2433067, 0.0717673 | 378.348 | warning | False | noncircular_boundary_residual |
| libero_object_task_t0_s4/A | -0.1498359, 0.0610005, 0.0187226 | -0.1509115, 0.0620019, 0.0180659 | 1.610 | warning | False | four_complete_physical_edges_not_observed |
| libero_object_task_t0_s4/B | -0.1257124, -0.2448358, 0.0728827 | -0.1184758, -0.2433067, 0.0717673 | 7.480 | warning | False | noncircular_boundary_residual |
| libero_object_task_t0_s5/A | -0.1519205, 0.0608862, 0.0187857 | -0.1523837, 0.0593208, 0.0180659 | 1.784 | warning | False | four_complete_physical_edges_not_observed, physical_boundary_not_verified |
| libero_object_task_t0_s5/B | -0.1256992, -0.2447209, 0.0728322 | -0.1184758, -0.2433067, 0.0717673 | 7.437 | warning | False | noncircular_boundary_residual |
| libero_object_task_t4_s1/A | 0.1523401, 0.0294842, 0.0186381 | 0.1539493, 0.0290846, 0.0180659 | 1.754 | warning | False | insufficient_physical_rectangle_boundary, four_complete_physical_edges_not_observed |
| libero_object_task_t4_s1/B | 0.0958347, -0.2042504, 0.0726433 | 0.1015242, -0.2033067, 0.0717673 | 5.833 | warning | False | noncircular_boundary_residual |
| libero_object_task_t4_s2/A | 0.1556020, 0.0258090, 0.0184700 | 0.1545071, 0.0262399, 0.0180659 | 1.244 | usable | True | — |
| libero_object_task_t4_s2/B | 0.0958340, -0.2042547, 0.0726446 | 0.1015242, -0.2033067, 0.0717673 | 5.835 | warning | False | noncircular_boundary_residual |
| libero_object_task_t4_s3/A | 0.1521690, 0.0305347, 0.0184739 | 0.1508813, 0.0308703, 0.0180659 | 1.392 | usable | True | — |
| libero_object_task_t4_s3/B | 0.0958340, -0.2042547, 0.0726446 | 0.1015242, -0.2033067, 0.0717673 | 5.835 | warning | False | noncircular_boundary_residual |
| libero_object_task_t4_s4/A | 0.1489870, 0.0341751, 0.0185046 | 0.1479955, 0.0348375, 0.0180659 | 1.271 | usable | True | — |
| libero_object_task_t4_s4/B | 0.0958486, -0.2043114, 0.0726320 | 0.1015242, -0.2033067, 0.0717673 | 5.828 | warning | False | noncircular_boundary_residual |
| libero_object_task_t4_s5/A | 0.1567020, 0.0365535, 0.0185456 | 0.1538719, 0.0335809, 0.0180659 | 4.132 | warning | False | degenerate_plane_support, four_complete_physical_edges_not_observed, physical_boundary_not_verified |
| libero_object_task_t4_s5/B | 0.0958350, -0.2042507, 0.0726385 | 0.1015242, -0.2033067, 0.0717673 | 5.832 | warning | False | noncircular_boundary_residual |
| libero_object_swap_t0_s1/A | -0.1464571, 0.0637376, 0.0187870 | -0.1467256, 0.0624950, 0.0180659 | 1.462 | warning | False | four_complete_physical_edges_not_observed, physical_boundary_not_verified |
| libero_object_swap_t0_s1/B | 0.1454205, 0.0278042, 0.0725780 | -0.1984758, -0.0833067, 0.0717673 | 361.401 | warning | False | noncircular_boundary_residual |
| libero_object_swap_t0_s2/A | -0.1504881, 0.0625733, 0.0187853 | -0.1502828, 0.0611274, 0.0180659 | 1.628 | warning | False | four_complete_physical_edges_not_observed, physical_boundary_not_verified |
| libero_object_swap_t0_s2/B | 0.1454201, 0.0278041, 0.0725806 | -0.1984758, -0.0833067, 0.0717673 | 361.401 | warning | False | noncircular_boundary_residual |
| libero_object_swap_t0_s3/A | -0.1540060, 0.0618915, 0.0188055 | -0.1539051, 0.0612275, 0.0180659 | 0.999 | warning | False | four_complete_physical_edges_not_observed, physical_boundary_not_verified |
| libero_object_swap_t0_s3/B | 0.1454196, 0.0278098, 0.0725979 | -0.1984758, -0.0833067, 0.0717673 | 361.402 | warning | False | noncircular_boundary_residual |
| libero_object_swap_t0_s4/A | -0.1515375, 0.0632194, 0.0187912 | -0.1524382, 0.0622786, 0.0180659 | 1.491 | warning | False | insufficient_physical_rectangle_boundary, four_complete_physical_edges_not_observed |
| libero_object_swap_t0_s4/B | 0.1454200, 0.0278097, 0.0725830 | -0.1984758, -0.0833067, 0.0717673 | 361.403 | warning | False | noncircular_boundary_residual |
| libero_object_swap_t0_s5/A | -0.1470060, 0.0558959, 0.0188109 | -0.1468712, 0.0557300, 0.0180659 | 0.775 | warning | False | four_complete_physical_edges_not_observed, physical_boundary_not_verified |
| libero_object_swap_t0_s5/B | 0.1454218, 0.0278053, 0.0725863 | -0.1984758, -0.0833067, 0.0717673 | 361.403 | warning | False | noncircular_boundary_residual |
| libero_object_swap_t4_s1/A | -0.1160062, -0.2390549, 0.0187098 | -0.1163537, -0.2393465, 0.0180659 | 0.788 | warning | False | four_complete_physical_edges_not_observed |
| libero_object_swap_t4_s1/B | 0.0958319, -0.2042541, 0.0726599 | 0.1015242, -0.2033067, 0.0717673 | 5.839 | warning | False | noncircular_boundary_residual |
| libero_object_swap_t4_s2/A | -0.1128036, -0.2393294, 0.0187716 | -0.1157913, -0.2402366, 0.0180659 | 3.201 | warning | False | four_complete_physical_edges_not_observed, physical_boundary_not_verified |
| libero_object_swap_t4_s2/B | 0.0958299, -0.2042695, 0.0726701 | 0.1015242, -0.2033067, 0.0717673 | 5.845 | warning | False | noncircular_boundary_residual |
| libero_object_swap_t4_s3/A | -0.1186953, -0.2374242, 0.0185263 | -0.1202290, -0.2375640, 0.0180659 | 1.607 | usable | True | — |
| libero_object_swap_t4_s3/B | 0.0958332, -0.2042528, 0.0726529 | 0.1015242, -0.2033067, 0.0717673 | 5.837 | warning | False | noncircular_boundary_residual |
| libero_object_swap_t4_s4/A | -0.1143640, -0.2360767, 0.0188173 | -0.1161246, -0.2350421, 0.0180659 | 2.176 | warning | False | four_complete_physical_edges_not_observed, physical_boundary_not_verified |
| libero_object_swap_t4_s4/B | 0.0958359, -0.2042527, 0.0726500 | 0.1015242, -0.2033067, 0.0717673 | 5.834 | warning | False | noncircular_boundary_residual |
| libero_object_swap_t4_s5/A | -0.1198524, -0.2421167, 0.0186813 | -0.1206659, -0.2423337, 0.0180659 | 1.043 | warning | False | four_complete_physical_edges_not_observed |
| libero_object_swap_t4_s5/B | 0.0958533, -0.2043106, 0.0726222 | 0.1015242, -0.2033067, 0.0717673 | 5.822 | warning | False | noncircular_boundary_residual |
| libero_goal_task_t0_s1/A | 0.0523804, -0.0248630, 0.9087798 | 0.0534000, -0.0258252, 0.9067250 | 2.488 | usable | True | — |
| libero_goal_task_t0_s1/B | -0.0963082, 0.0091174, 0.9468939 | -0.0980228, 0.0090189, 0.9501093 | 3.645 | warning | False | insufficient_circle_angular_coverage |
| libero_goal_task_t0_s2/A | 0.0449691, -0.0085252, 0.9081803 | 0.0451724, -0.0080457, 0.9067250 | 1.546 | warning | False | physical_boundary_not_verified |
| libero_goal_task_t0_s2/B | -0.1023209, 0.0043842, 0.9479505 | -0.1027318, 0.0043122, 0.9501093 | 2.199 | warning | False | insufficient_circle_angular_coverage |
| libero_goal_task_t0_s3/A | 0.0510989, -0.0253380, 0.9087367 | 0.0519600, -0.0246307, 0.9067250 | 2.300 | usable | True | — |
| libero_goal_task_t0_s3/B | -0.0985928, -0.0008813, 0.9477320 | -0.0999642, -0.0008382, 0.9501093 | 2.745 | warning | False | insufficient_circle_angular_coverage |
| libero_goal_task_t0_s4/A | 0.0450646, -0.0080733, 0.9087568 | 0.0463002, -0.0065325, 0.9067250 | 2.834 | usable | True | — |
| libero_goal_task_t0_s4/B | -0.0889060, 0.0014772, 0.9459267 | -0.0918471, 0.0018189, 0.9501093 | 5.124 | warning | False | insufficient_circle_angular_coverage |
| libero_goal_task_t0_s5/A | 0.0501971, -0.0279378, 0.9088192 | 0.0516381, -0.0288446, 0.9067250 | 2.699 | usable | True | — |
| libero_goal_task_t0_s5/B | -0.0908271, 0.0145891, 0.9473418 | -0.0925311, 0.0146602, 0.9501093 | 3.251 | warning | False | insufficient_circle_angular_coverage |
| libero_goal_task_t7_s1/A | 0.0574213, -0.0263574, 0.9087625 | 0.0579609, -0.0261452, 0.9067250 | 2.118 | usable | True | — |
| libero_goal_task_t7_s1/B | -0.0742939, -0.0108429, 0.9471481 | -0.0756308, -0.0108033, 0.9501093 | 3.249 | warning | False | insufficient_circle_angular_coverage |
| libero_goal_task_t7_s2/A | 0.0443735, -0.0200462, 0.9082190 | 0.0445668, -0.0195550, 0.9067250 | 1.585 | warning | False | physical_boundary_not_verified |
| libero_goal_task_t7_s2/B | -0.0940067, -0.0011403, 0.9474367 | -0.0955940, -0.0011514, 0.9501093 | 3.108 | warning | False | insufficient_circle_angular_coverage |
| libero_goal_task_t7_s3/A | 0.0600858, -0.0270198, 0.9088030 | 0.0616553, -0.0267590, 0.9067250 | 2.617 | usable | True | — |
| libero_goal_task_t7_s3/B | -0.0869663, 0.0016485, 0.9470113 | -0.0894835, 0.0017463, 0.9501093 | 3.993 | warning | False | insufficient_circle_angular_coverage |
| libero_goal_task_t7_s4/A | 0.0556422, -0.0316064, 0.9088237 | 0.0562235, -0.0325786, 0.9067250 | 2.385 | usable | True | — |
| libero_goal_task_t7_s4/B | -0.0945856, -0.0071329, 0.9480730 | -0.0945615, -0.0071039, 0.9501093 | 2.037 | warning | False | insufficient_circle_angular_coverage |
| libero_goal_task_t7_s5/A | 0.0492592, -0.0159450, 0.9081964 | 0.0492889, -0.0155078, 0.9067250 | 1.535 | warning | False | physical_boundary_not_verified |
| libero_goal_task_t7_s5/B | -0.0907848, 0.0107849, 0.9473149 | -0.0922963, 0.0106816, 0.9501093 | 3.179 | warning | False | insufficient_circle_angular_coverage |
| libero_goal_swap_t0_s1/A | 0.0495090, -0.0134476, 0.9082198 | 0.0496974, -0.0132283, 0.9067250 | 1.523 | warning | False | physical_boundary_not_verified |
| libero_goal_swap_t0_s1/B | -0.0991268, 0.0147419, 0.9478442 | -0.1001962, 0.0148023, 0.9501093 | 2.506 | warning | False | insufficient_circle_angular_coverage |
| libero_goal_swap_t0_s2/A | 0.0420950, -0.0063755, 0.9081881 | 0.0420027, -0.0060181, 0.9067250 | 1.509 | warning | False | physical_boundary_not_verified |
| libero_goal_swap_t0_s2/B | -0.0846323, -0.0017216, 0.9480656 | -0.0845458, -0.0015340, 0.9501093 | 2.054 | warning | False | insufficient_circle_angular_coverage |
| libero_goal_swap_t0_s3/A | 0.0460479, -0.0280999, 0.9081676 | 0.0460371, -0.0277541, 0.9067250 | 1.484 | warning | False | physical_boundary_not_verified |
| libero_goal_swap_t0_s3/B | -0.0808980, 0.0004904, 0.9463426 | -0.0836595, 0.0005740, 0.9501093 | 4.671 | warning | False | insufficient_circle_angular_coverage |
| libero_goal_swap_t0_s4/A | 0.0346756, -0.0344200, 0.9088471 | 0.0350741, -0.0336929, 0.9067250 | 2.278 | usable | True | — |
| libero_goal_swap_t0_s4/B | -0.0956325, 0.0139752, 0.9482835 | -0.0957070, 0.0141597, 0.9501093 | 1.837 | warning | False | insufficient_circle_angular_coverage |
| libero_goal_swap_t0_s5/A | 0.0614368, -0.0060292, 0.9087479 | 0.0606804, -0.0064365, 0.9067250 | 2.198 | usable | True | — |
| libero_goal_swap_t0_s5/B | -0.0958611, 0.0124770, 0.9475163 | -0.0976855, 0.0125669, 0.9501093 | 3.172 | warning | False | insufficient_circle_angular_coverage |
| libero_goal_swap_t7_s1/A | 0.0451136, -0.0269536, 0.9081828 | 0.0451769, -0.0265121, 0.9067250 | 1.525 | warning | False | physical_boundary_not_verified |
| libero_goal_swap_t7_s1/B | -0.1000045, -0.0028536, 0.9471819 | -0.1024334, -0.0028332, 0.9501093 | 3.804 | warning | False | insufficient_circle_angular_coverage |
| libero_goal_swap_t7_s2/A | 0.0511853, -0.0230911, 0.9191280 | 0.0524943, -0.0220799, 0.9173366 | 2.438 | usable | True | — |
| libero_goal_swap_t7_s2/B | -0.0738292, -0.0087421, 0.9473204 | -0.0757051, -0.0088204, 0.9501093 | 3.362 | warning | False | insufficient_circle_angular_coverage |
| libero_goal_swap_t7_s3/A | 0.0593449, -0.0342221, 0.9081705 | 0.0594332, -0.0340995, 0.9067250 | 1.453 | warning | False | physical_boundary_not_verified |
| libero_goal_swap_t7_s3/B | -0.0734763, -0.0054979, 0.9472785 | -0.0754992, -0.0055544, 0.9501093 | 3.480 | warning | False | insufficient_circle_angular_coverage |
| libero_goal_swap_t7_s4/A | 0.0567465, -0.0141384, 0.9209186 | 0.0587385, -0.0120919, 0.9196064 | 3.143 | usable | True | — |
| libero_goal_swap_t7_s4/B | -0.0843876, 0.0010572, 0.9473858 | -0.0859564, 0.0010438, 0.9501093 | 3.143 | warning | False | insufficient_circle_angular_coverage |
| libero_goal_swap_t7_s5/A | 0.0400325, -0.0129463, 0.9179211 | 0.0417007, -0.0126207, 0.9159333 | 2.615 | usable | True | — |
| libero_goal_swap_t7_s5/B | -0.1011954, -0.0022524, 0.9476656 | -0.1022107, -0.0022602, 0.9501093 | 2.646 | warning | False | insufficient_circle_angular_coverage |
| libero_10_task_t3_s1/A | 0.0194838, -0.0643918, 0.9450323 | 0.0175312, -0.0645374, 0.9501093 | 5.442 | warning | False | insufficient_circle_angular_coverage |
| libero_10_task_t3_s1/B | -0.1524951, 0.0696689, 0.9385836 | -0.1518459, 0.0699413, 0.9302179 | 8.395 | warning | False | axisymmetric_surface_residual_too_large_relative_to_radius |
| libero_10_task_t3_s2/A | 0.0408908, -0.0482085, 0.9459778 | 0.0409959, -0.0481926, 0.9501093 | 4.133 | warning | False | insufficient_circle_angular_coverage |
| libero_10_task_t3_s2/B | -0.1625635, 0.0507871, 0.9390076 | -0.1619829, 0.0511304, 0.9302179 | 8.816 | warning | False | axisymmetric_surface_residual_too_large_relative_to_radius |
| libero_10_task_t3_s3/A | 0.0295110, -0.0600426, 0.9459856 | 0.0285461, -0.0600574, 0.9501093 | 4.235 | warning | False | insufficient_circle_angular_coverage |
| libero_10_task_t3_s3/B | -0.1419374, 0.0432694, 0.9388334 | -0.1412863, 0.0435842, 0.9302179 | 8.646 | warning | False | axisymmetric_surface_residual_too_large_relative_to_radius |
| libero_10_task_t3_s4/A | 0.0412236, -0.0612607, 0.9456665 | 0.0402475, -0.0613454, 0.9501093 | 4.550 | warning | False | insufficient_circle_angular_coverage |
| libero_10_task_t3_s4/B | -0.1497878, 0.0626005, 0.9397504 | -0.1487380, 0.0627299, 0.9302179 | 9.591 | warning | False | axisymmetric_surface_residual_too_large_relative_to_radius, insufficient_barrel_consensus_on_complete_mask |
| libero_10_task_t3_s5/A | 0.0271304, -0.0675594, 0.9442178 | 0.0243104, -0.0674351, 0.9501093 | 6.533 | warning | False | insufficient_circle_angular_coverage |
| libero_10_task_t3_s5/B | -0.1297154, 0.0264864, 0.9386220 | -0.1292552, 0.0267602, 0.9302179 | 8.421 | warning | False | axisymmetric_surface_residual_too_large_relative_to_radius |
| libero_10_task_t9_s1/A | -0.1236912, -0.2345335, 0.9563869 | -0.1250006, -0.2773983, 1.0118675 | 70.123 | warning | False | circle_anchor_constraint_relaxed, low_circle_inlier_ratio, insufficient_circle_angular_coverage, noncircular_boundary_residual, circle_inconsistent_with_part_anchor |
| libero_10_task_t9_s1/B | -0.0089132, 0.0112657, 1.0022262 | -0.0106865, 0.0112783, 1.0038445 | 2.401 | warning | False | low_circle_inlier_ratio, insufficient_circle_angular_coverage |
| libero_10_task_t9_s2/A | -0.1159507, -0.2147524, 0.9567716 | -0.1169476, -0.2563617, 1.0118675 | 69.050 | warning | False | circle_anchor_constraint_relaxed, low_circle_inlier_ratio, insufficient_circle_angular_coverage, noncircular_boundary_residual, circle_inconsistent_with_part_anchor |
| libero_10_task_t9_s2/B | 0.0060602, 0.0340081, 1.0020312 | 0.0044880, 0.0341518, 1.0038445 | 2.404 | warning | False | low_circle_inlier_ratio, insufficient_circle_angular_coverage |
| libero_10_task_t9_s3/A | -0.0845986, -0.2015819, 0.9559513 | -0.0854412, -0.2437589, 1.0118675 | 70.044 | warning | False | circle_anchor_constraint_relaxed, low_circle_inlier_ratio, insufficient_circle_angular_coverage, noncircular_boundary_residual, circle_inconsistent_with_part_anchor |
| libero_10_task_t9_s3/B | -0.0033224, 0.0316784, 1.0020545 | -0.0051239, 0.0318299, 1.0038445 | 2.544 | warning | False | low_circle_inlier_ratio, insufficient_circle_angular_coverage |
| libero_10_task_t9_s4/A | -0.0768492, -0.2387486, 0.9566478 | -0.0779450, -0.2809185, 1.0118675 | 69.489 | warning | False | circle_anchor_constraint_relaxed, low_circle_inlier_ratio, insufficient_circle_angular_coverage, noncircular_boundary_residual, circle_inconsistent_with_part_anchor |
| libero_10_task_t9_s4/B | 0.0065001, 0.0084406, 1.0016406 | 0.0045954, 0.0084245, 1.0038445 | 2.913 | warning | False | low_circle_inlier_ratio, insufficient_circle_angular_coverage |
| libero_10_task_t9_s5/A | -0.0859676, -0.2487540, 0.9565258 | -0.0871952, -0.2913227, 1.0118675 | 69.831 | warning | False | circle_anchor_constraint_relaxed, low_circle_inlier_ratio, insufficient_circle_angular_coverage, noncircular_boundary_residual, circle_inconsistent_with_part_anchor |
| libero_10_task_t9_s5/B | -0.0177552, 0.0103324, 1.0020251 | -0.0193767, 0.0103338, 1.0038445 | 2.437 | warning | False | low_circle_inlier_ratio, insufficient_circle_angular_coverage |
| libero_10_swap_t3_s1/A | -0.1687886, 0.0665651, 0.9474429 | -0.1707480, 0.0668170, 0.9501093 | 3.318 | warning | False | insufficient_circle_angular_coverage |
| libero_10_swap_t3_s1/B | 0.0228911, -0.0694104, 0.9383797 | 0.0234113, -0.0691759, 0.9302179 | 8.182 | warning | False | axisymmetric_surface_residual_too_large_relative_to_radius |
| libero_10_swap_t3_s2/A | -0.1502554, 0.0313907, 0.9478867 | -0.1514462, 0.0316680, 0.9501093 | 2.537 | warning | False | insufficient_circle_angular_coverage |
| libero_10_swap_t3_s2/B | 0.0503018, -0.0452747, 0.9387388 | 0.0508220, -0.0450443, 0.9302179 | 8.540 | warning | False | axisymmetric_surface_residual_too_large_relative_to_radius |
| libero_10_swap_t3_s3/A | -0.1436021, 0.0507618, 0.9480687 | -0.1450273, 0.0508524, 0.9501093 | 2.491 | warning | False | insufficient_circle_angular_coverage |
| libero_10_swap_t3_s3/B | 0.0052695, -0.0445716, 0.9388713 | 0.0058766, -0.0443257, 0.9302179 | 8.678 | warning | False | axisymmetric_surface_residual_too_large_relative_to_radius |
| libero_10_swap_t3_s4/A | -0.1612821, 0.0740269, 0.9484755 | -0.1624167, 0.0741901, 0.9501093 | 1.996 | warning | False | insufficient_circle_angular_coverage |
| libero_10_swap_t3_s4/B | 0.0168054, -0.0278623, 0.9387718 | 0.0173893, -0.0275924, 0.9302179 | 8.578 | warning | False | axisymmetric_surface_residual_too_large_relative_to_radius |
| libero_10_swap_t3_s5/A | -0.1361591, 0.0627935, 0.9467780 | -0.1387117, 0.0631519, 0.9501093 | 4.212 | warning | False | insufficient_circle_angular_coverage |
| libero_10_swap_t3_s5/B | 0.0321669, -0.0282380, 0.9387316 | 0.0326037, -0.0279487, 0.9302179 | 8.530 | warning | False | axisymmetric_surface_residual_too_large_relative_to_radius |
| libero_10_swap_t9_s1/A | -0.0018827, 0.0129478, 1.0021788 | -0.0033487, 0.0130525, 1.0038445 | 2.221 | warning | False | low_circle_inlier_ratio, insufficient_circle_angular_coverage |
| libero_10_swap_t9_s1/B | -0.1106480, -0.2415904, 1.1074018 | -0.1123323, -0.2523276, 1.1069114 | 10.880 | warning | False | low_plane_inlier_ratio, four_complete_physical_edges_not_observed, face_boundary_clipped_by_image |
| libero_10_swap_t9_s2/A | 0.0210698, 0.0024799, 1.0018538 | 0.0191967, 0.0024339, 1.0038445 | 2.734 | warning | False | low_circle_inlier_ratio, insufficient_circle_angular_coverage |
| libero_10_swap_t9_s2/B | -0.1015709, -0.2506075, 1.1074045 | -0.1042951, -0.2617436, 1.1069114 | 11.475 | warning | False | low_plane_inlier_ratio, four_complete_physical_edges_not_observed, face_boundary_clipped_by_image |
| libero_10_swap_t9_s3/A | -0.0118134, 0.0308255, 1.0019713 | -0.0136567, 0.0309767, 1.0038445 | 2.632 | warning | False | low_circle_inlier_ratio, insufficient_circle_angular_coverage |
| libero_10_swap_t9_s3/B | -0.0779742, -0.2320153, 1.1073780 | -0.0809057, -0.2432643, 1.1069114 | 11.634 | warning | False | low_plane_inlier_ratio, face_boundary_clipped_by_image |
| libero_10_swap_t9_s4/A | -0.0128035, 0.0376289, 1.0019976 | -0.0144455, 0.0377573, 1.0038445 | 2.475 | warning | False | low_circle_inlier_ratio, insufficient_circle_angular_coverage |
| libero_10_swap_t9_s4/B | -0.0945891, -0.2437173, 1.1073805 | -0.0976804, -0.2548621, 1.1069114 | 11.575 | warning | False | low_plane_inlier_ratio, four_complete_physical_edges_not_observed, face_boundary_clipped_by_image |
| libero_10_swap_t9_s5/A | 0.0116945, 0.0095069, 1.0020909 | 0.0100407, 0.0096052, 1.0038445 | 2.412 | warning | False | low_circle_inlier_ratio, insufficient_circle_angular_coverage |
| libero_10_swap_t9_s5/B | -0.0622890, -0.2163075, 1.1073799 | -0.0775378, -0.2271310, 1.1069114 | 18.705 | warning | False | low_plane_inlier_ratio, four_complete_physical_edges_not_observed, face_boundary_clipped_by_image |

## Reproduction / 复现记录

- Published implementation / 已发布实现: [699fa46](https://github.com/nanji17/RPent/commit/699fa4603dbf2f991385f566d8e8009cf0cd15b2).
- Source tree / 源码树: `bc4f3e3a188aab96d8471c18553295aed5b7b1de`, identical to the tested local revision / 与被测本地版本完全一致。
- Run / 运行标识: `geometrytool-baseline-v3-frozen-160-r1`.
- Checklist / 清单 SHA256: `27f89e9a7a1afe32cfbe4071968b3b8e1dddc2565ef5958d046858bf6bab6e93`.
- Predictions / 预测 SHA256: `80aef6b91facc3aa87320a4d917612db91707a1c19707be3e8aed7a48cb1608c`.
- References / 参考坐标 SHA256: `809690847290c179f098880848252a616b79efec2114c9515556b056f22c2dd7`.
- Source archive / 源码归档 SHA256: `d8a665844ed4120c07acc180e7935657e883f882608dd86f51de5352192a1b42`.

The evaluator also writes full-precision `errors.csv` and `metrics.json` alongside this report in the run output. These raw artifacts are retained separately from the report included in the PR. 评估器还会在运行输出中生成完整精度的 `errors.csv` 和 `metrics.json`；原始数据与 PR 中的单份报告分开保存。

This evaluates initial-frame geometry, not manipulation success or calibrated uncertainty. 本报告评估起始帧几何误差，不代表操作任务成功率或经过标定的不确定度。

## Validation / 验证

- Geometry/evaluation offline checks / 几何及评估离线检查: 31 passed.
- Repository unit suite plus evaluation checks / 全仓库单元测试及评估检查: 1725 passed, 13 skipped (Linux, Python 3.11.14).
- The reused test environment lacks pytest-timeout, producing three configuration/marker warnings; all collected non-skipped tests completed / 复用的测试环境缺少 pytest-timeout，产生三条配置或标记提示，所有未跳过的用例均执行完成。
- `pre-commit run --all-files`: passed; Python 3.10–3.12 CI matrix not run locally / 通过，未在本地运行完整 CI Python 版本矩阵。
- Frozen pixels, segmentation masks, candidate masks and candidate 3-D clouds matched for all 160 requests / 160 项选点像素、分割掩码、候选掩码和候选三维点全部一致。
- Every returned position error was independently recomputed from full-precision coordinates / 所有返回位置误差均由完整精度坐标独立复算。
- One CPU, nice 19, idle I/O, single-thread BLAS, GPU disabled; owned processes ended / 单 CPU、nice 19、空闲 I/O 优先级、BLAS 单线程、GPU 关闭，所属进程已结束。
