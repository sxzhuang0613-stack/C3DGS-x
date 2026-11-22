# alpha 切片的批量高斯操作与阈值校准

结构算法标识为 `batch_residual_v5`（不做 Clone）。本文只描述结构操作；渲染模型语义由实际渲染库决定。

## 已实现的指标

每个点按可见次数平均位置梯度，先换算坐标单位：

`g_i = mean_visible(|| xyz.grad_i * [dx,dy,dz] ||_2)`。

新 geometry 的 `[dx,dy,dz] = spacing_xyz / scale_mm`；legacy_box 则按 bounds 和输出尺寸换算。
它表示移动一个体素的局部损失敏感度，不是屏幕梯度，也不是重建误差。
统计使用总损失的 xyz 梯度，仍可能发生视图间抵消；受损失权重、切片采样数和插层阶段影响。
"可见"指候选半径 > 0 且强度梯度非零，即本步实际参与像素合成；`denom` 是满足该条件的训练迭代次数。
仅当本统计周期 `denom >= structural_min_observations` 才允许增殖。

梯度门限 `T = max(structural_grad_threshold, T_frozen)`，`T_frozen` 在首个结构周期取
`structural_grad_multiplier * median(g_i > 0)` 后冻结并随 checkpoint 保存。
相对中位数门限选中的比例只由梯度分布形状决定，与拟合好坏无关：patient001 实测每周期稳定选中
29%–34%，点数每周期约 ×1.25，损失几乎不再下降。因此只用它标定一次。

每个结构周期全病例渲染一次做贡献审计。渲染对颜色线性，`dI_p/dc_i = T_i*alpha_i`，故：

- 贡献质量 `W_i = sum_p T*alpha`（像素当量），与颜色无关；
- 覆盖区平均残差 `E_i/W_i = sum_p T*alpha*|r_p| / W_i`（归一化强度单位）。

增殖要求 `g_i >= T`、`E_i/W_i > structural_residual_threshold` 且 `W_i >= structural_support_min_weight`。W 不用于删除。
误差降到 tau 以下的区域不再增殖，点数由剪枝收敛到平衡值。tau 是"可接受误差"旋钮，越小最终点数越多。

面内尺度使用 `D^-1 R diag(sigma^2) R^T D^-1` 的 XY 子块最大特征值的平方根，
其中 D 的 XY 分量为输出像素间距。实现使用 2×2 闭式公式，不做逐点 eigh。
它用于启发式区分大/小点，不能解释为精确的成像分辨率。

## 默认值

| 参数 | 默认 | 含义 |
|---|---:|---|
| structural_grad_multiplier | 2.0 | 首周期标定冻结门限的中位数倍数 |
| structural_grad_threshold | 0.0 | 绝对梯度下限，与冻结值取大 |
| structural_residual_threshold | 0.005 | 覆盖区平均残差门限，约为 ACDC 残差下限的 3 倍 |
| structural_support_min_weight | 0.5 | 增殖支撑门：贡献质量下限（像素当量），只门控增殖，不删除 |
| structural_min_observations | 8 | 一个统计周期内最低可见迭代数 |
| structural_split_sigma_pixels | 2.5 | 候选点面内标准差大于 2.5 个输出像素才 Split；小点不增殖（无 Clone） |
| structural_prune_after_densify | false | true 时增殖窗口后继续审计（剪除框外点、拆分尺寸超限点）；默认 densify_until_iter=10000 后高斯数目定型 |
| structural_size_split_from_iter | 3000 | 从该步开始启用尺寸超限强制 Split |
| structural_size_max_screen_radius | 20.0 | 本周期最大候选半径的像素上限，超过即拆分；0 关闭 |
| structural_size_max_scale_ratio | 0.1 | 面内投影标准差 / max(bounds) 的上限，超过即拆分；不限制法向拉长；0 关闭 |
| structural_prune_peak | 0.0 | 必须为零，不按颜色强度剪枝 |
| min_sigma_pixels | 0.5 | 任一局部轴标准差下界（面内像素） |

ACDC 缩放到 128 后面内约 1 mm/px，层间距 5–10 mm，心肌厚度 p10/中位数 5.6/7.5 mm。
`sigma=2.5` 对应约 5.9 像素 FWHM，与较薄心肌相当。推导见 [结构算法实验记录](stage1_structural_experiments.md) 第 9 节。
这些是由数据特性推得的理论起点，尚未在完整训练中逐项验证。

## 如何校准

1. 在一个有代表性的病例上跑完整训练，观察每周期日志 `selected/split/split_size/pruned_box/
   residual_high/grad_threshold/weight_median/error_median` 与点数曲线。预期点数先升、增速衰减、最后平稳。
2. 检查观测数。随机均匀采样时，只被一张切片看见的点在长度 K 的窗口内期望约有
   `K * min(real_slices_per_iter,S) / S` 次观测；K=200、S=10、每步 2 张约 40 次，默认 8 次可作为起点。
3. tau（`structural_residual_threshold`）决定平衡点数：误差不再改善而点数仍增长时增大 tau；
   残差明显但候选极少时降低 tau 或检查冻结梯度门限。若每周期新增与剪枝都很大且相互抵消，说明 tau 低于可达误差。
4. `structural_support_min_weight` 以像素当量计；`min_sigma_pixels=0.5` 时单层未遮蔽最小核约 1.5 像素当量，0.5 不会挡住合法细点的增殖。
5. 尺度阈值按输出像素设置，输出分辨率、渲染定义或图像 resize 改变后需重新校准；不用层间距替代面内像素尺度。
6. 不按颜色幅值裁剪；W 与颜色无关，暗色组织只要参与合成就保留。若剪枝后点数少于 structural_min_points，在提交前报错并保留原模型。

## 验证标准与限制

比较同一配置、相同随机种子下的最终点数、全病例 L1/SSIM 与结构操作后的损失跳变；插层效果另用留层验证比较。
Split 子点直接继承局部强度，不均分或补偿颜色，不保证混合输出不变；结构调整允许短期损失上升。
梯度统计仍可能存在视图间抵消（可参考 AbsGS 的逐像素绝对梯度），当前未实现。
当前不实现独立 Birth、Merge、Swap 或结构目标验收。
