# 对比实验（comparative）

在 C3DGS 的 `Dataloader.dataset(dim='3D')` 数据上，以统一协议逐病例运行 GaussianSVR、GSVR、NeSVoR、ECLARE、Dyna3DGR。
参考实现位于 `references_root`（默认 `/home/exouser/codes/References`），本目录只含适配代码，不修改参考仓库。

## 环境

使用 `3dgs` 环境（torch 2.5.1+cu124）。已安装的额外依赖：faiss-gpu-cu12、pytorch3d 0.7.9、simple-knn、nesvor、eclare、
kornia、pytorch_lightning、tensorboard、monai、SimpleITK、scikit-learn、scikit-image、antspyx、radifox-utils 1.0.3。
ECLARE 声明的 `scipy==1.11.4` 未安装（与 numpy 2 不兼容），沿用环境中的 scipy 1.15.3。
GaussianSVR 与 Dyna3DGR 的 CUDA 扩展在首次运行时 JIT 编译，需要 `CUDA_HOME` 指向含 nvcc 的环境：

```bash
export CUDA_HOME=$HOME/miniconda3/envs/3dgs PATH=$HOME/miniconda3/envs/3dgs/bin:$PATH TORCH_CUDA_ARCH_LIST=8.0
```

## 运行

在 `C3DGS/comparative` 目录或 C3DGS 根目录执行均可：

```bash
python comparative/training.py   --config comparative/config.yaml --method gaussiansvr   # gsvr / nesvor / eclare
python comparative/training.py   --config comparative/config.yaml --method dyna3dgr --stage stage1
python comparative/training.py   --config comparative/config.yaml --method dyna3dgr --stage full
python comparative/evaluation.py --config comparative/config.yaml --method all
```

`--protocol observed|holdout`、`--cases`、`--iters` 可覆盖配置。每例输出 `<output_root>/<protocol>/<method>/<case>/`：
模型文件、`prediction.npz`、`run.json`（实际配置、种子与用时）、评估生成的 `comparison.png`；
评估另写 `<method>/metrics.csv`（逐层）与 `<protocol>/summary.csv`（病例内均值后的病例间均值±标准差）。

## 统一协议

- 静态方法拟合 `reference_phase`（默认 0）；Dyna3DGR stage1 固定为初始帧 t=0。
- `observed`：全部层训练、全部层计分（拟合度）。`holdout`：`k % holdout_stride == 0` 的层训练，
  只对夹在训练层之间的留出层计分，外推层不计入。两种协议的结果分目录保存。
- 几何与 C3DGS 相同：像素中心 `(u+.5-W/2)·sx`，第 k 层 `z=(k-(S-1)/2)·sz`；层面轮廓 FWHM = `slice_thickness_fwhm_spacing·sz`。
- 预测在训练结束时由同一模型、按各方法自身前向模型（PSF/层面轮廓）在名义层位渲染，
  覆盖 0..S-1、步长 `interp_step_size`；评估只读预测文件，与方法实现解耦（各参考仓库的同名顶层包无法在同一进程共存）。
- 逐病例迭代方法统一使用 `iters`；方法内部按绝对步数写死的里程碑按原比例缩放，其余超参数保持预设。
- 运动校正全部关闭；所有方法均不抠图。指标定义与 `config_test.yaml` / `evaluate.py` 相同。

## 各方法适配要点（与原实现的差异）

| 方法 | 迭代映射 | 主要差异 |
|---|---|---|
| GaussianSVR | `max_iter = iters` | 不运行 SVoRT（单 stack、无适用权重），初始体为训练层 z 向线性插值；逐层位姿固定为名义几何 |
| GSVR | `max_epochs = iters`（全批量） | 复用原 `train.train()`，仅替换 `load_data` 与末尾导出；采用真实数据预设；`mini_batch_size=null`（原值大于单例体素数会断言失败） |
| NeSVoR | `n_iter = iters` | 每层一个 `Slice`（名义位姿、`resolution_z` = 层厚）；不配准/分割/偏场校正；输出掩膜改为全视野 |
| ECLARE | 训练步数 `= iters` | 内存输入、层厚取统一设置（不运行 ESPRESO）；输出先按层面轮廓模糊再在名义层位采样 |
| Dyna3DGR | stage1 `= iters`；full 为原 20000 步 | 原 `main()` 拆为静态阶段与联合阶段（见 `methods/dyna3dgr.py` 文档）；补上发布代码缺失的 `progressive_brand_time` 默认值 |

Dyna3DGR full 的联合阶段默认使用 ED 到 ES 的连续帧（`frames: ed_to_es`，ACDC 6–16 帧，ED 为时间原点）；
也可选 `ed_es`（原方法的两帧）或 `all`（完整序列）。每步渲染所有所选帧，A100 上约 0.012 s/帧，
`ed_to_es` 约每例 35 分钟，`all` 约 1.5 小时。stage1 按约定固定在初始帧 t=0，与 ED 不同的病例（ACDC 仅 1 例 ED=3）
第二阶段从 ED 开始。
