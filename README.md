# C3DGS 心脏 MRI 高斯重建

项目按病例独立优化。Stage1 拟合参考相位的固定 opacity=1、单通道颜色强度三维高斯；Stage2 读取参考高斯，拟合完整序列的演化网络。测试入口仅评估 Stage1，不更新模型。

## 环境

需要 CUDA 版 PyTorch、兼容的 CUDA Toolkit/nvcc、NumPy、PyYAML、plyfile、torchvision、matplotlib、tqdm 和 simple-knn。`temp.py` 额外需要 nibabel；独立使用 `tools.py` 的可视化/指标功能需要 OpenCV、python-pptx、scikit-learn、Pillow。本仓库不自动安装或修改环境。

Stage1 使用 `diff_gaussian_rasterization_slice` 的三维高斯切片与 alpha 合成。Python 包装与原生扩展须配套（`API_VERSION=2`、`RENDER_MODE=alpha`）；旧二进制会被明确拒绝。主工程正常导入安装包，不自动安装或替换包。

在已激活、含 CUDA PyTorch 的环境内安装修改后的库：

```bash
python -m pip install --no-build-isolation --no-deps --force-reinstall ./diff-gaussian-rasterization_slice
python -c "import diff_gaussian_rasterization_slice as r; print(r.API_VERSION, r.RENDER_MODE)"
```

从项目根目录执行，安装后重启已有 Python 进程。编译目标按当前可见 GPU 或 `TORCH_CUDA_ARCH_LIST` 选择，已移除固定 sm_86；跨机器安装应显式配置目标架构。此命令仅重装渲染包。

## 配置和运行

`config_train.yaml` 包含两个阶段的训练参数和显式预处理参数；`config_test.yaml` 只包含 Stage1 测试所需参数。旧 `config.yaml` 已移除。

首先修改配置中的数据路径、输出路径和 `device`。所有相对外部路径以配置文件所在目录为基准，支持环境变量和 `~`。`device: cuda:0` 表示当前可见设备中的第 0 张卡，代码不会覆盖外部 `CUDA_VISIBLE_DEVICES`。

```bash
python GetInitialGaussian.py --config config_train.yaml
python Motion.py --config config_train.yaml
python evaluate.py --config config_test.yaml
```

保留 Stage2 的 CLI 覆盖，例如 `--control-deformation-mode mls`、`--slice-training-mode random_slice`。

默认数据集为 ACDC，训练读取 `dataset_mode: all`，测试读取 `val`；这是逐病例重建评估，不代表对未见病例的泛化。主入口显式使用读取器的 `dim='3D'` 完整病例接口。没有对应 PLY 的病例会明确报错，不会在测试中重新训练。`all/train/val` 继续按排序后的全部病例及 `train_fraction` 选择集合；新增的 `test` 根据 NPZ 的 `original_split` 选择官方测试病例。这两种划分方式不同，不保证 `train/val` 与官方 `test` 互斥。评估入口始终排除无有效标签的病例，`test` 模式也执行相同过滤。

测试的 `stage1_ply_path` 使用 `{case}` 替换 NPZ 文件主名。输出逐切片 L1、PSNR、SSIM 到 `metrics.csv`，可选保存预测数组和固定显示范围的 PNG。默认指标使用未截断预测；`clamp_predictions` 可显式切换。测试配置的归一化、坐标边界和参考相位应与生成 PLY 时一致。

## 数据契约

当前读取器仅适配 ACDC（`datasets='acdc'`）和 M&Ms（`datasets='mnms'`，也接受 `mm/mms/m&ms`）。配置中的 `dataset_name` 使用 `acdc` 或 `mnms`，`dataset_paths` 已指向 `/home/exouser/Datasets` 下的两个 NPZ 目录，也可在构造读取器时通过 `data_path` 覆盖路径。

NPZ 包含 `cmr: [T,S,H,W]`、`labels: [P,S,H,W]` 和 JSON 字符串 `info`。`dataset(..., dim='3D')` 默认返回完整病例：`cmr: [S,T,128,128]`、`label: [S,P,128,128]`；`dim='2D'` 遍历各病例的所有切片，返回 `cmr: [T,128,128]`、`label: [P,128,128]`。先按病例划分集合，再展开切片。图像按完整病例执行 min–max 归一化，输出为 `[0,1]` 的 float32；标签为 long。仅将 H/W 缩放为 128×128，图像使用双线性插值，标签使用最近邻插值，空标签保留空相位轴。保留全部切片和时间帧，不对 S/T 插值或补齐，因此默认使用 `batch_size=1`。`info.source_spatial_shape` 和 `info.spatial_shape` 记录缩放前后尺寸；原始 JSON 的 affine、spacing 仍描述缩放前的数据。

对外统一类别为 ACDC 定义：`0=背景、1=右室腔、2=左室心肌、3=左室腔`；M&Ms 原始编号 1、3 在读取时交换。`P` 仅包含实际标注相位，ED/ES 对应时间帧位于 `info['time_indices']`，无标签时 `P=0`，不会补成背景。M&Ms 官方标记为 unlabeled 或相位元数据无效的病例也会被测试过滤，即使文件中存在标签数组；全背景标签不视为无标签。其他模式可显式指定 `require_labels=True` 使用同一过滤规则。

输出固定为 `cmr、label、info`。`info` 为字典，包含病例标识、文件名、数据集、原始集合、切片索引（3D 时为 -1）、轴序、标注相位、有效性以及统一类别定义和映射；原始元信息保存在 `info['metadata']` JSON 字符串中，避免其中的 null 和异构字段影响默认 DataLoader 拼接。原始类别定义保留在该 JSON 中。新增数据集时，在读取器的适配位置扩展类别映射、字段和轴序处理，共用后续索引及输出逻辑。

读取器输出轴序固定，训练和评估入口不再依据旧的 `cmr_layout` 配置重复转置。`reference_phase` 默认为 0；Stage2 当前要求参考相位为 0，不自动改选 ED。

几何统一使用 `source.files.cmr_4d.zooms[:3]` 的 XYZ spacing，不比较或使用 affine。
crop 不改变 spacing；从实际 NPZ crop 尺寸 Hc×Wc 缩放到 128×128 后，
`spacing_xyz=[sx*Wc/128,sy*Hc/128,sz]`，`source_spacing_xyz` 保留缩放前值。
两个字段均为 CPU float64 Tensor，2D/3D 接口相同；原 JSON 保留。
spacing 按本数据集的毫米约定解释，未知单位仍保留在原始元信息中，不声称恢复患者绝对位置。

旧 `InitializeGaussians` API 保留 ROI 兼容入口；当前 Stage1 主入口使用参考相位图像生成采样权重，不额外读取分割标签。

新模型在 spacing 定义的局部栈坐标中居中，三个轴共用 `L=max(W*sx, H*sy, max(S-1,1)*sz)/2` 缩放，病例 bounds 自动计算并同时传给初始化、Renderer 和结构优化。
面内使用像素中心 `(u+.5-W/2)*sx`，法向使用 `(k-(S-1)/2)*sz`；单层观测位于零平面。
配置中的 `bounds` 仅定义 legacy_box 坐标边界；加载模型始终要求匹配的旁置渲染与几何记录。
Stage1 渲染为 `I=sum(T_i*alpha_i*c_i)+T_end*b`，`alpha_i=min(0.99,G_i)`、`T_i=prod_{j<i}(1-alpha_j)`，按高斯中心到切片的绝对距离排序。
`c_i` 是颜色强度，opacity 固定为 1，不进入优化器或 PLY；这不表示每个像素的 alpha 均为 1。
库沿用 alpha 小于 `1/255` 的截断、透射率提前终止和 tile 候选范围，均为分段可导近似；不承诺跨排序、候选和截断边界的连续梯度。
渲染模型标识为 `innergs_alpha_fixed_v1`，不接受外部 `opacities` 或旧 `response_mode/psf_sigma/psf_std_vector/kernel_cutoff/cutoff_transition/covariance_jitter` 配置。
模型参数保留 spacing 坐标；Renderer 用 `A=diag(1/bounds)` 变换位置和完整协方差后传入新库，
保持像素中心、连续切片索引和单层位置定义。灰度适配为三通道，返回单通道图像。

初始化默认 1000 点（`num_pts`），50% 分层均匀、30% 图像前景、20% 图像边缘；采样分位数图不改变训练目标。
初始尺度按经典 3DGS：各向同性 sigma=sqrt(到最近 3 个中心平方距离的均值)，下界为 `min_sigma_pixels` 个面内像素，旋转从单位四元数开始；
不为层间覆盖单独设置法向尺度。无像素贡献的点重采样后按最终位置重算尺度。
固定几何时 alpha 合成对颜色仍为线性算子 A；先按 `b*(1-A(1))` 扣除剩余透射率背景，再标定共同强度并用最多 20 步矩阵自由 CG 拟合颜色，不构造全局响应矩阵，不增加正常 Adam 预热阶段。
CG 预算耗尽不等于收敛，初始化报告包含相对残差和拟合前后 MSE。病例级采样使用独立稳定种子，不消耗训练的全局随机数流。

## 保存和恢复

Stage1 每例保存（只保留最终结果）：

- `point_cloud_final.ply`：用于模型交换的参考高斯；不再保存中间步数的 PLY。
- `point_cloud_final.ply.geometry.json`：spacing、病例 bounds、缩放因子、参考相位、归一化、渲染条件、固定几何正则参考容量、迭代号、PLY 哈希，以及 `training`（`slice_protocol`、训练层 `observed_slices`、`elapsed_seconds`）；必须随 PLY 一起复制。
- `reconstructions/final.png`：`visualize: true` 时训练结束保存一张全病例拼图，每个真实层"原图 | 渲染"并排并标注该层 PSNR/SSIM，同一行再接按 `interp_step_size` 渲染的插值层（蓝色标注，无原图）；holdout 下未参与训练的层红色标注，标题分列训练层与留出层平均指标。
- `loss_curves.png`：各损失随迭代的曲线，每 `loss_plot_interval` 步覆盖更新。
- `structure_log.csv`：每 `structure_log_interval`（默认 500）步一行，汇总窗口内各结构周期的 split、split_size（尺寸超限拆分）、pruned/pruned_box、selected、residual_high（残差门放行）、eligible 之和，以及最后一次的梯度门限与 W、E/W、面内 σ 中位数和当前点数；续训时截掉超过起点的行。
- `training_state.pth`：参数、Adam 动量、迭代次数、增殖统计、随机状态、损失历史和累计用时；只在训练中每 `checkpoint_interval` 步写出用于中断续训，最终 PLY 写出后删除。

默认输出根目录为 `/home/exouser/codes/OutPut/C3DGS`，每例写入 `<根目录>/<病例名>/<output_case_subdir>/`（子目录默认为空）。

层监督协议 `slice_protocol`：`observed` 用全部层；`holdout` 只用 `k % holdout_stride == 0` 的层，初始化、损失、插层包络与结构统计均不读取留出层，几何仍按完整层栈，插层区间取相邻两训练层之间（仅支持 `additive_moment_v1`）。

默认从完整训练状态恢复；最终 PLY 记录的迭代数达到 `iters` 时跳过病例。`warm_start_ply_path` 仅用于模型参数热启动，不表示完整续训。恢复的随机状态要求使用相同数据、设备布局和实验参数；CUDA 原子累加仍可能产生非确定性。
新 checkpoint 保存相同几何记录；恢复、测试和 Stage2 读取时核对坐标/观测条件。
Stage1 拒绝将旧加性 checkpoint 或无语义记录的 PLY 自动解释为局部强度；新记录包含 `render_model`、`intensity_semantics`、`opacity_mode=fixed_one` 和 `rasterizer_api_version=2`。新最终 PLY 记录缺失时，可通过匹配的完整 checkpoint 重建产物。
续训允许延长迭代数和改变保存频率，拒绝混用已改变的 Stage1 优化条件；初始化不会在续训时重新执行。
评估新初始化应使用独立输出目录，或明确关闭原实验的恢复/跳过选项。

新 PLY 字段为 `x,y,z,intensity,scale_0..2,rot_0..3`；尺度保存为 log 标准差，旋转保存原始 wxyz 四元数。字段读取器保留 `legacy_color_policy`，但含旧 `opacity` 字段的 PLY 会被拒绝，不能通过丢弃 opacity 宣称模型等价；训练、评估及 Stage2 还必须核对匹配的渲染语义记录。

Stage1 默认真实切片无放回均匀采样，支持单层/双层病例。真实切片损失权重不变；插层从 `interpolate_iter=5000` 起、早于增殖结束，使层间误差也能驱动结构调整。
层间没有真值，插层不再拟合像素级线性混合（结构移动处会产生双重轮廓），改为预测落在相邻两层强度之间的区间损失 `lambda_interp_envelope`；
未命中插层的分支仍额外监督一张真实切片。当前不为层间覆盖设置或限制高斯沿法向的拉长（层间覆盖损失和法向尺度上下界均已取消）。
初始化与剪枝使用同一观测框 `[W*sx,H*sy,S*sz]/2`（每层代表厚度 sz 的薄片）；渲染坐标 bounds 不变。
尺度下界为 `min_sigma_pixels=0.5` 个面内像素，`log_scale_min/max` 只作模型坐标下的数值安全边界。不设点数上限或每周期增长预算，也不按颜色剪枝。

Stage1 使用独立的 `Gaussians/GaussianModel_structural.py`，不继承或导入 `GaussianModel_noop.py`；
属性、PLY、Adam、学习率和初始化入口由本模型维护。评估与 Stage2 统一读取本模型，字段格式一致不代表旧渲染参数可直接互换。
结构周期沿用 `densify_from_iter / densify_until_iter / densification_interval`。
正常训练采样、损失、反向传播和 Adam 更新次序不变；几何正则继续使用固定参考容量 N0。

结构操作采用 `structural_algorithm: batch_residual_v5`，目标是增殖速度随误差下降而收敛、点数最终平稳：

- 每个结构周期全病例逐层渲染一次做贡献审计：渲染对颜色线性，两次颜色反向得到每点贡献质量 `W=sum(T*alpha)` 与贡献加权残差 `E=sum(T*alpha*|r|)`（像素当量），与颜色无关。
- 增殖同时要求：按体素长度换算的平均位置梯度不低于冻结的绝对门限（首个周期以 `structural_grad_multiplier` 倍正梯度中位数标定后固定，随 checkpoint 保存），以及覆盖区平均残差 `E/W > structural_residual_threshold=0.005`。
  旧算法每周期按相对中位数门限选出固定约 30% 的点，导致点数按周期指数增长；新判据在误差降到门限以下后不再增殖。
- 只拆分面内投影标准差（输出像素单位）大于 `structural_split_sigma_pixels=2.5` 的候选点（ACDC 约 1 mm/px，心肌厚度 p10/中位数 5.6/7.5 mm）；不做 Clone。
  patient001 消融中，opacity 固定为 1、按深度排序时克隆副本多被父点遮蔽，只增加冗余点与 z 向跳变，只分裂的精度与插层指标最好（见 `docs/stage1_structural_experiments.md` 第 16–17 节）。
- Split 按经典 3DGS 在父点局部坐标按三轴标准差采样两个子点并删除父点，所有轴尺度除以 `0.8*2=1.6`；子点任一轴低于尺度下界时不拆分。
- 尺寸超限强制 Split：从 `structural_size_split_from_iter=3000` 起，本周期最大候选半径超过 20 像素，或面内投影标准差超过 0.1 倍 `max(bounds)` 的点
  不经梯度/残差门控、在增殖窗口后也一律按经典 Split 拆成两个 1/1.6 尺度的子点（针状点最短轴截到尺度下界），子点仍超限则下个周期继续拆。
  不再删除过大点：删除会在均匀亮区留下无父点可补的空洞（实测 patient001 第 8 层亮带点数 896→188、PSNR 降到 23.5 dB）。只看面内，不限制沿法向的拉长。
- 剪枝：无效点；中心超出观测框半个体素以上的点（含新子点）。不按贡献质量删除：W 只统计真实层，层间点可能服务插层渲染。
  W 仍用作残差门的分母，以及增殖支撑门 `W >= structural_support_min_weight=0.5`（避免小数相除的噪声）。
  `structural_prune_peak` 仍必须为 0。
- 增殖统计的可见性为"候选半径 > 0 且强度梯度非零"，即本步实际参与像素合成，包含插层渲染。
- `densify_until_iter` 后若 `structural_prune_after_densify=true`，按同一间隔继续审计：剪除框外/无效点，尺寸超限点仍拆分，不做误差驱动增殖。所有合格候选均参与增殖，不随机限流或按 Top-K 排序；`num_pts` 只决定初始点数。

每周期基于同一旧模型生成全部掩码和子点，各参数/Adam 状态统一提交一次。
保留未改旧行动量，新行动量清零，组级 Adam step 保留。
提交后清空梯度/观测统计及 `max_radii2D`；后者在训练渲染后累计本周期最大候选半径。
旧 `batch_classic_split_v1`/`batch_normalized_v1` 检查点不能作为同算法直接续训；恢复检查会拒绝结构版本不一致。
没有逐候选 Python 循环、Birth/Merge/Swap、局部拟合、SSIM 验收或回滚；唯一的额外渲染是每周期一次的贡献审计。
结构调整允许短期损失上升，不保证后续迭代能够补偿全部近似误差。

旧加性或归一化模型不能作为新 alpha Stage1 的精确续训或直接颜色热启动；默认输出目录改为独立的
`acdc_experiment_alpha_v4`，旧 `acdc_experiment_alpha`、`_v2`、`_v3` 结果保留。`warm_start_ply_path` 要求匹配的新语义及几何记录。
新版 checkpoint 保持 Adam、随机状态和统计的完整恢复。阈值定义及校准方法见
[批量高斯阈值说明](docs/batch_gaussian_thresholds.md)。

Stage2 仅调整模型导入和加载协议，不重建运动网络。当前用户的 `Gaussians/GaussianEvolution.py` 是占位文件；保留此内容，按需构造网络时明确报错，因此尚不能运行或验证完整 Stage2。checkpoint 仍按传播方式和切片训练模式分别保存。
新模型的 Stage2 渲染及稠密查询网格使用保存的病例 bounds；像素位移换算同步更新，旧模型保留旧导出约定。
Stage2 checkpoint/运动结果保存几何记录，避免将旧坐标结果误用为新坐标；不匹配时使用独立实验目录。

## 可选预处理

```bash
python temp.py --config config_train.yaml
```

只有显式运行才访问数据。`preprocess_*` 配置控制输入/输出、时间轴、目标帧数和归一化；默认保持原 `slice_uint8` 协议，设为 `none` 可保留原浮点强度。NPZ 同时保存源 affine、spacing、原始帧数等元数据。默认拒绝覆盖已有病例文件。

## 验证

```bash
python -m unittest discover -s tests -p test_stage1_normalized.py -v
python -m unittest discover -s tests -p test_structural_optimizer.py -v
python -m unittest discover -s tests -p test_initialization.py -k InitializationTests -v
```

原生 CUDA 回归在渲染库目录执行：

```bash
python -B -m unittest discover -s diff-gaussian-rasterization_slice/tests -v
```

测试临时二进制时可设置 `SLICE_CUDA_EXTENSION=/absolute/path/to/_C_G3D.so`，主工程和原生测试共用此显式测试入口，不覆盖安装包。
本轮验证 Stage1、评估及运动导入边界；历史 `test_slice_response.py` / `test_rotation_backward.py` / `test_renderer_safety.py` 还包含另一套 `diff_gaussian_slice_rendering` 库的测试，不能用于证明本库的数值正确性。

测试区分不依赖 GPU 的接口/配置回归和需要 PyTorch/CUDA 扩展的数值测试；依赖不具备时后者跳过。语法和接口检查不能替代 CUDA 编译、GPU 数值验证或完整训练。
