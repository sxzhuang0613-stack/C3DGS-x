# diff_gaussian_rasterization_mri 使用说明

由经典 3DGS 光栅化库改写而来的 MRI 切片渲染器。它沿每张切片自己的法向做有限层厚积分，各高斯的信号直接线性相加。这里没有透视投影，没有 SH，也没有 α 合成与遮挡。

## 1. 成像模型

**切片坐标系。** 切片位姿为 (R, o)。R 的三行依次是 u、v、n：u 是面内列方向，v 是面内行方向，n 是法向。高斯在切片坐标系下为：

```
μ' = R(μ − o)
Σ' = R Σ Rᵀ = [[A, b], [bᵀ, c]]
p  = μ'_n                         # 高斯中心到切片平面的法向距离
```

**层面轮廓与闭式积分。** 层面轮廓取归一化高斯 N(n; 0, s²)，沿法向积分后得到闭式解：

```
q      = c + s²
中心   m = μ'_uv − b·p/q                 （秩 1 修正）
协方差 M = A − b bᵀ/q
幅值   a = sqrt(c_cond/(c_cond+s²)) · exp(−p²/(2q))，  c_cond = c − bᵀA⁻¹b
```

**像素值。** 把 m、M 换成像素单位：mean_px = m/(du, dv)，conic = (D M D)⁻¹，其中 D = diag(1/du, 1/dv)。每个像素的值为

```
C(p) = bg + Σ_i I_i · a_i · exp(−½ dᵀ conic_i d),   d = mean_px,i − p
```

**两个极限。** 这两点都由测试验证：
- s = 0 时，结果正好是零厚度截面值 I·G3D。
- s → ∞ 时，a·s → sqrt(c_cond)。

**强度的含义。** 强度 I 是峰值信号密度，不随 s 变化，没有量纲。

## 2. 编译与安装

```bash
conda activate 3dgs
cd diff-gaussian-rasterization
pip install --no-build-isolation --no-deps .
# 或仅本地测试：python setup.py build_ext --inplace --force，然后在本目录设置 PYTHONPATH=.
python -c "import diff_gaussian_rasterization_mri as m; print(m.API_VERSION, m.RENDER_MODE)"
```

改动头文件（`.h`）后要加 `--force`。没有安装 ninja 时，distutils 不跟踪头文件依赖，不加的话改动不会被重新编译。

## 3. 坐标与单位约定

- **单位一致**：位姿、像素间距、`slice_sigma`、高斯的位置和尺度必须使用**同一个长度单位**，而且三个方向的单位相同。库本身不关心单位是什么。
- **不要按轴分别缩放**：坐标系只要是直角坐标系即可，不需要是世界坐标。如果把三个轴分别缩放（例如按 bounds 归一化），R 就不再正交，层厚也会被扭曲，所以不要这样做。
- **像素中心的约定**：像素 (row=i, col=j) 的中心位于 `o + j·du·u + i·dv·v`。输出图像下标为 `image[0, i, j]`。
- **R 必须正交**：库在输入时检查 `R Rᵀ ≈ I`，容差 1e-4。

不同数据来源下，位姿的构造方式：

| 数据 | 构造方式 |
|---|---|
| DICOM | `u` = ImageOrientationPatient[0:3]（列索引增大的方向），`v` = IOP[3:6]，`n = u × v`；`o` = ImagePositionPatient，即第一个像素中心；`du` = PixelSpacing[1]（列间距），`dv` = PixelSpacing[0]（行间距）。注意 PixelSpacing 的顺序是先行后列。 |
| NIfTI affine | affine 中对应图像列方向的那一列 = du·u，对应行方向的那一列 = dv·v；o = affine · (第 k 层第一个体素, 1)。数组轴和图像行列的对应关系由数据加载代码决定，需要调用方核对。 |
| 只有 spacing | 用 `stack_slice_pose`（图像栈局部坐标）。这种方式假设所有切片平行、等间距、面内没有错位，因此不能和其他方向的切片联合使用。 |

**短轴与长轴联合。** 只要所有切片的位姿都在同一个坐标系中（来自 DICOM 或可信的 affine），短轴和 2CH、3CH、4CH 切片可以用同一组高斯，每张切片各自调用一次渲染，再把损失加起来。各切片的像素间距、层厚和尺寸可以不同。

## 4. 主要接口

### `SliceRasterizationSettings`（NamedTuple）

| 字段 | 类型 | 说明 |
|---|---|---|
| `image_height`, `image_width` | int | 输出图像尺寸 H、W |
| `slice_rotation` | Tensor [3,3] | 行依次为 u、v、n，必须正交 |
| `slice_origin` | Tensor [3] | 像素 (0,0) 中心的坐标 |
| `pixel_spacing` | (du, dv) | 沿 u、v 方向的像素间距 |
| `slice_sigma` | float ≥ 0 | 层面轮廓的标准差 s。**必填**：0 表示零厚度；有层厚时取 `FWHM × FWHM_TO_SIGMA`，即 FWHM / 2.355 |
| `bg` | float = 0 | 加到整幅图像上的常数背景 |
| `scale_modifier` | float = 1 | 尺度整体乘的系数，与经典库相同；仅在使用 scales 时生效 |
| `amplitude_cutoff` | float = 1/255 | 截断阈值 ε，见第 6 节 |
| `debug` | bool = False | 出错时保存输入快照，与经典库相同 |

### `SliceRasterizer(settings)(means3D, means2D, intensities, scales=None, rotations=None, cov3D_precomp=None)`

| 输入 | 形状 | 说明 |
|---|---|---|
| `means3D` | [N,3] | 高斯中心 |
| `means2D` | [N,3] | 与 `means3D` 同形的零张量，设 `requires_grad=True`，只用来接收面内中心梯度，和经典库一样 |
| `intensities` | [N,1] | 峰值信号密度，可以为负；是否约束非负由调用方决定 |
| `scales` + `rotations` | [N,3] + [N,4] | 四元数顺序为 (w, x, y, z)。**库内不归一化**，调用方需先归一化 |
| 或 `cov3D_precomp` | [N,6] | 对称矩阵的 6 个元素，顺序为 (xx, xy, xz, yy, yz, zz) |

所有输入都必须是 CUDA float32 张量。协方差的两种给法二选一。

| 输出 | 形状 | 说明 |
|---|---|---|
| `image` | [1,H,W] | 渲染结果 |
| `radii` | [N] int32 | 覆盖半径，单位为像素；0 表示该高斯不参与本切片的渲染，也不会得到梯度 |
| `amplitudes` | [N] | 几何幅值 a，不可求导；被剔除的高斯为 0 |

### `stack_slice_pose(spacing_xyz, slice_coord, device='cuda')`

在只有 spacing 时构造位姿，返回 `(R=I, o=(0, 0, t·dz), (dx, dy))`。`slice_coord` 可以取小数，用于渲染层间的插帧位置。

### 常量

`FWHM_TO_SIGMA`（= 1/2.3548）、`API_VERSION`（= 1）、`RENDER_MODE`（= `'additive_slab'`）。

## 5. 示例

```python
import torch
import diff_gaussian_rasterization_mri as mri

spacing = (1.25, 1.25, 8.0)          # (dx, dy, dz)，单位 mm
thickness_fwhm = 8.0                 # 层厚：无元数据时由调用方假设（例如 = 层间距）
R, o, (du, dv) = mri.stack_slice_pose(spacing, slice_coord=3)   # 第 3 层
settings = mri.SliceRasterizationSettings(
    image_height=128, image_width=128, slice_rotation=R, slice_origin=o,
    pixel_spacing=(du, dv), slice_sigma=thickness_fwhm * mri.FWHM_TO_SIGMA)
rasterizer = mri.SliceRasterizer(settings)

means2D = torch.zeros_like(xyz, requires_grad=True)
image, radii, amps = rasterizer(xyz, means2D, intensity, scales=scales,
                                rotations=torch.nn.functional.normalize(quats, dim=-1))
loss = torch.nn.functional.l1_loss(image[0], target)
loss.backward()
grad_px = means2D.grad[:, :2]        # 面内中心梯度，单位为像素，可用于致密化统计
visible = radii > 0
```

斜切片，例如来自 DICOM 的长轴切片：

```python
u, v = torch.tensor(iop[:3]), torch.tensor(iop[3:])
R = torch.stack([u, v, torch.linalg.cross(u, v)])
settings = mri.SliceRasterizationSettings(H, W, R, torch.tensor(ipp), (ps[1], ps[0]),
                                          slice_sigma=slice_thickness * mri.FWHM_TO_SIGMA)
```

**层间插帧。** 例如 `slice_coord=3.25`。`slice_sigma=0` 渲染的是 3D 表示本身的截面，更清晰；`slice_sigma` 取采集层厚时，模拟的是"如果在这个位置扫一张会得到什么"。用哪一种由调用方决定。

## 6. 截断、可见性与数值行为

**截断支撑。** 像素 p 只有同时满足下面两个条件，才计入高斯 i 的贡献：
- 马氏距离 ≤ 3，即 power ≥ −4.5；
- w = a·exp(power) ≥ ε。

这等价于马氏半径 k = min(3, sqrt(2 ln(a/ε)))，所以幅值越小，覆盖的像素越少。前向和反向使用完全相同的判据。

**剔除。** 只看几何幅值 a：a < ε 的高斯被剔除。剔除时**不乘强度**，这样强度接近 0 的高斯依然能拿到强度梯度。

**面内模糊。** 不像经典库那样给二维协方差加 0.3 px² 的抗锯齿项。在 s = 0 时，比一个像素还小的高斯可能落在像素中心之间而采不到。采集分辨率对应的 PSF（S4）尚未实现。

**确定性。** tile 分桶使用稳定排序，前向结果逐位确定。反向和经典库一样使用 atomicAdd，梯度中存在浮点累加顺序带来的微小差异。为减少原子操作，反向先在 warp 内归约，再由每 32 个像素执行一次原子加。

**梯度约定。** `cov3D_precomp` 的梯度把非对角元当作独立参数，与经典库相同。`means2D` 的梯度是像素单位，不是经典库的 NDC 单位。

## 7. 与经典 3DGS 库的差异

| 经典库 | 本库 |
|---|---|
| 透视相机、雅可比 J、近平面剔除 | 正交切片位姿，沿法向做层厚积分（闭式解） |
| SH / RGB 颜色 + opacity | 单通道强度 I + 几何幅值 a |
| α 合成、透射率 T、0.99 截断、提前终止 | 线性加和，没有遮挡 |
| 按 tile + 深度排序 | 只按 tile 稳定排序 |
| `mark_visible` | 删除，可见性用 `radii > 0` 或 `amplitudes` 判断 |
| `getRect` 右端可能漏掉 1 像素 | 改为精确覆盖，保证截断支撑完整 |

## 8. 验证与性能

运行 `python tests/test_slice_mri.py`，在 pytest 可用时也可以用 pytest 运行。参考实现在 `tests/reference.py`，是逐像素的稠密 PyTorch 实现。测试内容：

- 闭式解与沿法向数值积分一致，s 取 0、0.7、2.5。
- s → ∞ 的极限。
- CUDA 前向与参考实现一致，最大误差约 5e-6；协方差的两种输入方式都测。
- 前向结果确定。
- CUDA 手写梯度与参考实现的 autograd 梯度一致，相对误差约 3e-6；对 means3D、means2D、强度、cov3D、尺度、旋转都检查，s 取 0 和 1.5。
- 强度为 0 的高斯仍有强度梯度。
- 斜切片渲染等于"先把高斯转到切片坐标再渲染"。
- s = 0 时与原零厚度 alpha 切片库一致：单个孤立高斯，误差 < 1e-5。

**性能。** A100，128×128 图像，单张切片前向加反向，按中位数统计（同一 GPU 上有其他任务在跑）：

| 高斯数 | 原 alpha 切片库 | 本库 s=0 | 本库 s=3.4 mm |
|---|---:|---:|---:|
| 5 万 | 2.41 ms | 2.18 ms | 2.33 ms |
| 8 万 | 2.97 ms | 2.36 ms | 2.59 ms |

## 9. 尚未实现（接口已预留或可平滑扩展）

- **S3 矩形层面轮廓**：用多个高斯的混合近似，单个 s 是只有一项时的特例。
- **S4 层面内 PSF。**
- **S5 层厚和位姿的梯度**：目前不回传 s 和位姿的梯度。
- **S7 一次调用渲染多张切片。**
- **S10 三维体素查询。**
- **S11 确定性梯度与双精度模式。**
