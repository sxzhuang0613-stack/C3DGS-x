"""ECLARE（JMI 2026）适配：单体数据自监督层间超分，直接调用原 run_eclare。

与原 run-eclare CLI 的差异：
- 输入为内存数组 [H,W,N]（N 个等间隔训练层，lr_axis=2），不读 NIfTI；按原 parse_image 做 [0,1] min-max 归一化。
- 层厚取统一设置（relative_slice_thickness = 层厚 / 面内 spacing），因此不运行 ESPRESO 估计层面轮廓。
- 训练步数 n_patches / batch_size = cfg.iters；"preset" 时 n_patches 取原默认 1e6（batch 128，7813 步）。
- 不传 inplane_acq_res：输出保持 Dataloader 的 128×128 网格，与其余方法逐像素可比。
输出层位：原分辨率输出沿 z 先按原层面轮廓模糊（训练降质模型的前半部分），再在名义层位线性采样。
"""
import numpy as np
import torch
import torch.nn.functional as F

from math import ceil

from case_data import query_index, resolve_iters

PRESET_N_PATCHES = 1000000   # 原 run-eclare 的 --n-patches 默认值


def _hr_coordinate(train_coordinate, num_lr, num_hr, scale):
    """训练层坐标 c → 超分输出 z 索引；radifox resize 按视野中心对齐（已用斜坡输入数值验证）。"""
    return (train_coordinate - (num_lr - 1) / 2) * scale + (num_hr - 1) / 2


def fit(case, cfg, method_cfg, out_dir, **_):
    """逐病例训练 ECLARE，返回名义层位预测。"""
    from eclare.eclare import run_eclare
    from eclare.utils.train_set import TrainSet
    from eclare.utils.blur_kernel_ops import parse_kernel
    from eclare.utils.parse_image_file import normalize

    device = torch.device('cuda', torch.cuda.current_device())
    images = case.frame(case.reference_phase)[case.train_index].numpy()   # [N,H,W]
    image = np.ascontiguousarray(images.transpose(1, 2, 0)).astype(np.float32)  # [H,W,N]
    image, orig_min, orig_max = normalize(image, 0, 1)
    inplane = min(case.spacing[0], case.spacing[1])
    slice_separation = round(float(case.stride * case.spacing[2] / inplane), 3)
    blur_kernel = parse_kernel(None, method_cfg['slice_profile_type'], case.thickness / inplane)

    batch_size = method_cfg['batch_size']
    iterations = resolve_iters(cfg, ceil(PRESET_N_PATCHES / batch_size))
    n_patches = PRESET_N_PATCHES if cfg.iters == 'preset' else iterations * batch_size
    dataset = TrainSet(image=image, lr_axis=2, verbose=False)
    eclare_vol, _, _ = run_eclare(
        image, slice_separation, [1, 1, 1], 2, None, None, orig_min, orig_max, blur_kernel,
        dataset, device, batch_size=batch_size, n_patches=n_patches)

    # 输出 [H,W,n] → 沿 z 按层面轮廓模糊（复制边界，避免端层被零填充拉暗）
    volume = torch.from_numpy(np.ascontiguousarray(eclare_vol)).float()
    num_hr = volume.shape[2]
    kernel = blur_kernel.view(1, 1, -1)
    pad = kernel.shape[-1] // 2
    rows = volume.reshape(-1, 1, num_hr)
    blurred = F.conv1d(F.pad(rows, (pad, pad), mode='replicate'), kernel).reshape(volume.shape)

    # 名义层位：全栈层索引 k → 训练层坐标 c → 输出 z 索引，线性插值
    query = query_index(case, cfg.interp_step_size)
    c = (query - case.train_index[0]) / case.stride
    j = np.clip(_hr_coordinate(c, len(case.train_index), num_hr, slice_separation), 0, num_hr - 1)
    lower = np.floor(j).astype(int)
    upper = np.minimum(lower + 1, num_hr - 1)
    weight = torch.as_tensor(j - lower, dtype=torch.float32)[:, None, None]
    slices = blurred.permute(2, 0, 1)                                     # [n,H,W]
    prediction = slices[lower] * (1 - weight) + slices[upper] * weight
    return dict(query_index=query, volume=prediction.numpy(), frame=case.reference_phase,
                iterations=iterations)
