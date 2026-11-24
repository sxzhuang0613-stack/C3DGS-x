"""GaussianSVR（ISBI 2026）适配：单个短轴 stack、逐层位姿固定为名义几何。

与原 gaussian_reconstruct 的差异（其余逐行保持原流程与预设）：
- 不运行 SVoRT：心脏单 stack 无多 stack 约束，且无适用的预训练权重；初始体改为训练层沿 z 的线性插值。
- 逐层刚体参数固定为名义位姿（运动校正统一关闭），因此不建运动优化器与调度器。
- 前景掩膜沿用原定义 init_vol > 0（不额外抠图）；背景点数取 min(预设, 可用零值体素数)。
- max_iter = cfg.iters；low_reso_stage、densify_from/until、position_lr_max_steps 按原比例缩放。
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from case_data import use_reference, query_index

REPO_SUBDIR = 'GaussianSVR/src'


def _import_reference(cfg):
    """加载 GaussianSVR 源码模块（首次调用会 JIT 编译其 CUDA 扩展）。"""
    use_reference(Path(cfg.references_root) / REPO_SUBDIR)
    from kornia.losses import ssim_loss
    from models.gaussian_model import GaussianModel
    from data.rescan import ReScanner
    from utils import create_grid_3d, tv_regularization
    from gaussian_reconstruct import OptimizationParams
    return ssim_loss, GaussianModel, ReScanner, create_grid_3d, tv_regularization, OptimizationParams


def _volume_geometry(case, resolution):
    """各向同性体网格 [D,H,W]：面内与图像网格逐像素对齐，z 向居中并覆盖首末层外半个层厚。"""
    num_slices, height, width = case.shape
    half_extent = (num_slices - 1) / 2 * case.spacing[2] + case.thickness / 2
    depth = 2 * int(np.ceil(half_extent / resolution)) + 1
    return depth, height, width


def _initial_volume(case, depth, resolution, device):
    """训练层沿 z 线性插值到体网格，替代 SVoRT 初始体；返回 [1,D,H,W,1]。"""
    images = case.frame(case.reference_phase)[case.train_index].to(device)      # [N,H,W]
    z_voxel = (torch.arange(depth, device=device) - (depth - 1) / 2) * resolution
    # z（mm）→ 训练层坐标 c，c=j 即第 j 个训练层；越界取最近端层
    k = z_voxel / case.spacing[2] + (case.shape[0] - 1) / 2
    c = ((k - float(case.train_index[0])) / case.stride).clamp(0, len(case.train_index) - 1)
    lower = c.floor().long().clamp(max=len(case.train_index) - 1)
    upper = (lower + 1).clamp(max=len(case.train_index) - 1)
    weight = (c - lower)[:, None, None]
    volume = images[lower] * (1 - weight) + images[upper] * weight
    return volume[None, ..., None]


def _slice_transforms(case, slice_index, device):
    """名义位姿：无旋转，平移为层中心 (0,0,z_k)（mm），axis-angle [N,6]。"""
    transforms = torch.zeros(len(slice_index), 6, device=device)
    transforms[:, 5] = torch.as_tensor(case.z_mm(slice_index), dtype=torch.float32, device=device)
    return transforms


def fit(case, cfg, method_cfg, out_dir, **_):
    """逐病例拟合 GaussianSVR，返回各层位经原 PSF 重扫描的预测。"""
    ssim_loss, GaussianModel, ReScanner, create_grid_3d, tv_regularization, OptimizationParams = \
        _import_reference(cfg)
    with open(Path(cfg.references_root) / method_cfg['preset'], encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    device = torch.device('cuda')

    # 迭代数统一；原预设里按绝对步数写死的里程碑按比例缩放
    ratio = cfg.iters / config['max_iter']
    config['max_iter'] = cfg.iters
    config['low_reso_stage'] = int(round(config['low_reso_stage'] * ratio))
    config['densify_from_iter'] = int(round(config['densify_from_iter'] * ratio))
    config['densify_until_iter'] = int(round(config['densify_until_iter'] * ratio))
    op = OptimizationParams()
    op.position_lr_max_steps = int(round(op.position_lr_max_steps * ratio))

    # 几何与观测：体素分辨率默认取面内 spacing（ACDC 面内各向同性）
    resolution = method_cfg.get('volume_resolution') or case.spacing[0]
    resolution_slice = case.spacing[0]
    depth, height, width = _volume_geometry(case, resolution)
    stacks = case.frame(case.reference_phase)[case.train_index][:, None].to(device)  # [N,1,H,W]
    train_transforms = _slice_transforms(case, case.train_index, device)
    re_scanner = ReScanner(slice_size=width)

    # 初始化（同原流程：低分辨率网格上按掩膜采样高斯）
    init_vol = _initial_volume(case, depth, resolution, device)
    grid = create_grid_3d(depth, height, width).to(device).unsqueeze(0)
    mask = (init_vol > 0).to(torch.uint8)
    grid_low = grid[:, ::2, ::2, ::2, :]
    image_low = init_vol[:, ::2, ::2, ::2, :]
    seg_low = mask[:, ::2, ::2, ::2, :][..., 0]
    num_background = min(config['num_bg_gaussian'], int((seg_low == 0).sum()))

    gaussians = GaussianModel()
    gaussians.random_initialize(
        seg_low, image_low, ini_intensity=config['ini_intensity'], ini_sigma=config['ini_sigma'],
        spatial_lr_scale=config['spatial_lr_scale'],
        fg_num_samples=config['num_fg_gaussian'], bg_num_samples=num_background,
    )
    gaussians.training_setup(op)

    def render_volume(iteration=None):
        """高斯 → 体数据 [1,1,D,H,W]；低分辨率阶段先在半分辨率网格采样再三线性上采样。"""
        if iteration is not None and iteration < config['low_reso_stage']:
            output = gaussians.grid_sample(grid_low)
            volume = F.interpolate(output.permute(0, 4, 1, 2, 3), size=init_vol.shape[1:4],
                                   mode='trilinear', align_corners=False)
        else:
            output = gaussians.grid_sample(grid)
            volume = output.permute(0, 4, 1, 2, 3)
        return output, volume

    # 训练（损失、增殖与剪枝调度同原实现）
    for iteration in range(config['max_iter']):
        gaussians.update_learning_rate(iteration)
        gaussians.optimizer.zero_grad(set_to_none=True)

        train_output, volume = render_volume(iteration)
        re_slices = re_scanner.scan(volume, train_transforms, resolution, resolution_slice,
                                    case.thickness)
        l1 = config['l1_weight'] * F.l1_loss(re_slices, stacks)
        ssim = (1 - config['l1_weight']) * ssim_loss(re_slices, stacks, window_size=11)
        tv = config['tv_weight'] * tv_regularization(train_output)
        loss = l1 + tv + ssim
        loss.backward()
        gaussians.optimizer.step()

        if config['do_density_control']:
            with torch.no_grad():
                if (iteration > config['densify_from_iter']
                        and iteration % config['densification_interval'] == 0):
                    if (gaussians.get_xyz.shape[0] < config['max_gaussians']
                            and iteration < config['densify_until_iter']):
                        gaussians.densify(config['max_grad'], sigma_extent=config['sigma_extent'])
                    gaussians.prune(config['min_intensity'])
        if iteration % config['log_iter'] == 0 or iteration == config['max_iter'] - 1:
            print(f'  iter {iteration + 1}/{config["max_iter"]} L1 {l1.item():.6f} '
                  f'TV {tv.item():.6f} N {gaussians.get_xyz.shape[0]}')

    # 保存高斯参数，并以原前向模型（PSF 重扫描）在名义层位输出
    torch.save(dict(xyz=gaussians._xyz.detach(), scaling=gaussians._scaling.detach(),
                    rotation=gaussians._rotation.detach(), intensity=gaussians._intensity.detach(),
                    volume_shape=(depth, height, width), resolution=resolution),
               out_dir / 'gaussians.pt')
    query = query_index(case, cfg.interp_step_size)
    with torch.no_grad():
        _, volume = render_volume()
        slices = re_scanner.scan(volume, _slice_transforms(case, query, device), resolution,
                                 resolution_slice, case.thickness)
    return dict(query_index=query, volume=slices[:, 0].cpu().numpy(), frame=case.reference_phase)
