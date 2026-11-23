"""训练与测试共用的配置、设备及数据参数。"""
import argparse
import math
import os
from pathlib import Path

import yaml


RENDER_MODEL = "innergs_alpha_fixed_v1"
RASTERIZER_API_VERSION = 2
# 加和层厚切片模型（diff_gaussian_rasterization_mri）：C = bg + Σ I·a·G，强度经 softplus 保证非负
ADDITIVE_RENDER_MODEL = "additive_slab_v1"
# 每种渲染模型的语义契约：库版本、强度/opacity 语义及配套的结构算法；geometry 与 checkpoint 按此核对
RENDER_SEMANTICS = {
    RENDER_MODEL: dict(api=RASTERIZER_API_VERSION, intensity='local_value', opacity='fixed_one',
                       structure='batch_residual_v5'),
    ADDITIVE_RENDER_MODEL: dict(api=1, intensity='peak_density_softplus', opacity='none',
                                structure='additive_moment_v1'),
}


def load_config(path):
    # 拒绝重复键，避免同一实验参数被后面的值静默覆盖。
    class UniqueLoader(yaml.SafeLoader):
        pass

    def mapping(loader, node):
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node)
            if key in result:
                raise ValueError(f'配置键重复：{key}')
            result[key] = loader.construct_object(value_node)
        return result

    UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    config_path = Path(path).resolve()
    with config_path.open(encoding='utf-8') as stream:
        values = yaml.load(stream, Loader=UniqueLoader)
    values.setdefault('dataset_name', 'acdc')
    values.setdefault('render_model', RENDER_MODEL)
    if not 0 <= values['train_fraction'] <= 1 or values['reference_phase'] < 0:
        raise ValueError('病例划分比例或参考相位非法')
    if len(values['bounds']) != 3 or any(value <= 0 for value in values['bounds']):
        raise ValueError('bounds 必须为三个正半边长')
    if values['render_model'] not in RENDER_SEMANTICS:
        raise ValueError(f'仅支持 {sorted(RENDER_SEMANTICS)} 渲染模型')
    obsolete = set(values) & {'response_mode', 'psf_sigma', 'psf_std_vector',
                              'kernel_cutoff', 'cutoff_transition', 'covariance_jitter'}
    if obsolete:
        raise ValueError(f'新库不支持这些旧渲染配置，请移除：{sorted(obsolete)}')
    obsolete_structure = set(values) & {'max_num_pts', 'structural_growth_fraction',
                                       'structural_split_fraction',
                                       # 层间拉长设定已取消，初始尺度改为经典 3DGS kNN
                                       'init_fine_sigma_pixels', 'init_coarse_sigma_pixels',
                                       'init_coarse_fraction', 'init_normal_sigma_spacing',
                                       'init_normal_floor_pixels', 'init_max_axis_ratio',
                                       'lambda_normal_floor', 'normal_floor_kappa',
                                       'lambda_interp_coverage', 'structural_prune_max_normal_spacing',
                                       # 尺寸超限改为强制 Split，旧剪枝参数名不再使用
                                       'structural_prune_from_iter', 'structural_prune_max_screen_radius',
                                       'structural_prune_max_scale_ratio',
                                       # 已取消按贡献删除，支撑门改用 structural_support_min_weight
                                       'structural_prune_min_weight',
                                       # Clone 已取消
                                       'structural_clone_offset', 'structural_clone_jitter',
                                       'structural_enable_clone'}
    if obsolete_structure:
        raise ValueError(f'已取消这些旧结构配置，请移除：{sorted(obsolete_structure)}')
    if 'iters' in values:
        for key in ('iters', 'checkpoint_interval', 'loss_plot_interval', 'status_interval',
                    'structure_log_interval',
                    'real_slices_per_iter', 'num_pts', 'interp_step_size',
                    'stage2_iters', 'stage2_save_interval', 'stage2_loss_plot_interval',
                    'init_edge_sigma', 'init_edge_truncate', 'init_ridge_relative',
                    'min_sigma_pixels'):
            if values[key] <= 0:
                raise ValueError(f'{key} 必须大于零')
        if not (0 <= values['init_uniform_fraction'] <= 1
                and 0 <= values['init_foreground_fraction'] <= 1-values['init_uniform_fraction']):
            raise ValueError('初始化均匀/前景比例非负且总和不能超过 1')
        quantiles = values['init_sampling_quantiles']
        if len(quantiles) != 2 or not 0 <= quantiles[0] < quantiles[1] <= 1:
            raise ValueError('初始化采样分位数必须为 [low,high]，满足 0 <= low < high <= 1')
        for key in ('init_foreground_threshold','init_edge_quantile'):
            if not 0 <= values[key] <= 1:
                raise ValueError(f'{key} 必须位于 [0,1]')
        for key in ('init_foreground_dilation','init_fit_steps','init_support_retries'):
            if not isinstance(values[key],int) or values[key] < 0:
                raise ValueError(f'{key} 必须为非负整数')
        if not (0 <= values['init_grid_jitter'] < .5 and 0 < values['init_normal_jitter'] <= .5
                and 0 < values['init_cg_rtol'] < 1):
            raise ValueError('初始化抖动或 CG 容差配置非法')
        if not 1 <= values['structural_min_points'] <= values['num_pts']:
            raise ValueError('初始点数需要 1 <= structural_min_points <= num_pts')
        expected = RENDER_SEMANTICS[values['render_model']]['structure']
        if values['structural_algorithm'] != expected:
            raise ValueError(f"{values['render_model']} 渲染模型需要 structural_algorithm={expected}")
        if values['render_model'] == ADDITIVE_RENDER_MODEL:
            for key in ('slice_thickness_fwhm_spacing', 'structural_normal_grad_multiplier',
                        'structural_prune_intensity', 'structural_size_max_sigma_pixels'):
                if not isinstance(values[key], (int, float)) or not math.isfinite(values[key]) or values[key] < 0:
                    raise ValueError(f'{key} 必须为有限非负数')
            for key in ('structural_split_ratio', 'structural_size_split_ratio'):
                if not 0 < values[key] < 1:
                    raise ValueError(f'{key} 必须位于 (0,1)')
            for key in ('amplitude_cutoff', 'structural_observe_amplitude'):
                if not 0 < values[key] < 1:
                    raise ValueError(f'{key} 必须位于 (0,1)')
            if not 0 <= values['structural_normal_conflict'] < 1:
                raise ValueError('structural_normal_conflict 必须位于 [0,1)')
            if not isinstance(values['init_nonneg_steps'], int) or values['init_nonneg_steps'] < 0:
                raise ValueError('init_nonneg_steps 必须为非负整数')
        if values['structural_prune_peak'] != 0:
            raise ValueError('固定 opacity 不支持按颜色强度剪枝，structural_prune_peak 必须为 0')
        for key in ('structural_grad_threshold', 'structural_grad_multiplier',
                    'structural_prune_peak',
                    'structural_size_max_screen_radius', 'structural_size_max_scale_ratio',
                    'structural_residual_threshold', 'structural_support_min_weight',
                    'lambda_interp_envelope'):
            if not isinstance(values[key], (int, float)) or not math.isfinite(values[key]) or values[key] < 0:
                raise ValueError(f'{key} 必须为有限非负数')
        if not (isinstance(values['structural_min_observations'], int)
                and values['structural_min_observations'] > 0):
            raise ValueError('structural_min_observations 必须为正整数')
        for key in ('structural_enable_split',
                    'structural_inplane_gradient', 'structural_residual_gate'):
            if not isinstance(values[key], bool):
                raise ValueError(f'{key} 必须为布尔值')
        if not isinstance(values['visualize'], bool):
            raise ValueError('visualize 必须为布尔值')
        if not isinstance(values['structural_prune_after_densify'], bool):
            raise ValueError('structural_prune_after_densify 必须为布尔值')
        if not math.isfinite(values['structural_split_sigma_pixels']) or values['structural_split_sigma_pixels'] <= 0:
            raise ValueError('structural_split_sigma_pixels 必须为有限正数')
        if (not isinstance(values['structural_size_split_from_iter'], int)
                or values['structural_size_split_from_iter'] < 0):
            raise ValueError('structural_size_split_from_iter 必须为非负整数')
        if (values['lambda_l1'] < 0 or values['lambda_ssim'] < 0
                or values['lambda_l1']+values['lambda_ssim'] <= 0):
            raise ValueError('训练需要非负且不全为零的 L1/SSIM 权重')
        if values['ssim_window_size'] <= 0 or values['ssim_window_size'] % 2 != 1:
            raise ValueError('SSIM 需要正奇数窗口')
        if not (0 <= values['interpolation_probability'] <= 1
                and 0 <= values['interpolation_alpha_min'] <= values['interpolation_alpha_max'] <= 1):
            raise ValueError('插层概率或 alpha 范围非法')
    # 相对路径以配置文件为基准，不依赖启动命令所在目录。
    for key, value in values.items():
        if key.endswith(('_path', '_dir', '_root')) and isinstance(value, str) and value:
            value = os.path.expandvars(os.path.expanduser(value))
            values[key] = str((config_path.parent / value).resolve())
    values['dataset_paths'] = {
        name: str((config_path.parent / os.path.expandvars(os.path.expanduser(value))).resolve())
        for name, value in values['dataset_paths'].items()
    }
    return argparse.Namespace(**values)


def setup_runtime(args):
    import random
    import numpy as np
    import torch

    device = torch.device(args.device)
    if device.type != 'cuda':
        raise ValueError('当前渲染扩展要求 CUDA 设备')
    torch.cuda.set_device(device)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)


def dataset_options(args):
    # 主训练/重建按完整病例工作；固定使用加载器的 [S,T,H,W] 输出。
    name = getattr(args, 'dataset_name', 'acdc').lower()
    name = {'mm':'mnms','mms':'mnms','m&ms':'mnms'}.get(name,name)
    return dict(
        datasets=name, data_path=args.dataset_paths[name], dim='3D',
        train_fraction=args.train_fraction,
        normalization_epsilon=args.normalization_epsilon, config=args,
    )


def render_options(args):
    return dict(bounds=args.bounds, raster_options={
        'background': args.background,
        'debug': args.raster_debug,
    })


def ssim_options(args):
    return dict(window_size=args.ssim_window_size, sigma=args.ssim_sigma,
                k1=args.ssim_k1, k2=args.ssim_k2, data_range=args.data_range)
