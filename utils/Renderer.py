"""高斯切片渲染包装：固定 opacity=1 的 alpha 合成（旧模型）或加和层厚积分（additive_slab_v1）。"""
import math
import numbers

import torch

from Gaussians.utils import covariance_from_scaling_rotation


def covariance_to_vector(covariance):
    """将 [N,3,3] 对称协方差压缩为 [xx,xy,xz,yy,yz,zz]。"""
    if covariance.ndim == 2 and covariance.shape[-1] == 6:
        return covariance
    if covariance.ndim != 3 or covariance.shape[-2:] != (3, 3):
        raise ValueError("covariance 必须为 [N,3,3] 或 [N,6]")
    return torch.stack([
        covariance[:, 0, 0], covariance[:, 0, 1], covariance[:, 0, 2],
        covariance[:, 1, 1], covariance[:, 1, 2], covariance[:, 2, 2],
    ], dim=-1)


ALPHA_RENDER_MODEL = 'innergs_alpha_fixed_v1'
ADDITIVE_RENDER_MODEL = 'additive_slab_v1'


def _gaussian_inputs(Gaussians, xyz, colors, scales, rotations, covariance):
    """取渲染输入：位置 [N,3]、强度 [N,1]、模型坐标协方差 [N,6]。"""
    xyz = Gaussians.get_xyz if xyz is None else xyz
    colors = Gaussians.get_color if colors is None else colors
    if colors.ndim != 2 or colors.shape != (len(xyz), 1):
        raise ValueError('Stage1 强度必须为 [N,1]')
    if covariance is None:
        scales = Gaussians.get_scaling if scales is None else scales
        if rotations is None:
            rotations = Gaussians._rotation if hasattr(Gaussians, '_rotation') else Gaussians.get_rotation
        packed = covariance_from_scaling_rotation(scales, rotations)
    else:
        packed = covariance_to_vector(covariance)
    return xyz, colors, packed


def _render_additive(Gaussians, img_size, slice_idx, axis, num_slices,
                     xyz, colors, scales, rotations, covariance, raster_options):
    """加和层厚切片：C = bg + Σ I·a·G（diff_gaussian_rasterization_mri），直接使用模型坐标。

    模型坐标以体数据中心为原点、单位 = mm/scale_mm：像素 (i,j) 中心 x=(j-(W-1)/2)dx、
    y=(i-(H-1)/2)dy，第 t 层（可为小数）z=(t-(S-1)/2)dz，与初始化采样及旧渲染器的像素映射一致。
    """
    import diff_gaussian_rasterization_mri as rasterizer
    if (getattr(rasterizer, 'API_VERSION', None) != 1
            or getattr(rasterizer, 'RENDER_MODE', None) != 'additive_slab'):
        raise ImportError('需要 API_VERSION=1 的 additive_slab 切片库，请重新编译安装')
    if axis != 2:
        raise ValueError('加和切片模型当前只支持 axis=2（层法向为 z）')
    options = dict(raster_options or {})
    unsupported = options.keys() - {'background', 'debug', 'slice_sigma', 'amplitude_cutoff'}
    if unsupported or 'slice_sigma' not in options:
        raise ValueError(f'加和切片需要 slice_sigma，且不支持这些选项：{sorted(unsupported)}')
    geometry = Gaussians.geometry
    if geometry['version'] == 0:
        raise ValueError('加和切片模型需要 spacing 局部坐标（geometry version 1）')
    dx, dy, dz = (value/geometry['scale_mm'] for value in geometry['spacing_xyz'])

    h, w = img_size
    xyz, colors, packed = _gaussian_inputs(Gaussians, xyz, colors, scales, rotations, covariance)
    rotation = torch.eye(3, device=xyz.device, dtype=torch.float32)
    origin = torch.tensor([-(w-1)/2*dx, -(h-1)/2*dy, (slice_idx-(num_slices-1)/2)*dz],
                          device=xyz.device, dtype=torch.float32)
    settings = rasterizer.SliceRasterizationSettings(
        image_height=h, image_width=w, slice_rotation=rotation, slice_origin=origin,
        pixel_spacing=(dx, dy), slice_sigma=float(options['slice_sigma']),
        bg=float(options.get('background', 0.)),
        amplitude_cutoff=float(options.get('amplitude_cutoff', 1./255.)),
        debug=bool(options.get('debug', False)),
    )
    # 每次调用独立的 means3D 别名与 means2D：反向后分别给出本切片的三维位置梯度与像素单位面内梯度，
    # 供结构统计逐切片累积（共享 xyz.grad 只能得到各切片之和）
    track = torch.is_grad_enabled() and xyz.requires_grad
    means3D = xyz + 0 if track else xyz
    if track:
        means3D.retain_grad()
    means2D = torch.zeros_like(xyz, requires_grad=track)
    image, radii, amplitudes = rasterizer.SliceRasterizer(settings)(
        means3D=means3D.contiguous(), means2D=means2D,
        intensities=colors.contiguous(), cov3D_precomp=packed.contiguous(),
    )
    return {'render': image[0], 'visibility_filter': radii > 0, 'radii': radii,
            'amplitudes': amplitudes, 'means2D': means2D, 'means3D': means3D,
            'slice_normal': rotation[2]}


def render(Gaussians, img_size, slice_idx, axis, num_slices,
           xyz=None, colors=None, opacities=None, scales=None,
           rotations=None, covariance=None, bounds=None, raster_options=None,
           render_model=ALPHA_RENDER_MODEL):
    """在病例坐标中求三维高斯切片；render_model 由 geometry['rendering'] 给出，旧记录默认 alpha 合成。"""
    if opacities is not None:
        raise ValueError('本模型固定 opacity=1，不接受外部 opacities')
    h, w = img_size
    if (isinstance(axis, bool) or not isinstance(axis, numbers.Integral)
            or axis not in (0, 1, 2)):
        raise ValueError('axis 必须为 0、1 或 2')
    if (isinstance(num_slices, bool) or not isinstance(num_slices, numbers.Integral)
            or num_slices < 1):
        raise ValueError('num_slices 必须为正整数')
    if (isinstance(slice_idx, bool) or not isinstance(slice_idx, numbers.Real)
            or not math.isfinite(slice_idx) or not 0 <= slice_idx <= num_slices-1):
        raise ValueError('slice_idx 必须为 [0,num_slices-1] 内的有限连续索引')
    if render_model == ADDITIVE_RENDER_MODEL:
        return _render_additive(Gaussians, img_size, slice_idx, axis, num_slices,
                                xyz, colors, scales, rotations, covariance, raster_options)
    if render_model != ALPHA_RENDER_MODEL:
        raise ValueError(f'未知渲染模型 {render_model}')

    # 正常导入已安装包；延迟导入使 CPU 几何与结构检查无需加载 CUDA 扩展。
    import diff_gaussian_rasterization_slice as rasterizer
    if (getattr(rasterizer, 'API_VERSION', None) != 2
            or getattr(rasterizer, 'RENDER_MODE', None) != 'alpha'):
        raise ImportError('需要 API_VERSION=2 的 alpha 切片库，请重新编译安装')
    GaussianRasterizationSettings = rasterizer.GaussianRasterizationSettings
    GaussianRasterizer = rasterizer.GaussianRasterizer

    bounds = bounds if bounds is not None else getattr(Gaussians, 'bounds', (1., 1., 1.))
    if len(bounds) != 3 or any(not math.isfinite(value) or value <= 0 for value in bounds):
        raise ValueError('bounds 必须为三个有限正半边长')
    options = dict(raster_options or {})
    unsupported = options.keys() - {'background', 'debug'}
    if unsupported:
        raise ValueError(f'alpha 切片不支持这些旧渲染选项：{sorted(unsupported)}')

    xyz, colors, packed = _gaussian_inputs(Gaussians, xyz, colors, scales, rotations, covariance)

    # 非等比盒子归一化必须变换完整协方差，不能只缩放三个主轴尺度。
    a, b, c = (1./value for value in bounds)
    inverse_bounds = xyz.new_tensor([a, b, c])
    factors = xyz.new_tensor([a*a, a*b, a*c, b*b, b*c, c*c])
    means = (xyz * inverse_bounds).contiguous()
    packed = (packed * factors).contiguous()
    plane = [0., 0., 0.]
    plane[axis] = 0. if num_slices == 1 else -1.+2.*slice_idx/(num_slices-1)
    identity = torch.eye(4, device=xyz.device, dtype=xyz.dtype)
    settings = GaussianRasterizationSettings(
        image_height=h, image_width=w, tanfovx=1., tanfovy=1.,
        bg=xyz.new_full((3,), options.get('background', 0.)), scale_modifier=1.,
        viewmatrix=identity, projmatrix=identity, sh_degree=0, campos=xyz.new_tensor(plane),
        prefiltered=False, debug=options.get('debug', False), antialiasing=False, axis=axis,
    )
    image, radii, _ = GaussianRasterizer(settings)(
        means3D=means, means2D=torch.zeros_like(means),
        colors_precomp=colors.expand(-1, 3).contiguous(),
        opacities=xyz.new_ones((len(xyz), 1)), cov3D_precomp=packed,
    )
    # radii 表示候选范围；遮挡、截断后不一定产生实际像素贡献。
    return {'render': image[0], 'visibility_filter': radii > 0, 'radii': radii}
