#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
"""MRI 切片高斯渲染：有限层厚沿法向积分 + 信号线性加和。用法见仓库根目录 USAGE.md。"""

import math
from typing import NamedTuple, Tuple

import torch
import torch.nn as nn

from . import _C

API_VERSION = 1
RENDER_MODE = 'additive_slab'

# 层面轮廓 FWHM 与高斯标准差之比 2·sqrt(2 ln 2)
FWHM_TO_SIGMA = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))


def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)


class SliceRasterizationSettings(NamedTuple):
    """单张切片的成像参数。所有长度（位姿、像素间距、slice_sigma、高斯位置与尺度）须为同一单位。

    slice_rotation: [3,3]，三行依次为面内 u（列方向）、面内 v（行方向）、法向 n，须正交。
    slice_origin:   [3]，像素 (row=0, col=0) 中心的坐标；像素 (i, j) 中心 = o + j·du·u + i·dv·v。
    pixel_spacing:  (du, dv)，沿 u、v 的像素间距。
    slice_sigma:    层面轮廓 N(0, s²) 的标准差；0 表示零厚度截面。必填，无默认值。
    """
    image_height: int
    image_width: int
    slice_rotation: torch.Tensor
    slice_origin: torch.Tensor
    pixel_spacing: Tuple[float, float]
    slice_sigma: float
    bg: float = 0.0
    scale_modifier: float = 1.0
    amplitude_cutoff: float = 1.0 / 255.0
    debug: bool = False


def _pack_pose(settings, device):
    """把 (R, o) 打包为 CUDA 端的 float[12]，并检查外部输入的基本约束。"""
    R = settings.slice_rotation.to(device=device, dtype=torch.float32)
    o = settings.slice_origin.to(device=device, dtype=torch.float32)
    if R.shape != (3, 3) or o.numel() != 3:
        raise ValueError('slice_rotation 须为 [3,3]，slice_origin 须为 3 个元素')
    # R 不正交会使积分方向与面内度量同时出错且不报错，这里显式检查
    if not torch.allclose(R @ R.T, torch.eye(3, device=device), atol=1e-4):
        raise ValueError('slice_rotation 必须是正交矩阵（行向量 u、v、n 两两正交且为单位长度）')
    du, dv = (float(x) for x in settings.pixel_spacing)
    if not (du > 0 and dv > 0 and math.isfinite(du) and math.isfinite(dv)):
        raise ValueError('pixel_spacing 必须为两个有限正数')
    if not (settings.slice_sigma >= 0 and math.isfinite(settings.slice_sigma)):
        raise ValueError('slice_sigma 必须为有限非负数')
    if not 0 < settings.amplitude_cutoff < 1:
        raise ValueError('amplitude_cutoff 必须在 (0, 1) 内')
    return torch.cat([R.reshape(-1), o.reshape(-1)]).contiguous(), du, dv


def rasterize_slice(means3D, means2D, intensities, scales, rotations, cov3Ds_precomp, raster_settings):
    return _RasterizeSlice.apply(means3D, means2D, intensities, scales, rotations, cov3Ds_precomp, raster_settings)


class _RasterizeSlice(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means3D, means2D, intensities, scales, rotations, cov3Ds_precomp, raster_settings):
        pose, du, dv = _pack_pose(raster_settings, means3D.device)
        args = (
            means3D, intensities, scales, rotations, raster_settings.scale_modifier, cov3Ds_precomp,
            pose, du, dv, float(raster_settings.slice_sigma), float(raster_settings.amplitude_cutoff),
            float(raster_settings.bg), raster_settings.image_height, raster_settings.image_width,
            raster_settings.debug,
        )
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)
            try:
                num_rendered, color, radii, amps, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_slice(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, radii, amps, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_slice(*args)

        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.pose_spacing = (du, dv)
        ctx.save_for_backward(intensities, means3D, scales, rotations, cov3Ds_precomp, radii, pose,
                              geomBuffer, binningBuffer, imgBuffer)
        ctx.mark_non_differentiable(radii, amps)
        return color, radii, amps

    @staticmethod
    def backward(ctx, grad_out_color, _radii, _amps):
        raster_settings = ctx.raster_settings
        du, dv = ctx.pose_spacing
        (intensities, means3D, scales, rotations, cov3Ds_precomp, radii, pose,
         geomBuffer, binningBuffer, imgBuffer) = ctx.saved_tensors
        args = (
            means3D, radii, intensities, scales, rotations, raster_settings.scale_modifier, cov3Ds_precomp,
            pose, du, dv, float(raster_settings.slice_sigma), float(raster_settings.amplitude_cutoff),
            grad_out_color, geomBuffer, ctx.num_rendered, binningBuffer, imgBuffer, raster_settings.debug,
        )
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)
            try:
                grads = _C.rasterize_slice_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
            grads = _C.rasterize_slice_backward(*args)
        grad_means2D, grad_intensities, grad_means3D, grad_cov3D, grad_scales, grad_rotations = grads

        # 未提供的输入是空张量，返回 None
        return (
            grad_means3D,
            grad_means2D,
            grad_intensities,
            grad_scales if scales.numel() else None,
            grad_rotations if rotations.numel() else None,
            grad_cov3D if cov3Ds_precomp.numel() else None,
            None,
        )


class SliceRasterizer(nn.Module):
    """渲染一张 MRI 切片：C(p) = bg + Σ_i I_i · a_i · exp(-½ dᵀ conic_i d)。"""

    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def forward(self, means3D, means2D, intensities, scales=None, rotations=None, cov3D_precomp=None):
        """返回 (图像 [1,H,W], radii [N] int32, 幅值 a [N])。

        means2D: 与 means3D 同形的零张量，仅用于接收像素单位的面内中心梯度（同经典 3DGS）。
        intensities: [N,1] 峰值信号密度；协方差二选一给出：scales [N,3] + rotations [N,4]（wxyz，
        库内不归一化）或 cov3D_precomp [N,6]（xx, xy, xz, yy, yz, zz）。
        """
        if ((scales is None or rotations is None) and cov3D_precomp is None) or \
                ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise ValueError('scales/rotations 与 cov3D_precomp 必须二选一给出')
        empty = means3D.new_empty(0)
        scales = empty if scales is None else scales.contiguous()
        rotations = empty if rotations is None else rotations.contiguous()
        cov3D_precomp = empty if cov3D_precomp is None else cov3D_precomp.contiguous()
        return rasterize_slice(means3D.contiguous(), means2D, intensities.contiguous(), scales, rotations,
                               cov3D_precomp, self.raster_settings)


def stack_slice_pose(spacing_xyz, slice_coord, device='cuda'):
    """只有 spacing 时的“图像栈局部坐标”位姿：所有切片平行于 xy 平面、等间距、面内无错位。

    spacing_xyz: (dx, dy, dz)，dx 为列方向像素间距，dy 为行方向，dz 为层间距（单位即渲染单位）。
    slice_coord: 连续层索引 t（可为小数，用于层间插帧），切片原点 o = (0, 0, t·dz)。
    返回 (slice_rotation [3,3], slice_origin [3], pixel_spacing (dx, dy))。
    """
    dx, dy, dz = (float(v) for v in spacing_xyz)
    rotation = torch.eye(3, device=device, dtype=torch.float32)
    origin = torch.tensor([0.0, 0.0, float(slice_coord) * dz], device=device, dtype=torch.float32)
    return rotation, origin, (dx, dy)
