import math
import os
import warnings

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from plyfile import PlyData, PlyElement

from .utils import get_expon_lr_func, normalize_rotations, quaternion_rotation_matrix


class GaussianModel:
    parameter_names = {'xyz': '_xyz', 'intensity': '_intensity',
                       'rotation': '_rotation', 'scaling': '_scaling'}
    # 与点一一对应的结构统计量；剪枝与 checkpoint 共用此列表
    statistic_names = ('xyz_gradient_accum', 'denom', 'max_radii2D',
                       'normal_grad_abs', 'normal_grad_slice', 'slice_observed')

    def __init__(self):
        self._xyz = torch.empty(0, 3)
        self._intensity = torch.empty(0, 1)
        self._scaling = torch.empty(0, 3)
        self._rotation = torch.empty(0, 4)
        self.time_func = torch.empty(0)
        self.use_dff = False
        # 首个结构周期标定后冻结的绝对梯度门限；随 checkpoint 保存。
        self.structural_grad_reference = None
        # 加和模型 D2 法向分裂的冻结门限，标定方式同上
        self.structural_normal_reference = None
        # True 时 _intensity 为 softplus 前的原始参数（加和模型，强度非负）；由 attach_geometry 按渲染语义设置
        self.intensity_softplus = False

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_intensity(self):
        return F.softplus(self._intensity) if self.intensity_softplus else self._intensity

    @property
    def get_color(self):
        """兼容共享 Renderer 的灰度属性名。"""
        return self.get_intensity

    def intensity_parameter(self, intensity):
        """激活后强度 → 原始参数；softplus 逆 x = I + log(1 - e^{-I})，对大 I 数值稳定，要求 I > 0。"""
        if not self.intensity_softplus:
            return intensity
        return intensity + torch.log(-torch.expm1(-intensity))

    @property
    def get_scaling(self):
        return self._scaling.exp()

    @property
    def get_rotation(self):
        return normalize_rotations(self._rotation)

    @property
    def get_time(self):
        return F.softmax(self.time_func.squeeze(), dim=0) if self.use_dff else self.time_func

    def initialize_from_images(self, images, geometry, args, rendering, case_name,
                               slice_index=None, num_slices=None):
        from .initialization import initialize_from_images
        return initialize_from_images(self, images, geometry, args, rendering, case_name,
                                      slice_index, num_slices)

    def training_setup(self, args):
        self.structural_args = args
        if not hasattr(self, 'geometry_reference_count'):
            self.geometry_reference_count = len(self._xyz)
        self._reset_statistics()
        self._structural_spacing = None
        self._scale_floor = None
        if hasattr(self, 'geometry'):
            self._voxel_spacing()
        groups = [dict(params=[getattr(self, name)], lr=getattr(args, lr), name=key)
                  for key, name, lr in (
                      ('xyz', '_xyz', 'position_lr_init'),
                      ('intensity', '_intensity', 'intensity_lr'),
                      ('rotation', '_rotation', 'rotation_lr'),
                      ('scaling', '_scaling', 'scaling_lr'))]
        self.optimizer = torch.optim.Adam(groups, lr=0., eps=args.adam_eps,
                                          betas=tuple(args.adam_betas))
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=args.position_lr_init, lr_final=args.position_lr_final,
            lr_delay_mult=args.position_lr_delay_mult, max_steps=args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        lr = self.xyz_scheduler_args(iteration)
        for group in self.optimizer.param_groups:
            if group['name'] == 'xyz':
                group['lr'] = lr
        return lr

    def _reset_statistics(self):
        self.xyz_gradient_accum = self._xyz.new_zeros((len(self._xyz), 1))
        self.denom = self._xyz.new_zeros((len(self._xyz), 1))
        # 记录本周期实际渲染到的最大候选半径，用于过大尺寸剪枝。
        self.max_radii2D = self._xyz.new_zeros(len(self._xyz))
        # 加和模型的法向统计：Σ|h_k| 及逐层带符号和 H_t = Σ_{k∈层t} h_k，h_k = n·∂L/∂μ（单次调用）
        slices = self.geometry['shape_shw'][0] if hasattr(self, 'geometry') else 1
        self.normal_grad_abs = self._xyz.new_zeros(len(self._xyz))
        self.normal_grad_slice = self._xyz.new_zeros((len(self._xyz), slices))
        self.slice_observed = self._xyz.new_zeros((len(self._xyz), slices))

    def _voxel_spacing(self):
        if getattr(self, '_structural_spacing', None) is not None:
            return self._structural_spacing
        geometry = getattr(self, 'geometry', None)
        if geometry is None:
            raise ValueError('批量结构操作需要病例 geometry；请先 attach_geometry')
        if geometry['version'] != 0:
            spacing = self._xyz.new_tensor(geometry['spacing_xyz']) / geometry['scale_mm']
        else:
            s, h, w = geometry['shape_shw']
            spacing = 2 * self._xyz.new_tensor(geometry['bounds']) / self._xyz.new_tensor([w, h, max(s-1, 1)])
        # 病例几何固定；避免逐训练步重复创建 CPU->GPU 的 spacing Tensor。
        self._structural_spacing = spacing
        return spacing

    def log_scale_floor(self):
        """按面内像素换算的 log 尺度下界；像素中心采样、无抗锯齿时更细的核退化为脉冲。"""
        if self._scale_floor is None:
            pixel = float(self._voxel_spacing()[:2].min())
            args = self.structural_args
            self._scale_floor = max(math.log(args.min_sigma_pixels*pixel), args.log_scale_min)
        return self._scale_floor

    def _observation_box(self):
        """观测体积半边长：每个像素/层代表一个体素，因此 z 为 S*sz/2 而非层中心范围。"""
        s, h, w = self.geometry['shape_shw']
        return self._voxel_spacing()*self._xyz.new_tensor([w, h, s])/2

    def _inplane_sigma(self, log_scales, raw_rotations):
        """面内投影协方差的最大标准差，模型坐标单位。"""
        matrix = self._rotation_matrix(raw_rotations)[:, :2, :] * log_scales.exp()[:, None, :]
        xx, yy = matrix[:, 0].square().sum(-1), matrix[:, 1].square().sum(-1)
        xy = (matrix[:, 0] * matrix[:, 1]).sum(-1)
        return ((xx+yy+torch.sqrt((xx-yy).square()+4*xy.square()))*.5).clamp_min(0).sqrt()

    @staticmethod
    def _rotation_matrix(raw):
        return quaternion_rotation_matrix(raw)

    def _invalid_point_mask(self):
        invalid = torch.zeros(len(self._xyz), device=self._xyz.device, dtype=torch.bool)
        for value in (self._xyz, self._intensity, self._scaling, self._rotation):
            invalid |= ~torch.isfinite(value).flatten(1).all(-1)
        scales = self.get_scaling
        invalid |= ~torch.isfinite(scales).all(-1) | (scales <= 0).any(-1)
        components = tuple(component.double() for component in self._rotation.unbind(-1))
        invalid |= torch.sqrt(sum(value.square() for value in components)) < 1e-12
        return invalid

    @torch.no_grad()
    def add_densification_stats_3d(self, visibility_filter, radii=None):
        """累计体素单位的位置梯度；可选 radii 为当前切片集合的最大像素半径。"""
        if radii is not None:
            self.max_radii2D[visibility_filter] = torch.maximum(
                self.max_radii2D[visibility_filter], radii[visibility_filter].to(self.max_radii2D))
        if self._xyz.grad is None:
            return
        # g_voxel = dL/dx_model * 一个体素的模型长度；不改变实际训练梯度。
        raw = torch.where(visibility_filter[:, None], self._xyz.grad.detach(), 0.)
        if self.structural_args.structural_inplane_gradient:
            # 经典 3DGS 用屏幕空间二维梯度；切片上的对应量是面内 (x,y) 分量。z 梯度在固定 opacity 下
            # 主要反映"用深度调本层可见度"，乘层间距后会主导统计量（实测约 98%），故不参与增殖统计。
            raw = raw * raw.new_tensor([1., 1., 0.])
        values = torch.linalg.vector_norm(raw * self._voxel_spacing(), dim=-1, keepdim=True)
        self.xyz_gradient_accum.add_(values)
        self.denom.add_(visibility_filter[:, None])

    @torch.no_grad()
    def contribution_audit(self, images, render_kwargs):
        """全病例逐层渲染，返回每点贡献质量 W=sum(T*alpha) 与贡献加权残差 E=sum(T*alpha*|r|)。

        渲染对颜色线性，dI/dc_i=T_i*alpha_i 与颜色取值无关，两次颜色反向即得 W、E；
        二者与颜色无关，按 W 剪枝不会误删暗色组织。单位为像素当量。
        """
        from utils.Renderer import render
        from .utils import covariance_from_scaling_rotation
        slices, height, width = images.shape
        xyz = self._xyz.detach()
        covariance = covariance_from_scaling_rotation(self.get_scaling, self._rotation)
        weight = xyz.new_zeros(len(xyz))
        error = xyz.new_zeros(len(xyz))
        for index in range(slices):
            with torch.enable_grad():
                colors = self._intensity.detach().clone().requires_grad_()
                image = render(self, [height, width], index, 2, slices, xyz=xyz, colors=colors,
                               covariance=covariance, **render_kwargs)['render']
                residual = (image.detach()-images[index]).abs()
                unit, = torch.autograd.grad(image, colors, torch.ones_like(image), retain_graph=True)
                weighted, = torch.autograd.grad(image, colors, residual)
            weight += unit[:, 0]
            error += weighted[:, 0]
        return weight, error

    @torch.no_grad()
    def _replace_structure(self, survivors, sources, data, reset=None):
        """每组参数只提交一次；保留旧行动量，新行动量清零，沿用 Adam 组级时钟。"""
        reset = {} if reset is None else reset
        for group in self.optimizer.param_groups:
            key = group['name']
            old = group['params'][0]
            parameter = nn.Parameter(data[key])
            state = self.optimizer.state.pop(old, None)
            if state is not None:
                state = dict(state)
                for name in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
                    if name in state:
                        moment = torch.cat((state[name][survivors], torch.zeros_like(data[key][len(survivors):])), 0)
                        if key in reset:
                            moment[:len(survivors)][reset[key]] = 0
                        state[name] = moment
                self.optimizer.state[parameter] = state
            group['params'][0] = parameter
            setattr(self, self.parameter_names[key], parameter)

    @torch.no_grad()
    def prune_points(self, mask):
        if mask.dtype != torch.bool or mask.shape != (len(self._xyz),):
            raise ValueError('prune mask 必须是 bool [N]')
        if not bool(mask.any()):
            return
        survivors = torch.where(~mask)[0]
        sources = survivors[:0]
        data = {key: getattr(self, name)[survivors] for key, name in self.parameter_names.items()}
        self._replace_structure(survivors, sources, data)
        for name in self.statistic_names:
            setattr(self, name, getattr(self, name)[survivors])

    @torch.no_grad()
    def structural_step(self, images, iteration, render_kwargs=None, similarity_kwargs=None, densify=True):
        """残差门控的批量 Split、尺寸超限强制 Split，以及观测框与无效点剪枝（不做 Clone）。

        误差驱动的增殖必须同时满足：冻结的绝对梯度门限（定位需要移动的点）与覆盖区平均残差 > tau
        （确认该处确有误差）。尺寸超限的点不受这些门控和 densify 开关限制，一律按经典 Split 拆小；
        densify=False 时只停止误差驱动的增殖。
        """
        args = self.structural_args
        before = len(self._xyz)
        invalid = self._invalid_point_mask()
        valid = ~invalid
        if int(valid.sum()) < args.structural_min_points:
            raise FloatingPointError('有效点少于 structural_min_points，无法安全调整结构')
        if args.structural_residual_gate:
            weight, error = self.contribution_audit(images, render_kwargs)
            weight = torch.where(valid, weight, 0.)
            mean_error = error / weight.clamp_min(1e-12)
        else:
            # 经典 3DGS 模式：不做贡献审计（结构周期无额外渲染），只靠绝对梯度门限与增殖窗口。
            weight = mean_error = None
        # 无效点不参与旋转/尺度计算，统一在本轮提交时删除。
        valid_ids = torch.where(valid)[0]
        scales = self.get_scaling[valid_ids]
        rotation = self._rotation_matrix(self._rotation[valid_ids])
        spacing = self._voxel_spacing()
        # 面内投影协方差的最大特征值，单位为输出像素平方；无需逐点 eigh。
        projected = rotation[:, :2, :] * scales[:, None, :] / spacing[None, :2, None]
        xx, yy = projected[:, 0].square().sum(-1), projected[:, 1].square().sum(-1)
        xy = (projected[:, 0] * projected[:, 1]).sum(-1)
        sigma_pixels = ((xx+yy+torch.sqrt((xx-yy).square()+4*xy.square()))*.5).clamp_min(0).sqrt()
        observed = self.denom[valid_ids, 0] >= args.structural_min_observations
        gradients = self.xyz_gradient_accum[valid_ids, 0] / self.denom[valid_ids, 0].clamp_min(1)
        eligible = observed & torch.isfinite(gradients)
        positive = gradients[eligible & (gradients > 0)]
        median = positive.median() if positive.numel() else gradients.new_zeros(())
        # 相对中位数门限每轮都选中固定比例的点，导致指数增殖；只在首个周期用它标定一次。
        if densify and self.structural_grad_reference is None and positive.numel():
            self.structural_grad_reference = float(median * args.structural_grad_multiplier)
        threshold = max(args.structural_grad_threshold, self.structural_grad_reference or 0.)
        # 支撑门：W 过小时 E/W 是小数相除、噪声大，不据此增殖；W 不再用于删除，层间点予以保留。
        if weight is None:
            supported = residual_high = torch.ones_like(eligible)
        else:
            supported = weight[valid_ids] >= args.structural_support_min_weight
            residual_high = mean_error[valid_ids] > args.structural_residual_threshold
        high = eligible & (gradients > 0) & (gradients >= threshold) & supported & residual_high
        if not densify:
            high = torch.zeros_like(high)
        # 经典 3DGS Split：生成两个子点，所有轴尺度除以 0.8*N；任一子轴低于尺度下界时不拆分。
        child_count = 2
        split_scale_factor = 0.8*child_count
        child_log = self._scaling[valid_ids] - math.log(split_scale_factor)
        can_split = child_log.min(1).values >= self.log_scale_floor()
        # 不做 Clone：opacity 固定为 1、按深度排序时，副本多被父点遮蔽而成为冗余；patient001 消融中
        # 只分裂优于克隆+分裂（克隆只增加冗余点与 z 向跳变）。误差驱动的增殖只拆分面内尺度大于门限的点。
        high &= (sigma_pixels > args.structural_split_sigma_pixels) & can_split
        # 尺寸超限：强制 Split 而非删除。删除会在均匀亮区留下无父点可补的空洞，
        # 邻点长大后再被删，区域被逐步侵蚀；拆分保留覆盖，同时把尺寸降到 1/1.6。
        oversized = torch.zeros_like(high)
        if iteration >= args.structural_size_split_from_iter:
            if args.structural_size_max_screen_radius > 0:
                oversized |= self.max_radii2D[valid_ids] > args.structural_size_max_screen_radius
            if args.structural_size_max_scale_ratio > 0:
                # 只看面内投影标准差，不限制沿切片法向的拉长；max(bounds) 是病例模型盒半边长。
                max_scale = args.structural_size_max_scale_ratio*max(self.bounds)
                oversized |= self._inplane_sigma(self._scaling[valid_ids], self._rotation[valid_ids]) > max_scale
        # 所有满足门限的点都参与拆分，不使用点数上限或每周期增长预算。
        candidates = torch.where(high)[0]
        split_mask_local = high | oversized
        # 消融开关：关闭 Split 时误差驱动与尺寸超限拆分都不执行（超限点保留原样）。
        if not args.structural_enable_split:
            split_mask_local = torch.zeros_like(split_mask_local)
        split_local = torch.where(split_mask_local)[0]
        split_ids = valid_ids[split_local]
        sources = split_ids.repeat(child_count)
        new_data = {key: getattr(self, name)[sources]
                    for key, name in self.parameter_names.items()}
        if len(split_ids):
            # 与经典 3DGS 一致：在父点局部坐标按三轴标准差采样，再旋转到模型坐标。
            stds = scales[split_local].repeat(child_count, 1)
            samples = torch.normal(mean=torch.zeros_like(stds), std=stds)
            rotations = rotation[split_local].repeat(child_count, 1, 1)
            new_data['xyz'] += (rotations @ samples[:, :, None]).squeeze(-1)
            # 误差驱动的 Split 已由 can_split 保证子点不低于下界；尺寸超限的针状点最短轴截到下界。
            new_data['scaling'] = (stds/split_scale_factor).log().clamp_min(self.log_scale_floor())

        # 不按贡献质量删除：W 只统计真实层，位于层间、服务插层渲染的点 W 天然很小。
        # 观测框剪枝：中心超出观测体积半个体素以上的点对任何像素都没有意义。
        limit = self._observation_box() + .5*spacing
        prune_box = valid & (self._xyz.abs() > limit).any(-1)
        prune_new = (new_data['xyz'].abs() > limit).any(-1)
        prune_old = invalid | prune_box
        split_mask = torch.zeros_like(invalid)
        split_mask[split_ids] = True
        removed = prune_old | split_mask
        survivors = torch.where(~removed)[0]
        kept_new = ~prune_new
        sources = sources[kept_new]
        if len(survivors)+len(sources) < args.structural_min_points:
            raise FloatingPointError('剪枝后点数少于 structural_min_points，请检查剪枝阈值')
        changed = bool(removed.any()) or len(sources) > 0
        if changed:
            data = {key: torch.cat((getattr(self, name)[survivors], new_data[key][kept_new]), 0)
                    for key, name in self.parameter_names.items()}
            self._replace_structure(survivors, sources, data)
        self._reset_statistics()
        kept_old = ~split_mask & ~invalid
        report = dict(before=before, after=len(self._xyz), split=len(split_ids),
                      pruned=int((prune_old & ~split_mask).sum())+int(prune_new.sum()),
                      split_size=int(oversized.sum()),
                      pruned_box=int((prune_box & kept_old).sum()),
                      eligible=int(eligible.sum()),
                      residual_high=int(residual_high.sum()) if weight is not None else 0,
                      selected=len(candidates), grad_median=float(median), grad_threshold=float(threshold),
                      weight_median=float(weight[valid_ids].median()) if weight is not None else float('nan'),
                      error_median=float(mean_error[valid_ids].median()) if weight is not None else float('nan'),
                      sigma_pixels_median=float(sigma_pixels.median()))
        print(f'[Structure {iteration}] {report}')
        return report

    # ==================== 加和切片模型（additive_moment_v1）====================
    @torch.no_grad()
    def add_slice_statistics(self, outputs):
        """逐次累积真实切片的结构统计；outputs 为 [(render 输出, 层索引 t)]，插层平面不得传入。

        只统计本切片上几何幅值 a ≥ structural_observe_amplitude 的点（确实落在该层上）：
        g_xy = ‖∂L/∂means2D‖（像素单位面内梯度，对应经典 3DGS 屏幕空间梯度），
        h = n·∂L/∂μ（法向位置梯度），并按层记带符号和以判断上下层是否把中心往相反方向拉。
        """
        threshold = self.structural_args.structural_observe_amplitude
        for output, index in outputs:
            radii = output['radii']
            visible = radii > 0
            self.max_radii2D[visible] = torch.maximum(self.max_radii2D[visible],
                                                      radii[visible].to(self.max_radii2D))
            grad2D, grad3D = output['means2D'].grad, output['means3D'].grad
            if grad2D is None or grad3D is None:
                continue
            observed = visible & (output['amplitudes'] >= threshold)
            grad_xy = torch.linalg.vector_norm(grad2D[:, :2], dim=-1)
            normal = grad3D @ output['slice_normal'].to(grad3D)
            self.xyz_gradient_accum[:, 0] += torch.where(observed, grad_xy, 0.)
            self.normal_grad_abs += torch.where(observed, normal.abs(), 0.)
            self.normal_grad_slice[:, index] += torch.where(observed, normal, 0.)
            self.slice_observed[:, index] += observed
            self.denom[:, 0] += observed

    def _normal_variances(self, scales, rotation):
        """沿层法向 z 的边缘方差 c = Σ_zz 与条件方差 cc = c - bᵀA⁻¹b = 1/[Σ⁻¹]_zz。

        Σ = R S² Rᵀ ⇒ Σ_zz = Σ_j R_zj² s_j²，[Σ⁻¹]_zz = Σ_j R_zj²/s_j²（Schur 补恒等式），无需求逆。
        """
        weight = rotation[:, 2, :].square()
        return (weight*scales.square()).sum(-1), 1./(weight/scales.square()).sum(-1)

    def _nearest_slice_amplitude(self, z, scales, rotation, slice_sigma):
        """中心在 z 的高斯在最近真实层上的幅值 a = sqrt(cc/(cc+s²))·exp(-p²/2(c+s²))，即全部真实层上的最大值。"""
        slices = self.geometry['shape_shw'][0]
        dz = float(self._voxel_spacing()[2])
        index = (z/dz + (slices-1)/2).round().clamp(0, slices-1)
        depth = z - (index - (slices-1)/2)*dz
        c, cc = self._normal_variances(scales, rotation)
        s2 = slice_sigma*slice_sigma
        return torch.sqrt(cc/(cc+s2))*torch.exp(-.5*depth.square()/(c+s2))

    def _moment_split(self, ids, axes, ratio, rotation):
        """沿主轴 e_j 对称一分为二，严格保持总信号、质心与协方差（0/1/2 阶矩，3 阶矩由对称性为 0）：

            μ± = μ ± δ e_j，δ = σ_j sqrt(1 - r²)，σ_j → r σ_j，其余轴与旋转不变，I' = I/(2r)。
        两子代之和与父高斯只差 4 阶小量 -I·ĝ_{σ'}(k)·k⁴δ⁴/12（沿分裂轴的频域），渲染为线性算子，
        故任意层位、层厚下切片图像在低频处保持不变。ids/axes/ratio 均为 [M]。
        """
        rows = torch.arange(len(ids), device=ids.device)
        sigma = self.get_scaling[ids, axes]
        direction = rotation[ids][rows, :, axes]
        offset = (sigma*torch.sqrt(1-ratio.square()))[:, None]*direction
        scaling = self._scaling[ids].clone()
        scaling[rows, axes] += torch.log(ratio)
        intensity = self.intensity_parameter(self.get_intensity[ids]/(2*ratio[:, None]))
        return dict(xyz=torch.cat((self._xyz[ids]+offset, self._xyz[ids]-offset)),
                    intensity=intensity.repeat(2, 1), scaling=scaling.repeat(2, 1),
                    rotation=self._rotation[ids].repeat(2, 1))

    @torch.no_grad()
    def structural_step_additive(self, images, iteration, render_kwargs=None, similarity_kwargs=None,
                                 densify=True):
        """加和模型的结构操作：解析剪枝 P1–P3 + 矩匹配分裂 D1/D2 + 尺寸强制分裂 D3（不做 Clone）。

        全部判据来自训练时累积的梯度统计与闭式足迹，结构周期不做额外渲染。densify=False 时
        只执行剪枝与尺寸强制分裂。images/similarity_kwargs 仅为与 structural_step 接口一致。
        """
        args = self.structural_args
        before = len(self._xyz)
        invalid = self._invalid_point_mask()
        valid = ~invalid
        if int(valid.sum()) < args.structural_min_points:
            raise FloatingPointError('有效点少于 structural_min_points，无法安全调整结构')
        slice_sigma = float(render_kwargs['raster_options']['slice_sigma'])
        spacing = self._voxel_spacing()
        # 无效点行可能含 NaN，统一用 valid 屏蔽；scales 列 j 对应旋转矩阵第 j 列主轴
        scales = torch.where(valid[:, None], self.get_scaling, 1.)
        rotation = self._rotation_matrix(torch.where(valid[:, None], self._rotation,
                                                     self._rotation.new_tensor([1., 0., 0., 0.])))
        intensity = self.get_intensity[:, 0]

        # 面内投影：各主轴面内长度（像素）与投影协方差最大标准差（像素）；法向：各主轴沿 z 的投影长度
        projected = rotation[:, :2, :]*scales[:, None, :]/spacing[None, :2, None]
        axis_pixels = torch.linalg.vector_norm(projected, dim=1)
        axis_normal = (rotation[:, 2, :]*scales).abs()
        xx, yy = projected[:, 0].square().sum(-1), projected[:, 1].square().sum(-1)
        xy = (projected[:, 0]*projected[:, 1]).sum(-1)
        sigma_pixels = ((xx+yy+torch.sqrt((xx-yy).square()+4*xy.square()))*.5).clamp_min(0).sqrt()

        # ---------- 剪枝 ----------
        # P1 解析不可见：任意层位 p 上 a ≤ a_max = sqrt(cc/(cc+s²))，|I|·a_max < ε 即在任何平面（含插层）
        # 对任何像素的贡献都低于量化精度；对应经典 3DGS 的 opacity 剪枝，但与视角无关
        _, cc = self._normal_variances(scales, rotation)
        peak = intensity*torch.sqrt(cc/(cc+slice_sigma**2))
        invisible = valid & (peak < args.structural_prune_intensity)
        # P2 观测框：面内超出半个像素；法向再放宽 2s（层厚使框外 2s 内的点仍对首末层有贡献）
        limit = self._observation_box() + .5*spacing
        limit[2] += 2*slice_sigma
        prune_box = valid & (self._xyz.abs() > limit).any(-1)
        keep = valid & ~invisible & ~prune_box

        # ---------- 统计量与冻结门限 ----------
        denom = self.denom[:, 0]
        gradients = self.xyz_gradient_accum[:, 0]/denom.clamp_min(1)
        normal_mean = self.normal_grad_abs/denom.clamp_min(1)
        # 冲突比 ρ = 1 - |Σ_t H_t|/Σ_t |H_t|：逐层先求和再比较，层内的时间振荡相互抵消而不计为冲突
        layered = self.normal_grad_slice
        conflict = 1 - layered.sum(1).abs()/layered.abs().sum(1).clamp_min(1e-30)
        multi_slice = (self.slice_observed > 0).sum(1) >= 2

        eligible = keep & (denom >= args.structural_min_observations) & torch.isfinite(gradients)
        positive = gradients[eligible & (gradients > 0)]
        median = positive.median() if positive.numel() else gradients.new_zeros(())
        if densify and self.structural_grad_reference is None and positive.numel():
            self.structural_grad_reference = float(median*args.structural_grad_multiplier)
        threshold = max(args.structural_grad_threshold, self.structural_grad_reference or 0.)
        normal_ok = keep & multi_slice & torch.isfinite(normal_mean)
        positive_normal = normal_mean[normal_ok & (normal_mean > 0)]
        normal_median = positive_normal.median() if positive_normal.numel() else normal_mean.new_zeros(())
        if densify and self.structural_normal_reference is None and positive_normal.numel():
            self.structural_normal_reference = float(normal_median*args.structural_normal_grad_multiplier)
        normal_threshold = (self.structural_normal_reference if self.structural_normal_reference is not None
                            else math.inf)

        # ---------- 增殖候选（每点每周期至多分裂一次，优先级 D3 > D2 > D1）----------
        floor = self.log_scale_floor()
        inplane_axis = axis_pixels.argmax(1)
        normal_axis = axis_normal.argmax(1)
        rows = torch.arange(len(scales), device=scales.device)
        ratio, size_ratio = args.structural_split_ratio, args.structural_size_split_ratio
        log_scales = scales.log()
        # D3 尺寸强制：面内过大的点沿面内最长轴矩匹配对半拆（r_size），不受梯度门控与 densify 开关限制
        oversized = torch.zeros_like(keep)
        if iteration >= args.structural_size_split_from_iter:
            if args.structural_size_max_screen_radius > 0:
                oversized |= self.max_radii2D > args.structural_size_max_screen_radius
            if args.structural_size_max_sigma_pixels > 0:
                oversized |= sigma_pixels > args.structural_size_max_sigma_pixels
        oversized &= keep
        # D2 法向：被 ≥2 层观测、法向拉力大且上下层把中心往相反方向拉（中心被卡住，一阶移动无法同时满足），
        # 且两个子代各自仍落在某真实层上（a ≥ 观测门限），避免在层间造出无监督的点
        normal_split_ok = log_scales[rows, normal_axis] + math.log(ratio) >= floor
        delta = scales[rows, normal_axis]*math.sqrt(1-ratio**2)*rotation[rows, 2, normal_axis]
        child_scales = scales.clone()
        child_scales[rows, normal_axis] *= ratio
        observable = torch.ones_like(keep)
        for sign in (1., -1.):
            amplitude = self._nearest_slice_amplitude(self._xyz[:, 2]+sign*delta, child_scales, rotation,
                                                      slice_sigma)
            observable &= amplitude >= args.structural_observe_amplitude
        normal_high = (normal_ok & (normal_mean >= normal_threshold)
                       & (conflict > args.structural_normal_conflict) & normal_split_ok & observable)
        # D1 面内：平均面内梯度超过冻结门限，且面内尺度大于分裂下限
        inplane_split_ok = log_scales[rows, inplane_axis] + math.log(ratio) >= floor
        inplane_high = (eligible & (gradients > 0) & (gradients >= threshold)
                        & (sigma_pixels > args.structural_split_sigma_pixels) & inplane_split_ok)
        if not densify:
            normal_high = inplane_high = torch.zeros_like(keep)
        normal_high &= ~oversized
        inplane_high &= ~oversized & ~normal_high
        if not args.structural_enable_split:
            oversized = normal_high = inplane_high = torch.zeros_like(keep)

        split_mask = oversized | normal_high | inplane_high
        split_ids = torch.where(split_mask)[0]
        axes = torch.where(normal_high, normal_axis, inplane_axis)[split_ids]
        ratios = torch.full_like(scales[:, 0], ratio)
        ratios[oversized] = size_ratio
        ratios = ratios[split_ids]
        new_data = self._moment_split(split_ids, axes, ratios, rotation)
        # 尺寸强制拆分的子轴若低于尺度下界则截到下界（误差驱动分裂已由 *_split_ok 保证）
        new_data['scaling'] = new_data['scaling'].clamp_min(floor)

        # ---------- 提交 ----------
        prune_new = (new_data['xyz'].abs() > limit).any(-1)
        removed = invalid | invisible | prune_box | split_mask
        survivors = torch.where(~removed)[0]
        kept_new = ~prune_new
        sources = split_ids.repeat(2)[kept_new]
        if len(survivors)+len(sources) < args.structural_min_points:
            raise FloatingPointError('剪枝后点数少于 structural_min_points，请检查剪枝阈值')
        if bool(removed.any()) or len(sources) > 0:
            data = {key: torch.cat((getattr(self, name)[survivors], new_data[key][kept_new]), 0)
                    for key, name in self.parameter_names.items()}
            self._replace_structure(survivors, sources, data)
        self._reset_statistics()
        report = dict(before=before, after=len(self._xyz), split=len(split_ids),
                      split_size=int(oversized.sum()), split_normal=int(normal_high.sum()),
                      selected=int(inplane_high.sum()),
                      pruned=int((invalid | invisible | prune_box).sum())+int(prune_new.sum()),
                      pruned_box=int(prune_box.sum()), pruned_invisible=int(invisible.sum()),
                      eligible=int(eligible.sum()),
                      grad_median=float(median), grad_threshold=float(threshold),
                      normal_median=float(normal_median), normal_threshold=float(normal_threshold),
                      sigma_pixels_median=float(sigma_pixels[valid].median()))
        print(f'[Structure {iteration}] {report}')
        return report

    def InitializeGaussians(self, num_pts=50000, bounds=(1.,1.,1.), roi_mask=None,
                            intensity_init=0.6410473958869391, roi_layout='SHW',
                            roi_jitter_factor=.5, initial_z_scale=2., initial_distance_floor=1e-7):
        device = torch.device('cuda', torch.cuda.current_device())
        low = np.negative(np.asarray(bounds, dtype=np.float32))
        high = np.asarray(bounds, dtype=np.float32)
        if roi_mask is None:
            points = torch.from_numpy(np.random.uniform(low, high, (num_pts, 3))).float().to(device)
        else:
            coordinates = roi_mask.detach().to(device).float()
            if coordinates.ndim != 2:
                layout = roi_layout.upper().translate(str.maketrans({'S':'Z','D':'Z','H':'Y','W':'X'}))
                if coordinates.ndim != 3 or sorted(layout) != ['X','Y','Z']:
                    raise ValueError('roi_layout 必须为三维 mask 的 XYZ 轴序')
                order = [layout.index(axis) for axis in 'XYZ']
                shape = coordinates.new_tensor([coordinates.shape[i] for i in order])
                indices = torch.nonzero(coordinates > 0).float()[:, order]
                normalized = torch.where(shape > 1, indices/(shape-1).clamp_min(1), .5)
                coordinates = normalized * torch.as_tensor(high-low,device=device) + torch.as_tensor(low,device=device)
            if coordinates.shape[-1] != 3 or not len(coordinates):
                raise ValueError('ROI 需要非空的模型空间 XYZ 点或非空三维 mask')
            if len(coordinates) < num_pts:
                selected = torch.randint(len(coordinates),(num_pts,),device=device)
            else:
                selected = torch.randperm(len(coordinates),device=device)[:num_pts]
            points = coordinates[selected]
            jitter = points.new_tensor(high-low)/np.cbrt(num_pts)*roi_jitter_factor
            points = points + torch.empty_like(points).uniform_(-1,1)*jitter
            points = points.clamp(min=points.new_tensor(low),max=points.new_tensor(high))
        if not math.isfinite(intensity_init):
            raise ValueError('intensity_init 必须有限')
        rotation = torch.logit(torch.rand(num_pts,4,device=device))
        from simple_knn._C import distCUDA2
        distance = distCUDA2(points).clamp_min(initial_distance_floor)
        log_scale = distance.sqrt().log()[:,None].repeat(1,3)
        log_scale[:,2] += math.log(initial_z_scale)
        self._xyz = nn.Parameter(points)
        self._intensity = nn.Parameter(torch.full((num_pts,1),intensity_init,device=device))
        self._scaling = nn.Parameter(log_scale)
        self._rotation = nn.Parameter(rotation)


    def construct_attributes(self):
        return ['x', 'y', 'z', 'intensity'] + [f'scale_{i}' for i in range(3)] + [f'rot_{i}' for i in range(4)]

    def save_ply(self, path):
        """将模型的高斯点参数和全局时序参数保存为 .ply 文件"""
        folder_path = os.path.dirname(path)
        if folder_path and not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 1. 提取所有点云属性并展平为 2D 矩阵 (N, C)
        xyz = self._xyz.detach().cpu().numpy()
        # 单通道强度直接写入 intensity；不写入 opacity。
        # 加和模型写入 softplus 激活后的强度，PLY 字段始终是物理强度
        intensity = self.get_intensity.detach().flatten(start_dim=1).contiguous().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        # [SIGMA-DISABLED] 处理 (N, 1, 1) 的 sigma 变为 (N, 1)
        # sigma = self.sigma.detach().flatten(start_dim=1).cpu().numpy()

        # [SIGMA-DISABLED] 拼接所有 attributes 为 (N, Total_Channels)，移除了 sigma
        attributes = np.concatenate((xyz, intensity, scale, rotation), axis=1) # , sigma)

        # 构建类型结构
        dtype_full = [(attribute, 'f4') for attribute in self.construct_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')

        # 2. 处理全局的时序/形变参数 time_func
        # 如果 time_func 尚未初始化或为空，给一个默认的零矩阵防止报错
        if self.time_func.numel() == 0:
            time_tensor = torch.zeros((1, 1))
        else:
            time_tensor = self.time_func.detach().cpu().view(1, -1)

        time_dims = time_tensor.shape[1]
        dtype_time = [(f'time_func_{i}', 'f4') for i in range(time_dims)]

        time_arr = np.empty(1, dtype=dtype_time)
        time_arr[:] = list(map(tuple, time_tensor.numpy()))
        time_el = PlyElement.describe(time_arr, 'time')

        # 写入双元素 (vertex + time) PLY 文件
        PlyData([el, time_el]).write(path)
        print(f"成功保存模型到: {path}")

    def load_ply(self, path, legacy_color_policy="strict", device="cuda"):
        """从 .ply 文件读取参数并加载到高斯模型中"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到指定的 PLY 文件: {path}")

        plydata = PlyData.read(path)
        vertex = plydata['vertex']

        # 1. 解析基础坐标 (N, 3)
        xyz = np.stack((np.asarray(vertex["x"]),
                        np.asarray(vertex["y"]),
                        np.asarray(vertex["z"])), axis=1)

        # 旧 opacity 通常为 logit，不能静默丢弃并视为固定 1 的等价模型。
        names = {p.name for p in vertex.properties}
        if 'opacity' in names:
            raise ValueError('PLY 含旧 opacity 字段，需显式迁移后重新拟合，不能自动加载为固定 opacity=1')
        if 'intensity' in names:
            intensity = np.asarray(vertex['intensity'], dtype=np.float32)[:, None]
        else:
            color_names = sorted((n for n in names if n.startswith('color_')), key=lambda n: int(n.split('_')[-1]))
            if not color_names:
                raise ValueError("PLY has neither intensity nor legacy color_* fields")
            color = np.stack([np.asarray(vertex[n]) for n in color_names], axis=1)
            if legacy_color_policy == 'first':
                intensity = color[:, :1]
            elif legacy_color_policy == 'mean':
                intensity = color.mean(axis=1, keepdims=True)
            elif legacy_color_policy == 'strict':
                if not np.allclose(color, color[:, :1], rtol=1e-5, atol=1e-6):
                    raise ValueError("Legacy color channels differ: explicitly select legacy_color_policy='first' or 'mean'")
                intensity = color[:, :1]
            else:
                raise ValueError("legacy_color_policy must be strict, first or mean")
            warnings.warn(f"Loaded legacy color_* as intensity using {legacy_color_policy}", UserWarning)

        # 4. 解析 Scale (N, 3)
        scale_names = sorted([p.name for p in vertex.properties if p.name.startswith("scale_")], key=lambda n: int(n.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(vertex[attr_name])

        # 解析原始 wxyz 四元数 (N, 4)
        rot_names = sorted([p.name for p in vertex.properties if p.name.startswith("rot_")], key=lambda n: int(n.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(vertex[attr_name])

        # [SIGMA-DISABLED] 6. 解析 Sigma 并恢复为特定的 (N, 1, 1) 维度
        # sigma = np.asarray(vertex["sigma"])[..., np.newaxis, np.newaxis]

        if scales.shape[1] != 3 or rots.shape[1] != 4:
            raise ValueError("PLY requires scale_0..2 and rot_0..3")
        if not all(np.isfinite(a).all() for a in (xyz, intensity, scales, rots)):
            raise ValueError("PLY contains nonfinite attributes")
        # 转换为需要梯度的 PyTorch nn.Parameter
        self._xyz = torch.nn.Parameter(torch.tensor(xyz, dtype=torch.float, device=device).requires_grad_(True))
        intensity = torch.tensor(intensity, dtype=torch.float, device=device)
        if self.intensity_softplus:
            # PLY 存激活后强度；softplus 逆要求严格正值，非正值说明文件与渲染语义不符
            if not bool((intensity > 0).all()):
                raise ValueError('加和模型 PLY 的强度必须为正')
            intensity = self.intensity_parameter(intensity)
        self._intensity = torch.nn.Parameter(intensity.requires_grad_(True))
        self._scaling = torch.nn.Parameter(torch.tensor(scales, dtype=torch.float, device=device).requires_grad_(True))
        self._rotation = torch.nn.Parameter(torch.tensor(rots, dtype=torch.float, device=device).requires_grad_(True))
        # self.sigma = torch.nn.Parameter(torch.tensor(sigma, dtype=torch.float, device=device).requires_grad_(True))  # [SIGMA-DISABLED]

        # 7. 解析全局时序属性 `time_func`
        if 'time' in plydata:
            time_el = plydata['time']
            time_names = [p.name for p in time_el.properties if p.name.startswith("time_func_")]
            time_val = np.zeros((len(time_names), 1))
            for idx, attr_name in enumerate(time_names):
                time_val[idx, 0] = np.asarray(time_el[attr_name])[0]
            self.time_func = torch.nn.Parameter(torch.tensor(time_val, dtype=torch.float, device=device).requires_grad_(True))
        else:
            self.time_func = torch.empty(0, device=device)

        print(f"成功加载 {xyz.shape[0]} 个高斯顶点参数。")
