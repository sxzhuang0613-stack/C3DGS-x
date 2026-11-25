"""spacing 驱动的初值；固定几何系数求解与正常 Adam 训练相互独立。"""
import hashlib
import math

import torch
import torch.nn.functional as F
from torch import nn

from utils.Renderer import render
from .utils import covariance_from_scaling_rotation


def _sampling_maps(images, args):
    """分位数归一化图与膨胀前景 [S,H,W]；只产生采样/覆盖权重，训练强度始终保持 dataloader 原输出。"""
    low,high = torch.quantile(images.flatten(),images.new_tensor(args.init_sampling_quantiles))
    normalized = ((images-low)/(high-low)).clamp(0,1) if high > low else torch.zeros_like(images)
    foreground = (normalized > args.init_foreground_threshold).float()[:,None]
    radius = args.init_foreground_dilation
    foreground = F.max_pool2d(foreground,2*radius+1,stride=1,padding=radius)[:,0]
    return normalized, foreground


def _sample_positions(images, spacing, box, count, args, generator, guided=True,
                      slice_index=None, num_slices=None):
    """在观测层上采样初始中心（病例局部坐标）；images [N,H,W] 为第 slice_index 层（完整栈共 num_slices 层）。"""
    slices, height, width = images.shape
    slice_index = torch.arange(slices) if slice_index is None else torch.as_tensor(slice_index)
    num_slices = slices if num_slices is None else num_slices
    device = images.device
    uniform_count = round(count*args.init_uniform_fraction) if guided else count
    foreground_count = min(round(count*args.init_foreground_fraction),count-uniform_count) if guided else 0
    edge_count = count-uniform_count-foreground_count
    positions = []
    # 每层分配均匀份额，余数随机分给若干层，避免少量重采样点全部落在前几层。
    extra = [0]*slices
    if uniform_count%slices:
        for layer in torch.randperm(slices,device=device,generator=generator)[:uniform_count%slices].tolist():
            extra[layer] = 1
    # 矩形网格用随机子集，避免尾行偏置。
    for layer in range(slices):
        n = uniform_count//slices + extra[layer]
        if not n:
            continue
        nx = math.ceil(math.sqrt(n*width/height))
        ny = math.ceil(n/nx)
        cells = torch.randperm(nx*ny,device=device,generator=generator)[:n]
        jitter = (torch.rand(n,2,device=device,generator=generator)*2-1)*args.init_grid_jitter
        u = -.5+(cells%nx+.5+jitter[:,0])*width/nx
        v = -.5+(cells//nx+.5+jitter[:,1])*height/ny
        positions.append(torch.stack((torch.full_like(u,layer),v,u),1))
    if foreground_count or edge_count:
        normalized, foreground = _sampling_maps(images, args)
        radius = math.ceil(args.init_edge_truncate*args.init_edge_sigma)
        coordinate = torch.arange(-radius,radius+1,device=device,dtype=images.dtype)
        kernel = torch.exp(-.5*(coordinate/args.init_edge_sigma).square())
        kernel /= kernel.sum()
        window = (kernel[:,None]*kernel[None,:])[None,None]
        smooth = F.conv2d(F.pad(normalized[:,None],(radius,)*4,mode='replicate'),window)[:,0]
        padded = F.pad(smooth[:,None],(1,1,1,1),mode='replicate')[:,0]
        gx = (padded[:,1:-1,2:]-padded[:,1:-1,:-2])/2
        gy = (padded[:,2:,1:-1]-padded[:,:-2,1:-1])/2
        edge = torch.sqrt(gx.square()+gy.square())*foreground
        valid = edge[foreground.bool()]
        threshold = torch.quantile(valid,args.init_edge_quantile) if valid.numel() else edge.new_zeros(())
        edge *= edge >= threshold
        for n,weights in ((foreground_count,foreground),(edge_count,edge)):
            if not n:
                continue
            weights = weights.flatten()
            if not bool(weights.sum() > 0):
                weights = torch.ones_like(weights)
            indices = torch.multinomial(weights,n,replacement=True,generator=generator)
            jitter = (torch.rand(n,2,device=device,generator=generator)*2-1)*args.init_grid_jitter
            positions.append(torch.stack((indices//(height*width),
                (indices//width)%height+jitter[:,1],indices%width+jitter[:,0]),1))
    locations = torch.cat(positions)
    # 观测图序号 → 完整栈层号，z 按完整栈居中，使 holdout 时训练层位置与全层训练一致
    layers = slice_index.to(device)[locations[:,0].long()].to(locations)
    z = (layers-(num_slices-1)/2)*spacing[2]
    # 观测框按每层代表厚度 sz 的薄片定义，首末层与中间层的法向抖动对称。
    lower = (z-args.init_normal_jitter*spacing[2]).clamp_min(-box[2])
    upper = (z+args.init_normal_jitter*spacing[2]).clamp_max(box[2])
    z = lower+(upper-lower)*torch.rand(count,device=device,generator=generator)
    return torch.stack(((locations[:,2]+.5-width/2)*spacing[0],
                        (locations[:,1]+.5-height/2)*spacing[1],z),1)


def _mean_knn_sq_distance(points, neighbors=3, chunk=2048):
    """每点到最近 neighbors 个其他点的平方距离均值 [N]，与 simple_knn.distCUDA2 定义一致；分块以支持 CPU。"""
    result = []
    for start in range(0, len(points), chunk):
        distance = torch.cdist(points[start:start+chunk], points).square()
        # 第一个最小值是点自身；点数不足时取可用邻居。
        k = min(neighbors+1, len(points))
        nearest = distance.topk(k, dim=1, largest=False).values[:, 1:]
        result.append(nearest.mean(1) if nearest.shape[1] else distance.new_ones(len(distance)))
    return torch.cat(result).clamp_min(1e-7)


class IntensityOperator:
    """固定几何、固定 opacity 的 alpha 颜色线性算子及伴随；不构造稠密响应矩阵。"""
    def __init__(self, model, images, rendering, slice_index=None, num_slices=None):
        # images [N,H,W] 为完整栈（共 num_slices 层）中第 slice_index 层的观测
        self.model, self.images = model, images
        self.slice_index = list(range(len(images))) if slice_index is None else [int(k) for k in slice_index]
        self.num_slices = len(images) if num_slices is None else num_slices
        self.rendering = dict(rendering,raster_options=dict(rendering['raster_options'],background=0.))
        # CG 期间几何固定，协方差只构造一次；xyz 仍共享数据以支持未观测点重采样。
        self.parameters = dict(
            xyz=model.get_xyz.detach(),
            covariance=covariance_from_scaling_rotation(
                model.get_scaling.detach(), model._rotation.detach()),
        )

    def view(self, coefficients, index):
        return render(self.model,list(self.images.shape[-2:]),self.slice_index[index],2,self.num_slices,
                      colors=coefficients[:,None],**self.parameters,**self.rendering)

    @torch.no_grad()
    def forward(self, coefficients, visibility=False):
        coefficients = coefficients.to(self.images)
        outputs = [self.view(coefficients,i) for i in range(len(self.images))]
        prediction = torch.stack([output['render'] for output in outputs]).double()
        if visibility:
            # radii 仅代表候选范围；单位伴随的正值才说明高斯实际参与像素合成。
            visible = self.adjoint(torch.ones_like(prediction)) > 0
            return prediction, visible
        return prediction

    def adjoint(self, values):
        # W=1/(S*H*W)，与全病例均方误差归一化一致；只保留系数梯度。
        with torch.enable_grad():
            coefficients = self.model._intensity.detach().flatten().clone().requires_grad_()
            result = coefficients.new_zeros(coefficients.shape,dtype=torch.float64)
            for index in range(len(self.images)):
                prediction = self.view(coefficients,index)['render']
                incoming = (values[index]/self.images.numel()).to(prediction)
                result += torch.autograd.grad(prediction,coefficients,grad_outputs=incoming)[0].double()
        return result


@torch.no_grad()
def initialize_from_images(model, images, geometry, args, rendering, case_name,
                           slice_index=None, num_slices=None):
    """由观测层初始化高斯；holdout 时 images 只含训练层 [N,H,W]，slice_index 为其完整栈层号。"""
    if not torch.isfinite(images).all():
        raise ValueError('初始化图像含非有限值')
    seed = int.from_bytes(hashlib.blake2b(
        f'{args.random_seed}:{case_name}'.encode(),digest_size=8).digest(),'little')
    generator = torch.Generator(device=images.device).manual_seed(seed)
    spacing = images.new_tensor(geometry['spacing_xyz'])/geometry['scale_mm']
    height, width = images.shape[-2:]
    slices = len(images) if num_slices is None else num_slices
    # 观测体积半边长 [W*sx,H*sy,S*sz]/2（S 为完整栈层数）；bounds 只定义渲染坐标归一化，不在此改变。
    box = spacing*images.new_tensor([width,height,slices])/2
    layout = dict(slice_index=slice_index, num_slices=num_slices)
    count = args.num_pts
    points = _sample_positions(images,spacing,box,count,args,generator,**layout)
    # 经典 3DGS 初始尺度：各向同性，sigma=sqrt(到最近 3 个点平方距离的均值)；
    # 不为层间覆盖设置法向尺度，下界与训练一致为 min_sigma_pixels 个面内像素。
    floor = args.min_sigma_pixels*spacing[:2].min()
    def classic_log_scales(centers):
        logs = _mean_knn_sq_distance(centers).sqrt().clamp_min(floor).log()[:,None].repeat(1,3)
        if not bool(((logs >= args.log_scale_min)&(logs <= args.log_scale_max)).all()):
            raise ValueError('初始化物理尺度超出配置的 log_scale 范围')
        return logs
    logs = classic_log_scales(points)
    options = rendering['raster_options']
    rotations = images.new_zeros(count,4)
    rotations[:,0] = 1
    model._xyz = nn.Parameter(points)
    model._scaling = nn.Parameter(logs)
    model._rotation = nn.Parameter(rotations)
    model._intensity = nn.Parameter(images.new_ones(count,1))
    model.geometry_reference_count = count
    operator = IntensityOperator(model,images,rendering,**layout)
    dropped = 0
    for attempt in range(args.init_support_retries+1):
        field,visible = operator.forward(torch.ones(count,device=images.device),visibility=True)
        if bool(visible.all()):
            break
        if attempt == args.init_support_retries:
            # 细小 kNN 尺度的点若离切片较远会被候选半径剔除；重算尺度也可能让个别邻点失去贡献，
            # 重试后剩余的少量点对任何像素无贡献，直接丢弃不影响拟合；大量剩余仍视为配置错误。
            dropped = int((~visible).sum())
            if dropped > max(10, .01*count):
                raise ValueError(f'初始化仍有 {dropped} 个无像素贡献的高斯，请检查点数、采样和初始尺度')
            for name in ('_xyz','_scaling','_rotation','_intensity'):
                setattr(model, name, nn.Parameter(getattr(model, name)[visible]))
            count = len(model._xyz)
            model.geometry_reference_count = count
            operator = IntensityOperator(model,images,rendering,**layout)
            field = operator.forward(torch.ones(count,device=images.device))
            break
        model._xyz[~visible] = _sample_positions(images,spacing,box,int((~visible).sum()),args,generator,
                                                 guided=False,**layout)
        # 尺度由最终位置决定：重采样后重算 kNN 尺度，并按新几何重建线性算子。
        model._scaling.copy_(classic_log_scales(model._xyz))
        operator = IntensityOperator(model,images,rendering,**layout)
    additive = rendering.get('render_model') == 'additive_slab_v1'
    if additive:
        # 加和模型 C = bg + A·I：背景与高斯无关，为常数
        background_image = options['background']
    else:
        # 单位颜色、零背景响应 A(1)=1-T；背景贡献在所有像素均为 T*b。
        background_image = (1-field)*options['background']
    target = images.double()-background_image
    field_norm = field.square().mean()
    if not bool(torch.isfinite(field_norm) & (field_norm > 0)):
        raise FloatingPointError('初始化的单位系数响应为零或非有限')
    common = (field*target).mean()/field_norm
    baseline = common.expand(count).clone()
    initial_prediction = operator.forward(baseline)
    ridge = args.init_ridge_relative*field_norm/count
    residual = operator.adjoint(target-initial_prediction)
    direction = residual.clone()
    residual_norm = residual@residual
    initial_norm = residual_norm.clone()
    stopping_norm = residual_norm*args.init_cg_rtol**2
    delta = baseline.new_zeros(count)
    steps = 0
    for _ in range(args.init_fit_steps):
        if bool(residual_norm <= stopping_norm):
            break
        product = operator.adjoint(operator.forward(direction))+ridge*direction
        curvature = direction@product
        if not bool(torch.isfinite(curvature) & (curvature > 0)):
            raise FloatingPointError('初始化线性求解出现非正或非有限曲率')
        alpha = residual_norm/curvature
        delta += alpha*direction
        residual -= alpha*product
        next_norm = residual@residual
        direction = residual+(next_norm/residual_norm)*direction
        residual_norm = next_norm
        steps += 1
    coefficients = (baseline+delta).to(images)
    if not bool(torch.isfinite(coefficients).all()):
        raise FloatingPointError('初始化强度求解出现 NaN/Inf')
    clamped = 0
    if additive:
        # 强度经 softplus 参数化须严格为正：无约束最小二乘的非正解截到可见下限 ε_I，
        # 训练若不能把它们提亮，首个结构周期由 P1 解析不可见剪枝删除
        floor = max(args.structural_prune_intensity, 1e-6)
        clamped = int((coefficients < floor).sum())
        coefficients = coefficients.clamp_min(floor).double()
        # 直接截断会破坏拟合（实测 40% 系数非正时 MSE 反而高于常数基线），故接非负最小二乘细化：
        # 加和模型 A≥0、目标 y≥0，Lee–Seung 乘性更新 I ← I ⊙ Aᵀy / (AᵀA I + λI) 保持 I>0 且目标单调下降；
        # 每步一次前向 + 一次伴随，与 CG 一步同价
        numerator = operator.adjoint(target.clamp_min(0))
        for _ in range(args.init_nonneg_steps):
            denominator = operator.adjoint(operator.forward(coefficients))+ridge*coefficients
            coefficients = (coefficients*numerator.clamp_min(0)/denominator.clamp_min(1e-30)).clamp_min(floor)
        coefficients = coefficients.to(images)
    model._intensity.copy_(model.intensity_parameter(coefficients[:,None]))
    prediction = operator.forward(coefficients)
    report = dict(common_intensity=float(common),fit_steps=steps,
                  relative_residual=float(torch.sqrt(residual_norm/initial_norm)) if initial_norm > 0 else 0.,
                  mse_before=float((initial_prediction-target).square().mean()),
                  mse_after=float((prediction-target).square().mean()),
                  count=count, dropped_invisible=dropped, clamped_nonpositive=clamped)
    print(f'[Initialization] {report}')
    return report
