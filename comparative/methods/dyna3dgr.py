"""Dyna3DGR（MICCAI 2025）适配：把原 train.main() 拆成两个阶段。

stage1：只在初始帧 t=0 上拟合静态高斯（d_xyz=0、d_scaling=1、无密度残差），只更新高斯参数，迭代数 = cfg.iters；
        原实现无独立阶段 1，"preset" 时取 full 中静态段的长度 iterations_node_sampling（原 5000）。
full：  原 main() 全流程（总步数 total_iters，原预设 20000）。前 iterations_node_sampling（原 5000）步按 stage1
        静态拟合，之后从该步数接续原联合循环（所选帧、形变网络与高斯同时优化，含原 5000/10000 步的特殊分支）。
与原实现的其余差异：
- 数据改为 Dataloader 的 128×128 层网格，不裁剪、不按标签抠图。原 process_ACDC 把层间距线性重采样到 3.15mm
  后在整个体上算 L1；此处同样把训练层沿 z 线性插值到渲染网格（步长 interp_step_size·sz，代替 3.15mm，
  使整数层位落在网格上），holdout 时只用训练层插值。初始点同原实现，从插值后的初始帧体中采样。
- 渲染网格为 H×W×Dz 的规则网格；原体素化 kernel 只支持规则网格，训练与输出共用同一网格，
  使其支撑盒的离散化在两者间一致。
- 最优状态选择、输出渲染路径（deform.step）同原 val_4ddata。
"""
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

from case_data import use_reference, resolve_iters

REPO_SUBDIR = 'Dyna3DGR'
ORIGINAL_TOTAL_ITERS = 20000


def _setup(case, cfg, scale):
    """按原 main() 构建形变模型与优化参数；scale 用于缩放 full 模式下按步数写死的常数。"""
    use_reference(Path(cfg.references_root) / REPO_SUBDIR)
    import train as dyna
    from scene.deform_model import DeformModel
    from arguments import ModelParams, OptimizationParams, PipelineParams

    parser = ArgumentParser()
    ModelParams(parser)
    op = OptimizationParams(parser)
    PipelineParams(parser)
    opt = op.extract(parser.parse_args([]))

    # 原 main() 的固定设置
    K, deform_type, skinning, hyper_dim, node_num = 3, 'node', False, 8, 12288
    deform = DeformModel(K=K, deform_type=deform_type, skinning=skinning, hyper_dim=hyper_dim,
                         node_num=node_num, d_rot_as_res=True, local_frame=True, with_arap_loss=False,
                         max_d_scale=-1, enable_densify_prune=False, is_scene_static=False)
    # 发布代码缺陷：DeformNetwork.update 读取从未赋值的 progressive_brand_time，原 main() 首次调用
    # deform.update 即 AttributeError；补上 ModelParams 中的默认值 False（不启用渐进时间编码）
    deform.deform.network.progressive_brand_time = False
    steps = lambda n: int(round(n * scale))
    opt.position_lr_init, opt.position_lr_final = 0.0001, 1e-07
    opt.position_lr_delay_steps, opt.position_lr_max_steps = 0, steps(30000)
    opt.opacity_lr, opt.densify_grad_threshold, opt.scaling_lr, opt.rotation_lr = 0.005, 0.0002, 0.0001, 0.0001
    opt.densification_interval, opt.densify_from_iter = steps(500), steps(500)
    opt.deform_lr_scale, opt.deform_lr_max_steps = 0.01, steps(40000)
    opt.iterations_node_sampling, opt.iterations_node_rendering = steps(5000), steps(10000)
    opt.node_warm_up = steps(opt.node_warm_up)
    deform.train_setting(opt)
    return dyna, deform, opt, node_num


def _resample_z(images, case, step):
    """训练层 [N,H,W] 沿 z 线性插值到渲染网格，返回 [H,W,Dz]（Dz=(S-1)/step+1，首末层外取最近训练层）。

    对应原 process_ACDC 中 SimpleITK 线性重采样；第 p 个平面位于连续层索引 k=p·step。
    """
    depth = int(round((case.shape[0] - 1) / step)) + 1
    k = torch.arange(depth, dtype=torch.float32) * step
    c = ((k - float(case.train_index[0])) / case.stride).clamp(0, len(case.train_index) - 1)
    lower = c.floor().long().clamp(max=len(case.train_index) - 1)
    upper = (lower + 1).clamp(max=len(case.train_index) - 1)
    weight = (c - lower)[:, None, None]
    volume = images[lower] * (1 - weight) + images[upper] * weight          # [Dz,H,W]
    return volume.permute(1, 2, 0).contiguous()


def fit(case, cfg, method_cfg, out_dir, stage='stage1', **_):
    """stage1：初始帧静态拟合；full：两阶段完整流程。返回渲染网格各平面的预测。"""
    total = method_cfg['total_iters'] if stage == 'full' else ORIGINAL_TOTAL_ITERS
    dyna, deform, opt, node_num = _setup(case, cfg, total / ORIGINAL_TOTAL_ITERS)
    num_slices, height, width = case.shape

    def ground_truth(t):
        """第 t 帧目标体 [H,W,Dz]：训练层 z 向线性插值（与渲染输出逐体素对应）。"""
        return _resample_z(case.frame(t)[case.train_index], case, cfg.interp_step_size).cuda()

    target = ground_truth(0)
    depth = target.shape[2]
    grid = dyna.create_grid_3d(height, width, depth).cuda().unsqueeze(0)

    # 初始点：同原 sample_points_from_cardiac，从初始帧体的非零体素采样，坐标按 (shape-1) 归一化到 [0,1]
    points = dyna.sample_points_from_cardiac(target.cpu().numpy(), node_num)
    deform.deform.init(init_pcl=torch.tensor(points, dtype=torch.float32).cuda(), force_init=True,
                       opt=opt, as_gs_force_with_motion_mask=False, force_gs_keep_all=False)

    def render_static():
        """无形变渲染 [H,W,Dz]。"""
        return dyna.rendering2(deform.deform.as_gaussians, grid, 0, 0, 1.0, None)[0, ..., 0]

    def render_frame(index, total_frame):
        """同原 val_4ddata：经 deform.step（控制节点插值）得到第 index 个时间点的渲染 [H,W,Dz]。"""
        fid = torch.tensor(index / total_frame).cuda()
        time_input = fid.unsqueeze(0).expand(deform.deform.node_num, -1)
        d_values = deform.step(deform.deform.as_gaussians.get_xyz.detach(), time_input, feature=None,
                               motion_mask=deform.deform.as_gaussians.motion_mask)
        d_scaling = torch.mean(d_values['d_scaling'], dim=1, keepdim=True).expand_as(d_values['d_scaling'])
        return dyna.rendering2(deform.deform.as_gaussians, grid, d_values['d_xyz'], 0, d_scaling,
                               d_values['d_density'])[0, ..., 0]

    def snapshot():
        """同原实现：以训练数据上的 L1 选择最优状态（state_dict 含 gs_ 前缀的高斯参数）。"""
        return {k: v.clone().detach() for k, v in deform.deform.state_dict().items()}

    # 阶段 1：初始帧静态拟合；full 时长度为原 iterations_node_sampling
    stage1_iters = (resolve_iters(cfg, opt.iterations_node_sampling) if stage == 'stage1'
                    else opt.iterations_node_sampling)
    best_loss, best_state = float('inf'), None
    for iteration in range(stage1_iters):
        gaussians = deform.deform.as_gaussians
        loss = dyna.l1_loss(render_static(), target)
        loss.backward()
        with torch.no_grad():
            gaussians.optimizer.step()
            gaussians.update_learning_rate(iteration)
            gaussians.optimizer.zero_grad(set_to_none=True)
        if stage == 'stage1' and iteration % 50 == 0:
            with torch.no_grad():
                val_loss = dyna.l1_loss(render_static(), target).item()
            if val_loss < best_loss:
                best_loss, best_state = val_loss, snapshot()
            if iteration % 1000 == 0:
                print(f'  stage1 iter {iteration} L1 {val_loss:.6f}')

    if stage == 'stage1':
        deform.deform.load_state_dict(best_state)
        with torch.no_grad():
            volume = render_static().permute(2, 0, 1).cpu().numpy()
        return dict(query_index=np.arange(depth) * cfg.interp_step_size, volume=volume, frame=0,
                    iterations=stage1_iters)

    # 阶段 2：接续原联合循环（iteration_node_rendering 从 iterations_node_sampling 开始计）
    # 时间输入 t=index/len(frames)，ED 对应 t=0（同原实现以首帧为时间原点）
    ed, es = case.time_indices
    frames = {'ed_to_es': list(range(ed, es + 1)), 'ed_es': [ed, es],
              'all': list(range(case.sequence.shape[1]))}[method_cfg['frames']]
    targets = [ground_truth(t) for t in frames]
    total_frame = len(frames)
    best_loss, best_state = float('inf'), None
    for iteration in range(stage1_iters, total):
        loss = torch.zeros(1).cuda()
        for index in range(total_frame):
            fid = torch.tensor(index / total_frame).cuda()
            gaussians = deform.deform.as_gaussians
            time_input = fid.unsqueeze(0).expand(gaussians.get_xyz.shape[0], -1)
            d_values = deform.deform.query_network(x=gaussians.get_xyz.detach(), t=time_input)
            d_scaling = torch.mean(d_values['d_scaling'], dim=1, keepdim=True).expand_as(d_values['d_scaling'])
            output = dyna.rendering2(gaussians, grid, d_values['d_xyz'], 0, d_scaling, d_values['d_density'])
            loss = loss + dyna.l1_loss(output[0, ..., 0], targets[index])
        loss = loss / total_frame
        loss.backward(retain_graph=True)

        with torch.no_grad():
            # 原实现的阶段切换分支（默认 deform_downsamp_strategy='samp_hyper' 时不重建）
            if iteration == opt.iterations_node_sampling and opt.deform_downsamp_strategy == 'direct':
                original = deform.deform.as_gaussians
                deform.deform.init(init_pcl=original.get_xyz, force_init=True, opt=opt,
                                   as_gs_force_with_motion_mask=False, force_gs_keep_all=False)
                gaussians = deform.deform.as_gaussians
                gaussians._scaling = torch.nn.Parameter(original._scaling)
                gaussians._density = torch.nn.Parameter(original._density)
                gaussians._rotation = torch.nn.Parameter(original._rotation)
                deform.train_setting(opt)
            if (iteration == opt.iterations_node_rendering - 1
                    and iteration > opt.iterations_node_sampling):
                deform.deform.nodes.data[..., :3] = deform.deform.as_gaussians._xyz
            if iteration not in (opt.iterations_node_sampling, opt.iterations_node_rendering - 1):
                gaussians = deform.deform.as_gaussians
                gaussians.optimizer.step()
                gaussians.update_learning_rate(iteration)
                gaussians.optimizer.zero_grad(set_to_none=True)
                deform.update_learning_rate(iteration)
                deform.optimizer.step()
                deform.optimizer.zero_grad()
            deform.update(max(0, iteration + 1 - opt.node_warm_up))

        if iteration % 50 == 0:
            with torch.no_grad():
                val_loss = sum(dyna.l1_loss(render_frame(i, total_frame), targets[i]).item()
                               for i in range(total_frame)) / total_frame
            if val_loss < best_loss:
                best_loss, best_state = val_loss, snapshot()
            if iteration % 1000 == 0:
                print(f'  stage2 iter {iteration} train L1 {loss.item():.6f} val L1 {val_loss:.6f}')

    deform.deform.load_state_dict(best_state)
    with torch.no_grad():
        sequence = torch.stack([render_frame(i, total_frame).permute(2, 0, 1)
                                for i in range(total_frame)]).cpu().numpy()
    return dict(query_index=np.arange(depth) * cfg.interp_step_size, volume=sequence[0],
                frame=frames[0], sequence=sequence, frames=frames, iterations=total)
