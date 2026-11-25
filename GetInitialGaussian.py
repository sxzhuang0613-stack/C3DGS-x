import csv
import os
import torch
from Gaussians.GaussianModel_structural import GaussianModel
from Gaussians.utils import covariance_from_scaling_rotation
from Dataloader import dataset
from utils.CostFunctions import l1_loss, ssim, anisotropic_scale_loss, envelope_loss
from utils.Renderer import render
import argparse
import random
import time
import numpy as np
from utils.config_utils import load_config, setup_runtime, dataset_options, ssim_options, RENDER_SEMANTICS
from utils.case_geometry import make_geometry, check_geometry, attach_geometry, save_model_ply, read_model_geometry
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



# ==================== [功能模块 1: 动态自适应子损失函数趋势图] ====================
def update_loss_plot(out_dir, iters_list, loss_dict, dpi=150):
    plt.figure(figsize=(10, 6))
    
    # 预设一条色板，自动分配给不同的子损失函数
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f1c40f', '#9b59b6', '#e67e22', '#1abc9c']
    
    # 优先将 Total Loss 画在最上层，粗体红色
    if "Total Loss" in loss_dict:
        plt.plot(iters_list, loss_dict["Total Loss"], label='Total Loss', color=colors[0], linewidth=2.0)
        
    color_idx = 1  
    for loss_name, loss_values in loss_dict.items():
        if loss_name == "Total Loss":
            continue
        # 给其他子损失自动分配颜色，使用虚线
        color = colors[color_idx % len(colors)]
        plt.plot(iters_list, loss_values, label=loss_name, color=color, linestyle='--', alpha=0.8, linewidth=1.5)
        color_idx += 1
        
    plt.title("Training Loss Trajectory (Dynamic Sub-Losses)", fontsize=14, fontweight='bold')
    plt.xlabel("Iteration Step", fontsize=12)
    plt.ylabel("Loss Value", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    
    # 静默覆盖保存
    plot_path = os.path.join(out_dir, "loss_curves.png")
    plt.savefig(plot_path, dpi=dpi)
    plt.close('all')

# ==================== [功能模块 2: 周期性全序列快照重建 (含插帧联动)] ====================
def reconstruct_full_case(Gaussians, imgs, out_dir, iter_num, include_interp, step_size=0.25,
                          dpi=150, similarity=None, roles=None, **render_kwargs):
    """渲染整个病例并与原图并排拼成一张 PNG：reconstructions/final.png（只保存训练结束时的结果）。

    imgs 为参考相位原图 [S,H,W]。每个真实层显示"原图 | 渲染"并标注该层 PSNR/SSIM；
    进入插层训练后，同一行再接 k+step, k+2*step, ... 的插值渲染（无原图，不计指标）。
    roles 为 {层号: 'held-out' | 'extrapolated'}（holdout 协议下未参与训练的层，红色标注）；
    标题分别给出训练层与留出层（不含外推层）的平均指标。固定 [0,1] 灰度、最近邻显示。
    """
    roles = roles or {}
    rec_dir = os.path.join(out_dir, "reconstructions")
    os.makedirs(rec_dir, exist_ok=True)
    s, h, w = imgs.shape
    per_interval = max(1, round(1 / step_size)) if include_interp else 1

    with torch.no_grad():
        covariance = covariance_from_scaling_rotation(Gaussians.get_scaling, Gaussians._rotation)
        def view(index):
            return render(Gaussians, img_size=[h, w], slice_idx=index, axis=2, num_slices=s,
                          covariance=covariance, **render_kwargs)['render']
        rows = []
        for k in range(s):
            prediction = view(k)
            mse = (prediction - imgs[k]).square().mean().clamp_min(1e-12)
            psnr_value = float(10 * torch.log10(1 / mse))
            ssim_value = float(ssim(prediction[None, None], imgs[k][None, None], **(similarity or {})))
            # 插值层：k+step ... k+1-step，最后一层之后没有区间。
            interps = [(k + j * step_size, view(min(k + j * step_size, s - 1)).clamp(0, 1).cpu().numpy())
                       for j in range(1, per_interval) if k < s - 1]
            rows.append((k, imgs[k].cpu().numpy(), prediction.clamp(0, 1).cpu().numpy(),
                         psnr_value, ssim_value, interps))

    # 插层期每行对应一个真实层：原图、渲染、插值；否则每行放两组"原图 | 渲染"。
    pairs_per_row = 1 if include_interp else 2
    cols = (2 + per_interval - 1) if include_interp else 2 * pairs_per_row
    nrows = int(np.ceil(len(rows) / pairs_per_row))
    fig, axes = plt.subplots(nrows, cols, figsize=(1.7 * cols, 1.95 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis('off')
    show = dict(cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    for n, (k, gt, prediction, psnr_value, ssim_value, interps) in enumerate(rows):
        r, c = divmod(n, pairs_per_row)
        c *= cols // pairs_per_row
        role, color = (f" ({roles[k]})", 'tab:red') if k in roles else ("", 'black')
        axes[r, c].imshow(gt, **show)
        axes[r, c].set_title(f"GT {k}{role}", fontsize=8, color=color)
        axes[r, c + 1].imshow(prediction, **show)
        axes[r, c + 1].set_title(f"render {k}\nPSNR {psnr_value:.1f} SSIM {ssim_value:.3f}",
                                 fontsize=7, color=color)
        for j, (index, image) in enumerate(interps):
            axes[r, c + 2 + j].imshow(image, **show)
            axes[r, c + 2 + j].set_title(f"render {index:.2f}", fontsize=8, color='tab:blue')
    def mean_text(selected):
        return (f"PSNR {np.mean([row[3] for row in selected]):.2f} dB   "
                f"SSIM {np.mean([row[4] for row in selected]):.4f}")
    if roles:
        train_rows = [row for row in rows if row[0] not in roles]
        held_rows = [row for row in rows if roles.get(row[0]) == 'held-out']
        summary = f"train {mean_text(train_rows)}" + (f"   held-out {mean_text(held_rows)}" if held_rows else "")
    else:
        summary = f"per-slice mean {mean_text(rows)}"
    fig.suptitle(f"iter {iter_num}   {summary}" + ("   (blue = interpolated)" if include_interp else ""),
                 fontsize=10)
    # 为总标题预留约 0.4 英寸，避免与首行子图标题重叠。
    fig.tight_layout(rect=(0, 0, 1, 1 - .4 / fig.get_figheight()))
    fig.savefig(os.path.join(rec_dir, "final.png"), dpi=dpi)
    plt.close(fig)
    torch.cuda.empty_cache()

# ==================== [功能模块 3: 结构操作日志] ====================
# 计数类字段在窗口内求和；门限/中位数取窗口内最后一次结构周期的值；points 为写入时的实际点数。
STRUCTURE_COUNT_FIELDS = ('split', 'split_size', 'split_normal', 'pruned', 'pruned_box',
                          'pruned_invisible', 'selected', 'residual_high', 'eligible')
STRUCTURE_LAST_FIELDS = ('grad_threshold', 'grad_median', 'normal_threshold', 'normal_median',
                         'weight_median', 'error_median', 'sigma_pixels_median')
STRUCTURE_FIELDS = ('iteration', 'points', 'structure_steps') + STRUCTURE_COUNT_FIELDS + STRUCTURE_LAST_FIELDS


def prepare_structure_log(path, start_iteration):
    """新训练覆盖日志；续训只保留不超过起点的行，避免中断后重复记录。"""
    rows = []
    if start_iteration > 0 and os.path.isfile(path):
        with open(path, newline='', encoding='utf-8') as stream:
            rows = [row for row in csv.DictReader(stream) if int(row['iteration']) <= start_iteration]
    with open(path, 'w', newline='', encoding='utf-8') as stream:
        # 旧版日志可能含已取消的列（如 clone），续训时忽略。
        writer = csv.DictWriter(stream, fieldnames=STRUCTURE_FIELDS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def append_structure_log(path, iteration, points, reports):
    """把一个日志窗口内的结构周期报告汇总为一行；窗口内无结构操作时计数为 0。

    两种结构算法的报告字段不完全相同，缺失的计数记 0、缺失的门限/中位数留空。
    """
    row = dict(iteration=iteration, points=points, structure_steps=len(reports))
    row.update({key: sum(report.get(key, 0) for report in reports) for key in STRUCTURE_COUNT_FIELDS})
    row.update({key: reports[-1].get(key, '') if reports else '' for key in STRUCTURE_LAST_FIELDS})
    with open(path, 'a', newline='', encoding='utf-8') as stream:
        csv.DictWriter(stream, fieldnames=STRUCTURE_FIELDS).writerow(row)


def save_training_checkpoint(path, gaussians, iteration, history_iters, history_losses, args,
                             elapsed_seconds=0.):
    """保存拓扑、Adam 动量、增殖统计、随机状态与累计训练用时；PLY 只用于交换模型。"""
    state = {
        'iteration': iteration, 'config': vars(args), 'elapsed_seconds': elapsed_seconds,
        'geometry': getattr(gaussians,'geometry',None),
        'geometry_reference_count': getattr(gaussians,'geometry_reference_count',len(gaussians.get_xyz)),
        'parameters': {name: getattr(gaussians, name).detach() for name in
                       ('_xyz', '_intensity', '_scaling', '_rotation')},
        'optimizer': gaussians.optimizer.state_dict(),
        'statistics': {name: getattr(gaussians, name) for name in gaussians.statistic_names},
        'structural_grad_reference': gaussians.structural_grad_reference,
        'structural_normal_reference': gaussians.structural_normal_reference,
        'history_iters': history_iters, 'history_losses': history_losses,
        'random': random.getstate(), 'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all(),
    }
    torch.save(state, path + '.tmp')
    os.replace(path + '.tmp', path)

def restore_training_checkpoint(state, gaussians, args):
    check_resume_config(state['config'], args)
    geometry = state.get('geometry')
    semantics = RENDER_SEMANTICS[args.render_model]
    if (geometry is None or geometry.get('render_model') != args.render_model
            or geometry.get('intensity_semantics') != semantics['intensity']
            or geometry.get('opacity_mode') != semantics['opacity']
            or geometry.get('rasterizer_api_version') != semantics['api']):
        raise ValueError('checkpoint 缺少匹配的渲染语义记录，不能恢复为当前模型')
    attach_geometry(gaussians,geometry)
    for name, tensor in state['parameters'].items():
        setattr(gaussians, name, torch.nn.Parameter(tensor.to(args.device)))
    gaussians.geometry_reference_count = state.get('geometry_reference_count',len(gaussians.get_xyz))
    gaussians.training_setup(args)
    gaussians.optimizer.load_state_dict(state['optimizer'])
    for name, tensor in state['statistics'].items():
        setattr(gaussians, name, tensor.to(args.device))
    gaussians.structural_grad_reference = state['structural_grad_reference']
    gaussians.structural_normal_reference = state.get('structural_normal_reference')
    random.setstate(state['random'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])

def check_resume_config(saved, args):
    # 旧渲染模型与固定 opacity 的 alpha 合成语义不同，不自动转换旧参数或 Adam 状态。
    if saved.get('render_model') != args.render_model:
        raise ValueError('渲染模型已改变，不能完整续训；请使用新的实验输出目录')
    if saved.get('structural_algorithm') != args.structural_algorithm:
        raise ValueError('结构算法已改变，不能完整续训；请使用新的实验输出目录')
    # 允许延长训练和改变输出频率；优化目标或 Adam 参数改变时应参数热启动。
    keys = ('position_lr_init','position_lr_final','position_lr_max_steps','position_lr_delay_mult',
            'intensity_lr','scaling_lr','rotation_lr','adam_eps','adam_betas',
            'log_scale_min','log_scale_max','min_sigma_pixels','real_slices_per_iter','interpolation',
            'interpolate_iter','interpolation_probability','interpolation_alpha_min',
            'interpolation_alpha_max','anisotropic_max_ratio','anisotropic_epsilon',
            'densify_from_iter','densify_until_iter','densification_interval',
            'lambda_l1','lambda_ssim','lambda_anisotropic','lambda_interp_l1','lambda_interp_ssim',
            'lambda_interp_envelope',
            'background','raster_debug','slice_thickness_fwhm_spacing','amplitude_cutoff',
            'slice_protocol','holdout_stride')
    keys = set(keys)|{key for key in vars(args) if key.startswith('structural_')}
    changed = sorted(key for key in keys if key in saved and saved[key] != getattr(args,key))
    if changed:
        raise ValueError(f'续训配置改变了优化条件 {changed}；请使用独立实验或 PLY 参数热启动')

def training(args):
    setup_runtime(args)
    loader = torch.utils.data.DataLoader(
        dataset(mode=args.dataset_mode, **dataset_options(args)),
        batch_size=1, num_workers=args.num_workers, pin_memory=args.pin_memory,
    )
    similarity = ssim_options(args)
    for sample_idx, data in enumerate(loader):
        sample_name = os.path.splitext(data['info']['filename'][0])[0]
        # 输出 <output_path>/<case>/<output_case_subdir>/；子目录为空时即旧布局 <output_path>/<case>/
        case_dir = os.path.join(args.output_path, sample_name, args.output_case_subdir)
        checkpoint_path = os.path.join(case_dir, 'training_state.pth')
        final_ply = os.path.join(case_dir, 'point_cloud_final.ply')
        # 训练完成后 checkpoint 被删除，是否已完成以最终 PLY 记录的步数判断
        final_record = (read_model_geometry(final_ply)
                        if args.stage1_skip_existing and os.path.isfile(final_ply) else None)
        skip_existing = final_record is not None and final_record.get('iteration', 0) >= args.iters
        state = None
        if args.stage1_resume and not skip_existing and os.path.isfile(checkpoint_path):
            # 只加载本地可信训练文件；完整随机状态包含 NumPy 对象。
            state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            check_resume_config(state['config'],args)
        sequence = data['cmr'][0]
        imgs = sequence[:, args.reference_phase].to(args.device)
        s, h, w = imgs.shape
        # holdout 只用 k % holdout_stride == 0 的层监督；几何仍按完整 s 层栈，
        # 留出层图像不进入初始化、损失和结构统计
        observed = (list(range(0, s, args.holdout_stride)) if args.slice_protocol == 'holdout'
                    else list(range(s)))
        roles = {k: 'held-out' if k < observed[-1] else 'extrapolated'
                 for k in range(s) if k not in observed}
        record = None
        if skip_existing:
            geometry = final_record['geometry']
        elif state is not None:
            geometry = state.get('geometry')
            if geometry is None:
                raise ValueError('旧 checkpoint 缺少渲染和几何记录，不能按 alpha 模型恢复')
        elif args.warm_start_ply_path:
            record = read_model_geometry(args.warm_start_ply_path)
            if record is None:
                raise ValueError('PLY 缺少局部强度及渲染语义记录，不能自动解释为 alpha 模型')
            geometry = record['geometry']
        else:
            geometry = make_geometry(data['info']['spacing_xyz'][0],imgs.shape,args)
        current = make_geometry(data['info']['spacing_xyz'][0],imgs.shape,args,legacy=geometry['version']==0)
        check_geometry(geometry,current)
        rendering = geometry['rendering']
        if skip_existing:
            tqdm.write(f'[{sample_name}] 已完成 {final_record["iteration"]} 步，跳过。')
            continue
        os.makedirs(case_dir, exist_ok=True)
        gaussians = GaussianModel()
        attach_geometry(gaussians,geometry)
        history_iters, history_losses = [], {}
        start_iteration, elapsed_before = 0, 0.
        if state is not None:
            tqdm.write(f'[{sample_name}] 续训已有参数，初始化不重新执行。')
            restore_training_checkpoint(state, gaussians, args)
            start_iteration = state['iteration']
            history_iters = state['history_iters']
            history_losses = state['history_losses']
            elapsed_before = state.get('elapsed_seconds', 0.)
        else:
            if args.warm_start_ply_path:
                gaussians.load_ply(args.warm_start_ply_path,
                                   legacy_color_policy=args.legacy_color_policy, device=args.device)
                if record is not None:
                    gaussians.geometry_reference_count = record['geometry_reference_count']
            else:
                gaussians.initialize_from_images(imgs[observed],geometry,args,rendering,sample_name,
                                                 observed,s)
            gaussians.training_setup(args)

        with torch.no_grad():
            gaussians.prune_points(gaussians._invalid_point_mask())
        if gaussians.get_xyz.shape[0] == 0:
            raise FloatingPointError(f'{sample_name}: 没有有效高斯点，无法开始优化')
        structure_log = os.path.join(case_dir, 'structure_log.csv')
        prepare_structure_log(structure_log, start_iteration)
        structure_reports = []
        # 加和模型用逐切片梯度统计 + 闭式判据；旧 alpha 模型保持原残差门控流程
        additive = args.structural_algorithm == 'additive_moment_v1'
        structural_step = gaussians.structural_step_additive if additive else gaussians.structural_step
        start_time = time.time()
        pbar = tqdm(range(start_iteration + 1, args.iters + 1),
                    desc=f'Case [{sample_idx + 1}/{len(loader)}] {sample_name}')
        for iteration in pbar:
            gaussians.update_learning_rate(iteration)
            # 同一优化步的真实切片与插层共享几何图；更新参数后下一步重新构造。
            covariance = covariance_from_scaling_rotation(gaussians.get_scaling, gaussians._rotation)
            # 无放回均匀采样，单层和双层病例也能获得真实监督。
            indices = random.sample(observed, min(len(observed), args.real_slices_per_iter))
            outputs = [render(gaussians, [h, w], index, 2, s, covariance=covariance, **rendering)
                       for index in indices]
            # 结构统计只用真实切片；插层平面没有观测，不参与增殖判据
            real_outputs = list(zip(outputs, indices))
            predictions = torch.stack([output['render'] for output in outputs])[:, None]
            targets = imgs[indices, None]
            loss_l1 = l1_loss(predictions, targets)
            loss_ssim = 1 - ssim(predictions, targets, **similarity)
            loss_aniso = anisotropic_scale_loss(
                gaussians.get_scaling, args.anisotropic_max_ratio, args.anisotropic_epsilon,
                reference_count=gaussians.geometry_reference_count)
            zero = imgs.new_zeros(())
            loss_interp_l1, loss_interp_ssim, loss_envelope = zero, zero, zero
            # 插层区间取相邻两观测层 [observed[j], observed[j+1]]；全层训练时即 [k, k+1]
            n_observed = len(observed)
            if args.interpolation and iteration >= args.interpolate_iter and n_observed > 1:
                if random.random() < args.interpolation_probability:
                    alpha = random.uniform(args.interpolation_alpha_min,
                                           args.interpolation_alpha_max)
                    left = random.randrange(max(1, n_observed - 2))
                    intervals = range(left, min(left + 2, n_observed - 1))
                    interp_pred, lower, upper = [], [], []
                    for j in intervals:
                        below, above = observed[j], observed[j + 1]
                        output = render(gaussians, [h, w], below + alpha * (above - below), 2, s,
                                        covariance=covariance, **rendering)
                        outputs.append(output)
                        interp_pred.append(output['render'])
                        lower.append(torch.minimum(imgs[below], imgs[above]))
                        upper.append(torch.maximum(imgs[below], imgs[above]))
                    # 无层间真值：区间约束防止越界伪影，不拟合像素级线性混合（避免双重轮廓）。
                    loss_envelope = envelope_loss(torch.stack(interp_pred), torch.stack(lower),
                                                  torch.stack(upper))
                else:
                    index = observed[random.randrange(n_observed)]
                    output = render(gaussians, [h, w], index, 2, s, covariance=covariance, **rendering)
                    outputs.append(output)
                    real_outputs.append((output, index))
                    interp_pred = output['render'][None, None]
                    interp_gt = imgs[index][None, None]
                    loss_interp_l1 = l1_loss(interp_pred, interp_gt)
                    loss_interp_ssim = 1 - ssim(interp_pred, interp_gt, **similarity)
            loss = (args.lambda_l1 * loss_l1 + args.lambda_ssim * loss_ssim
                    + args.lambda_anisotropic * loss_aniso
                    + args.lambda_interp_l1 * loss_interp_l1
                    + args.lambda_interp_ssim * loss_interp_ssim
                    + args.lambda_interp_envelope * loss_envelope)
            loss.backward()
            with torch.no_grad():
                if iteration < args.densify_until_iter and additive:
                    gaussians.add_slice_statistics(real_outputs)
                elif iteration < args.densify_until_iter:
                    # radii>0 只表示候选范围；强度梯度非零才说明该点在本步实际参与了像素合成。
                    candidate = torch.stack([output['visibility_filter'] for output in outputs]).any(dim=0)
                    visible = candidate & (gaussians._intensity.grad[:, 0] != 0)
                    radii = torch.stack([output['radii'] for output in outputs]).amax(dim=0)
                    gaussians.add_densification_stats_3d(visible, radii)
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                # 尺度下界按 min_sigma_pixels 个面内像素换算，log_scale_min 仅作数值安全边界。
                gaussians._scaling.clamp_(gaussians.log_scale_floor(), args.log_scale_max)
                gaussians.prune_points(gaussians._invalid_point_mask())
                if gaussians.get_xyz.shape[0] == 0:
                    raise FloatingPointError(f'{sample_name}: 优化后无有效高斯点')
                if args.densification_interval > 0 and iteration % args.densification_interval == 0:
                    if args.densify_from_iter <= iteration < args.densify_until_iter:
                        structure_reports.append(structural_step(imgs, iteration, rendering, similarity))
                    elif iteration >= args.densify_until_iter and args.structural_prune_after_densify:
                        # 增殖窗口后的审计：剪除框外点、拆分尺寸超限点，不做误差驱动增殖。
                        structure_reports.append(structural_step(
                            imgs, iteration, rendering, similarity, densify=False))
            current = {'Total Loss': loss.item(), 'L1 Loss': loss_l1.item(),
                       'SSIM Loss': loss_ssim.item(), 'Aniso': loss_aniso.item(),
                       'L1 interpo': loss_interp_l1.item(), 'ssim inter': loss_interp_ssim.item(),
                       'Envelope': loss_envelope.item()}
            history_iters.append(iteration)
            for name, value in current.items():
                history_losses.setdefault(name, []).append(value)
            if iteration % args.status_interval == 0:
                pbar.set_postfix(loss=current['Total Loss'], points=gaussians.get_xyz.shape[0])
            if iteration % args.loss_plot_interval == 0 or iteration == args.iters:
                update_loss_plot(case_dir, history_iters, history_losses, args.plot_dpi)
            if iteration % args.structure_log_interval == 0 or iteration == args.iters:
                append_structure_log(structure_log, iteration, gaussians.get_xyz.shape[0], structure_reports)
                structure_reports = []
            # checkpoint 只用于中断续训；正常结束后删除，只保留 final 权重
            if iteration % args.checkpoint_interval == 0 or iteration == args.iters:
                save_training_checkpoint(checkpoint_path, gaussians, iteration,
                                         history_iters, history_losses, args,
                                         elapsed_before + time.time() - start_time)
        elapsed = elapsed_before + time.time() - start_time
        if args.save_ply:
            training_record = dict(slice_protocol=args.slice_protocol, observed_slices=observed,
                                   num_slices=s, elapsed_seconds=elapsed)
            save_model_ply(gaussians,final_ply,max(start_iteration,args.iters),training_record)
        if args.visualize:
            reconstruct_full_case(
                gaussians, imgs, case_dir, max(start_iteration, args.iters),
                args.interpolation and args.iters >= args.interpolate_iter,
                args.interp_step_size, args.plot_dpi, similarity, roles, **rendering)
        if args.save_ply and os.path.isfile(checkpoint_path):
            os.remove(checkpoint_path)
        del gaussians, state
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config_train.yaml')
    training(load_config(parser.parse_args().config))
