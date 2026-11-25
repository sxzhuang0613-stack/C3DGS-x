"""对比实验统一测试入口：读取各方法保存的预测，在计分层上计算指标并汇总，不训练、不加载参考代码。

预测在训练结束时由同一模型按各方法自身前向模型渲染（见 methods/__init__.py），
因此本入口与具体方法无关，可在一个进程内比较全部方法。
用法（在 C3DGS 根目录）：
    python comparative/evaluation.py --config comparative/config.yaml --method all
    python comparative/evaluation.py --method dyna3dgr --stage full
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from case_data import load_config, iter_cases
from methods import METHODS, method_tag
from metrics import slice_metrics, ssim_options, summarize
from training import PREDICTION_FILE


def write_csv(path, rows):
    """写出字典列表；空列表不建文件。"""
    if not rows:
        return
    with open(path, 'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def slice_role(k, case):
    """层在当前协议下的角色：observed / train / held-out（计分）/ extrapolated（训练层范围外，不计分）。"""
    if case.stride == 1:
        return 'observed'
    if k in case.train_index:
        return 'train'
    return 'held-out' if k in case.test_index else 'extrapolated'


def save_reconstruction_figure(path, gt, pred, dense, query, metrics, case, title, step):
    """最终可视化图，版式同 C3DGS reconstructions/final.png。

    每个整数层一行："原图 | 重建（标 PSNR/SSIM）| k+step … k+1-step 插层重建（蓝色，无原图）"；
    未参与训练的层红色标注。gt/pred [S,H,W]，dense [Q,H,W] 为 query 层位的预测；显示固定 [0,1]、最近邻。
    """
    num_slices = gt.shape[0]
    per_interval = int(round(1 / step))
    position = {int(round(float(q) / step)): i for i, q in enumerate(query)}
    cols = 1 + per_interval
    fig, axes = plt.subplots(num_slices, cols, figsize=(1.7 * cols, 1.95 * num_slices), squeeze=False)
    for ax in axes.flat:
        ax.axis('off')
    show = dict(cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    for k in range(num_slices):
        role = slice_role(k, case)
        label, color = ((f' ({role})', 'tab:red') if role in ('held-out', 'extrapolated')
                        else ('', 'black'))
        axes[k, 0].imshow(gt[k], **show)
        axes[k, 0].set_title(f'GT {k}{label}', fontsize=8, color=color)
        axes[k, 1].imshow(np.clip(pred[k], 0, 1), **show)
        axes[k, 1].set_title(f"recon {k}\nPSNR {metrics[k]['psnr']:.1f} SSIM {metrics[k]['ssim']:.3f}",
                             fontsize=7, color=color)
        for j in range(1, per_interval if k < num_slices - 1 else 1):
            image = dense[position[k * per_interval + j]].astype(np.float32)
            axes[k, 1 + j].imshow(np.clip(image, 0, 1), **show)
            axes[k, 1 + j].set_title(f'recon {k + j * step:.2f}', fontsize=8, color='tab:blue')

    def mean_text(indices):
        return (f"PSNR {np.mean([metrics[k]['psnr'] for k in indices]):.2f} dB   "
                f"SSIM {np.mean([metrics[k]['ssim'] for k in indices]):.4f}")
    if case.stride == 1:
        summary = f'per-slice mean {mean_text(range(num_slices))}'
    else:
        summary = f'train {mean_text(case.train_index)}   held-out {mean_text(case.test_index)}'
    fig.suptitle(f'{title}   {summary}   (blue = interpolated)', fontsize=10)
    # 为总标题预留约 0.4 英寸，避免与首行子图标题重叠
    fig.tight_layout(rect=(0, 0, 1, 1 - .4 / fig.get_figheight()))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def linear_baseline(case, frame, device):
    """层间线性插值参照：留出层 k 取相邻训练层按距离加权，返回 {k: [H,W]}。"""
    images = case.frame(frame).to(device)
    prediction = {}
    for k in case.test_index:
        upper = int(np.searchsorted(case.train_index, k))
        below, above = int(case.train_index[upper - 1]), int(case.train_index[upper])
        weight = (k - below) / (above - below)
        prediction[int(k)] = images[below] * (1 - weight) + images[above] * weight
    return prediction


DISPLAY_NAMES = {'c3dgs': 'C3DGS (ours)', 'gaussiansvr': 'GaussianSVR', 'gsvr': 'GSVR',
                 'nesvor': 'NeSVoR', 'eclare': 'ECLARE', 'dyna3dgr_stage1': 'Dyna3DGR (stage1)',
                 'dyna3dgr_full': 'Dyna3DGR (full)', 'linear_interp': 'Linear interpolation'}
PROTOCOL_TITLES = {'observed': '全帧（observed：全部层训练，全部层计分）',
                   'holdout': '插帧（holdout：偶数层训练，夹在训练层之间的留出层计分）'}


def elapsed_stats(method_dir):
    """各病例 run.json 的用时均值与标准差（秒）；无记录时返回 None。"""
    values = [json.loads(path.read_text(encoding='utf-8'))['elapsed_seconds']
              for path in method_dir.glob('*/run.json')]
    return (float(np.mean(values)), float(np.std(values))) if values else None


def write_report(cfg):
    """把两种协议的 summary.csv 汇总为 <output_root>/results.md（每协议一张表，均值 ± 病例间标准差）。"""
    order = list(DISPLAY_NAMES)
    lines = ['# 对比实验结果', '',
             f'数据：{cfg.dataset_name.upper()} {cfg.dataset_mode} 集，参考相位 {cfg.reference_phase}（ED）。'
             '指标逐层计算，先病例内平均，再给出病例间均值 ± 标准差；PSNR、SSIM 越高越好，L1 越低越好。', '']
    for protocol in ('observed', 'holdout'):
        path = Path(cfg.output_root) / protocol / 'summary.csv'
        if not path.is_file():
            continue
        with open(path, newline='', encoding='utf-8') as stream:
            rows = [row for row in csv.DictReader(stream) if row['scope'] != 'all_frames']
        rows.sort(key=lambda row: order.index(row['method']) if row['method'] in order else len(order))
        lines += [f'## {PROTOCOL_TITLES[protocol]}', '',
                  '| 方法 | PSNR (dB) ↑ | SSIM ↑ | L1 ↓ | 时间/例 (s) | 病例数 | 计分层数 |',
                  '|---|---|---|---|---|---|---|']
        for row in rows:
            timing = elapsed_stats(Path(cfg.output_root) / protocol / row['method'])
            timing = f'{timing[0]:.1f} ± {timing[1]:.1f}' if timing else '–'
            lines.append(
                f"| {DISPLAY_NAMES.get(row['method'], row['method'])} "
                f"| {float(row['psnr_mean']):.2f} ± {float(row['psnr_std']):.2f} "
                f"| {float(row['ssim_mean']):.4f} ± {float(row['ssim_std']):.4f} "
                f"| {float(row['l1_mean']):.4f} ± {float(row['l1_std']):.4f} "
                f"| {timing} | {row['n_cases']} | {row['n_slices']} |")
        lines.append('')
    lines += ['注：',
              '- 层面轮廓 FWHM 取 Dataloader 给出的逐例层距 sz（数据不含单独层厚）；插帧协议下层厚仍为 sz，训练层间距为 '
              f'{cfg.holdout_stride}·sz，训练层范围外的外推层不计分。',
              '- 各方法采用原文/原代码预设的迭代数；Dyna3DGR 只运行其静态阶段（stage1，5000 步）。运动校正全部关闭，不抠图。',
              '- 时间为每例训练与输出的墙钟时间，各方法在同一张 A100 上串行独占运行。',
              '- Linear interpolation 为相邻训练层按距离线性插值，不训练，只作插帧参照。', '']
    report = Path(cfg.output_root) / 'results.md'
    report.write_text('\n'.join(lines), encoding='utf-8')
    print(f'汇总表已写入 {report}')


def main():
    # 读取配置与待评估的方法目录
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(Path(__file__).with_name('config.yaml')))
    parser.add_argument('--method', default='all', help="方法名或 all（评估输出根目录下已有的全部方法）")
    parser.add_argument('--stage', choices=['stage1', 'full'], default=None, help='仅 dyna3dgr')
    parser.add_argument('--cases', nargs='*', default=None)
    parser.add_argument('--protocol', choices=['observed', 'holdout'], default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.cases is not None:
        cfg.cases = args.cases
    if args.protocol is not None:
        cfg.eval_protocol = args.protocol
    root = Path(cfg.output_root) / cfg.eval_protocol
    if args.method == 'all':
        tags = sorted(p.name for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
    else:
        if args.method not in METHODS:
            raise ValueError(f'未知方法 {args.method}')
        stage = (args.stage or cfg.methods['dyna3dgr']['stage']) if args.method == 'dyna3dgr' else None
        tags = [method_tag(args.method, stage)]
    if not tags:
        raise ValueError(f'{root} 下没有可评估的方法输出')

    # 逐病例读取 GT，逐方法计算计分层指标
    device = torch.device(cfg.device)
    similarity = ssim_options(cfg)
    rows = {tag: [] for tag in tags}
    missing = {tag: [] for tag in tags}
    baseline = []
    for case in iter_cases(cfg):
        for tag in tags:
            path = root / tag / case.name / PREDICTION_FILE
            if not path.is_file():
                missing[tag].append(case.name)
                continue
            prediction = np.load(path)
            if not (np.array_equal(prediction['train_index'], case.train_index)
                    and np.array_equal(prediction['test_index'], case.test_index)):
                raise ValueError(f'{path}: 训练/计分层划分与当前协议不一致')
            if 'sequence' in prediction.files:
                frames, predictions = prediction['frames'].tolist(), prediction['sequence']
            else:
                frames, predictions = [int(prediction['frame'])], prediction['slices'][None]

            case_dir = root / tag / case.name
            for frame, pred_frame in zip(frames, predictions):
                gt_frame = case.frame(int(frame)).to(device)
                pred_frame = torch.from_numpy(pred_frame).to(device)
                if cfg.clamp_predictions:
                    pred_frame = pred_frame.clamp(0, cfg.data_range)
                # 全部整数层都算指标用于图示；只有计分层进入 metrics.csv
                metrics = [slice_metrics(pred_frame[k], gt_frame[k], similarity, cfg.data_range)
                           for k in range(case.shape[0])]
                for k in case.test_index:
                    rows[tag].append(dict(method=tag, case=case.name, frame=int(frame), slice=int(k),
                                          **metrics[k]))
                # 每例一张最终图（多帧方法只画第一帧，与静态方法对应）
                if cfg.save_images and frame == frames[0]:
                    save_reconstruction_figure(
                        case_dir / 'reconstruction.png', gt_frame.cpu().numpy(), pred_frame.cpu().numpy(),
                        prediction['dense'], prediction['query_index'], metrics, case,
                        f'{DISPLAY_NAMES.get(tag, tag)}   {case.name}', cfg.interp_step_size)

        if cfg.eval_protocol == 'holdout':
            gt_frame = case.frame(case.reference_phase).to(device)
            for k, image in linear_baseline(case, case.reference_phase, device).items():
                baseline.append(dict(method='linear_interp', case=case.name, frame=case.reference_phase,
                                     slice=k, **slice_metrics(image, gt_frame[k], similarity,
                                                              cfg.data_range)))

    # 保存逐层指标与汇总（Dyna3DGR full 另给初始帧单独一行，便于与静态方法横向比较）
    summary = []
    for tag in tags:
        write_csv(root / tag / 'metrics.csv', rows[tag])
        if missing[tag]:
            print(f'[{tag}] 缺少预测的病例 {len(missing[tag])} 个：{missing[tag]}')
        if not rows[tag]:
            continue
        frames = sorted({r['frame'] for r in rows[tag]})
        scopes = [('all_frames' if len(frames) > 1 else f'frame{frames[0]}', rows[tag])]
        if len(frames) > 1:
            scopes.append((f'frame{frames[0]}', [r for r in rows[tag] if r['frame'] == frames[0]]))
        for scope, scope_rows in scopes:
            summary.append(dict(method=tag, protocol=cfg.eval_protocol, scope=scope,
                                **summarize(scope_rows)))
    if baseline:
        summary.append(dict(method='linear_interp', protocol=cfg.eval_protocol,
                            scope=f'frame{cfg.reference_phase}', **summarize(baseline)))
    write_csv(root / 'summary.csv', summary)
    for item in summary:
        print(f"{item['method']:<18} {item['scope']:<10} cases={item['n_cases']:<3} "
              f"PSNR {item['psnr_mean']:.2f}±{item['psnr_std']:.2f}  "
              f"SSIM {item['ssim_mean']:.4f}±{item['ssim_std']:.4f}  "
              f"L1 {item['l1_mean']:.4f}±{item['l1_std']:.4f}")
    write_report(cfg)


if __name__ == '__main__':
    main()
