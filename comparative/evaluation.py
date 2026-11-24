"""对比实验统一测试入口：读取各方法保存的预测，在计分层上计算指标并汇总，不训练、不加载参考代码。

预测在训练结束时由同一模型按各方法自身前向模型渲染（见 methods/__init__.py），
因此本入口与具体方法无关，可在一个进程内比较全部方法。
用法（在 C3DGS 根目录）：
    python comparative/evaluation.py --config comparative/config.yaml --method all
    python comparative/evaluation.py --method dyna3dgr --stage full
"""
import argparse
import csv
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


def save_comparison(path, gt, pred, slice_index, data_range):
    """计分层的 GT | 预测 | 绝对误差 拼图（固定显示范围）。"""
    fig, axes = plt.subplots(len(slice_index), 3, figsize=(6, 2 * len(slice_index)), squeeze=False)
    for row, k in enumerate(slice_index):
        for col, (image, title) in enumerate(((gt[k], 'GT'), (pred[k], 'pred'),
                                              (np.abs(pred[k] - gt[k]), '|err|'))):
            axes[row, col].imshow(image, cmap='gray', vmin=0, vmax=data_range)
            axes[row, col].set_axis_off()
            axes[row, col].set_title(f'{title} k={k}', fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


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

            for frame, pred_frame in zip(frames, predictions):
                gt_frame = case.frame(int(frame)).to(device)
                pred_frame = torch.from_numpy(pred_frame).to(device)
                if cfg.clamp_predictions:
                    pred_frame = pred_frame.clamp(0, cfg.data_range)
                for k in case.test_index:
                    rows[tag].append(dict(method=tag, case=case.name, frame=int(frame), slice=int(k),
                                          **slice_metrics(pred_frame[k], gt_frame[k], similarity,
                                                          cfg.data_range)))
            if cfg.save_images:
                save_comparison(root / tag / case.name / 'comparison.png',
                                case.frame(int(frames[0])).numpy(), predictions[0],
                                case.test_index, cfg.data_range)

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
    write_csv(root / 'summary.csv', summary)
    for item in summary:
        print(f"{item['method']:<18} {item['scope']:<10} cases={item['n_cases']:<3} "
              f"PSNR {item['psnr_mean']:.2f}±{item['psnr_std']:.2f}  "
              f"SSIM {item['ssim_mean']:.4f}±{item['ssim_std']:.4f}  "
              f"L1 {item['l1_mean']:.4f}±{item['l1_std']:.4f}")


if __name__ == '__main__':
    main()
