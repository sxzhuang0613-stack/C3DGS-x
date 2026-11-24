"""对比评估指标：逐层 L1 / PSNR / SSIM，定义与 C3DGS evaluate.py 一致。"""
import numpy as np
import torch

import case_data  # noqa: F401  确保 C3DGS 根目录在 sys.path 中
from utils.CostFunctions import l1_loss, ssim


def ssim_options(cfg):
    """SSIM 参数（窗口、sigma、k1/k2、data_range）取自对比配置。"""
    return dict(window_size=cfg.ssim_window_size, sigma=cfg.ssim_sigma,
                k1=cfg.ssim_k1, k2=cfg.ssim_k2, data_range=cfg.data_range)


def slice_metrics(pred, gt, similarity, data_range):
    """单层指标；pred/gt 为同设备 [H,W] 张量。"""
    mse = ((pred - gt) ** 2).mean()
    psnr = float('inf') if mse.item() == 0 else float(10 * torch.log10(data_range ** 2 / mse))
    return dict(l1=l1_loss(pred, gt).item(), psnr=psnr,
                ssim=ssim(pred[None, None], gt[None, None], **similarity).item())


def summarize(rows, keys=('l1', 'psnr', 'ssim')):
    """先在病例内对层取均值，再给出病例间均值与标准差（避免层数多的病例占更大权重）。"""
    per_case = {}
    for row in rows:
        per_case.setdefault(row['case'], []).append(row)
    summary = dict(n_cases=len(per_case), n_slices=len(rows))
    for key in keys:
        case_means = np.array([np.mean([r[key] for r in case_rows]) for case_rows in per_case.values()])
        summary[f'{key}_mean'] = float(case_means.mean())
        summary[f'{key}_std'] = float(case_means.std())
    return summary
