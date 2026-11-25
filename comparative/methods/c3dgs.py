"""C3DGS（本方法）接入：读取 GetInitialGaussian.py 按同一协议训练好的最终 PLY，在名义层位渲染。

训练由 C3DGS 自身入口完成（其配置的 slice_protocol / holdout_stride / output_case_subdir 指定协议与
输出子目录 <model_root>/<case>/<protocol>/），此处不训练；用时取 PLY 记录中的 elapsed_seconds。
渲染用 PLY 几何记录中的层面轮廓与坐标约定，连续层索引 k 即完整栈第 k 层位置（与训练插层一致）。
"""
from pathlib import Path

import numpy as np
import torch

from case_data import query_index


def fit(case, cfg, method_cfg, out_dir, **_):
    """加载 <model_root>/<case>/<protocol>/point_cloud_final.ply，返回名义层位渲染。"""
    from Gaussians.GaussianModel_structural import GaussianModel
    from utils.Renderer import render
    from utils.case_geometry import attach_geometry, read_model_geometry

    ply = Path(method_cfg['model_root']) / case.name / cfg.eval_protocol / 'point_cloud_final.ply'
    record = read_model_geometry(ply)
    if record is None or 'training' not in record:
        raise FileNotFoundError(f'{ply}: 缺少带训练记录的最终 PLY，请先用 C3DGS 按该协议训练')
    training, geometry = record['training'], record['geometry']
    # 训练层与几何必须与对比协议一致，否则留出层可能已参与训练
    if (training['slice_protocol'] != cfg.eval_protocol
            or training['observed_slices'] != case.train_index.tolist()
            or not np.allclose(geometry['spacing_xyz'], case.spacing)):
        raise ValueError(f'{ply}: 训练协议、训练层或 spacing 与对比设置不一致')

    model = GaussianModel()
    attach_geometry(model, geometry)
    model.load_ply(str(ply), legacy_color_policy=method_cfg['legacy_color_policy'], device=cfg.device)
    num_slices, height, width = case.shape
    query = query_index(case, cfg.interp_step_size)
    with torch.no_grad():
        volume = torch.stack([render(model, [height, width], float(k), 2, num_slices,
                                     **geometry['rendering'])['render'] for k in query])
    return dict(query_index=query, volume=volume.cpu().numpy(), frame=case.reference_phase,
                iterations=record['iteration'], elapsed_seconds=training['elapsed_seconds'])
