"""GSVR（MIDL 2026）适配：复用原 train.train() 训练循环，只替换数据入口与最终输出。

与原实现的差异：
- load_data 改为由 CaseData 构造同结构张量（世界坐标 mm、逐层 PSF 协方差、包围盒），不读 NIfTI、不做 ANTs 预处理；
  全部像素参与监督（不抠图，等价于全 1 掩膜）。
- motion_correction=False（统一关闭）；apply_mask=False（原掩膜为胎儿脑设计）。
- max_epochs = cfg.iters（全批量，1 epoch = 1 次更新）；mini_batch_size 取对比配置（原值大于单例体素数会触发断言）。
- 原训练末尾的 visualize_gaussians（体数据导出）被替换为截获模型，在名义层位用层面 PSF 采样输出。
"""
from pathlib import Path

import numpy as np
import torch
import yaml

from case_data import use_reference, query_index

REPO_SUBDIR = 'GSVR'


def _stack_tensors(case, generate_cov_psf):
    """按原 load_data 的返回结构构造训练数据（单 stack，体素索引 (x=u, y=v, z=训练层序号)）。"""
    images = case.frame(case.reference_phase)[case.train_index].numpy()         # [N,H,W]
    img_data = np.ascontiguousarray(images.transpose(2, 1, 0)).astype(np.float64)  # [W,H,N]
    sx, sy, sz = case.spacing
    nx, ny, nz = img_data.shape
    # 体素 → 世界：x=(u+.5-W/2)sx，y=(v+.5-H/2)sy，z=(k-(S-1)/2)sz，其中 k=k0+stride·j
    affine = np.diag([sx, sy, case.stride * sz, 1.0])
    affine[:3, 3] = [(0.5 - nx / 2) * sx, (0.5 - ny / 2) * sy, float(case.z_mm(case.train_index[0]))]
    spacing = (sx, sy, case.stride * sz)

    coords_vxl = np.argwhere(np.ones_like(img_data, dtype=bool))
    coords_wrld = (affine @ np.hstack([coords_vxl, np.ones((len(coords_vxl), 1))]).T).T[:, :3]
    slice_centers_voxel = np.array([[nx / 2.0, ny / 2.0, k, 1] for k in range(nz)])
    slice_centers_world = (affine @ slice_centers_voxel.T).T[:, :3]
    slice_idcs = coords_vxl[:, 2].astype(int)
    cov_psf = generate_cov_psf(affine, spacing, case.thickness)
    values = img_data[coords_vxl[:, 0], coords_vxl[:, 1], coords_vxl[:, 2]]

    # 包围盒与强度归一化同原 load_data
    bbox_min_raw, bbox_max_raw = coords_wrld.min(0), coords_wrld.max(0)
    padding = (bbox_max_raw - bbox_min_raw) * 0.05
    bbox_world = [bbox_min_raw - padding, bbox_max_raw + padding]
    bbox_world_size = bbox_world[1] - bbox_world[0]
    min_value, max_value = values.min(), values.max()
    values_nrmd = (values - min_value) / (max_value - min_value)
    stack_nrmd = (img_data.clip(0.0, None) - min_value) / (max_value - min_value)
    return dict(stack_nrmd=stack_nrmd, affine=affine, spacing=spacing, slice_idcs=slice_idcs,
                slice_centers=slice_centers_world[slice_idcs], coords=coords_wrld,
                values=values_nrmd, cov_psf=np.broadcast_to(cov_psf, (nz, 3, 3)), cov_single=cov_psf,
                bbox_world=bbox_world, bbox_world_size=bbox_world_size, n_slices=nz,
                min_value=min_value, max_value=max_value)


def fit(case, cfg, method_cfg, out_dir, **_):
    """逐病例运行 GSVR 原训练循环，返回名义层位的 PSF 采样预测。"""
    use_reference(Path(cfg.references_root) / REPO_SUBDIR)
    import train as gsvr_train
    from utils import generate_cov_psf, topK_neighbors

    with open(Path(cfg.references_root) / method_cfg['preset'], encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    config['gsvr'].update(max_epochs=cfg.iters, motion_correction=False,
                          mini_batch_size=method_cfg['mini_batch_size'])
    # reconstruction.spacing 只决定原代码预分配的可视化网格，此处取面内分辨率以免占用过多显存
    config['data']['reconstruction'].update(apply_mask=False,
                                            spacing=[case.spacing[0], case.spacing[1], case.spacing[0]])
    data = _stack_tensors(case, generate_cov_psf)
    shape_reconstruction = [round(b / s) + 1 for b, s in
                            zip(data['bbox_world_size'], config['data']['reconstruction']['spacing'])]

    # 替换数据入口与末尾导出；训练循环本身不变
    gsvr_train.load_data = lambda cfg_data: [
        [data['stack_nrmd']], data['affine'][None], data['slice_idcs'], data['slice_centers'],
        data['coords'], data['values'], data['cov_psf'], data['bbox_world'],
        data['bbox_world_size'], shape_reconstruction, data['n_slices'],
        data['min_value'], data['max_value']]
    captured = {}

    def capture(gsvr, gpu_topk, K, *args, **kwargs):
        """截获训练结束时的模型；torch.compile 包装时取原模块，避免按新输入形状重新编译。"""
        captured.update(model=getattr(gsvr, '_orig_mod', gsvr), gpu_topk=gpu_topk, K=K)

    gsvr_train.visualize_gaussians = capture
    gsvr_train.train(cfg_data=config['data'], cfg_gsvr=config['gsvr'], output_root=str(out_dir),
                     cfg_vis=None)

    model, gpu_topk, K = captured['model'], captured['gpu_topk'], captured['K']
    torch.save(model.state_dict(), out_dir / 'gsvr_model.pt')

    # 名义层位：像素中心世界坐标 + 层面 PSF 协方差（与训练前向相同），再按原方式反归一化
    num_slices, height, width = case.shape
    sx, sy, _ = case.spacing
    device = model.gaussian_primitives.mu.device
    v, u = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device),
                          indexing='ij')
    xy = torch.stack([(u + 0.5 - width / 2) * sx, (v + 0.5 - height / 2) * sy], -1).reshape(-1, 2)
    cov = torch.as_tensor(data['cov_single'], dtype=torch.float32, device=device)
    cov = cov.expand(xy.shape[0], 3, 3)
    query = query_index(case, cfg.interp_step_size)
    slices = []
    with torch.no_grad():
        num_active = model.gaussian_primitives.num_active.item()
        for z in case.z_mm(query):
            coords = torch.cat([xy, torch.full_like(xy[:, :1], float(z))], -1).float().contiguous()
            top_k = topK_neighbors(gpu_topk, model.gaussian_primitives.mu.data[:num_active], K, coords)
            values = model(coords, top_k, covs_psf=cov)
            slices.append(values.view(height, width))
    volume = torch.stack(slices).cpu().numpy()
    volume = (volume * (data['max_value'] - data['min_value']) + data['min_value']).clip(0.0, None)
    return dict(query_index=query, volume=volume, frame=case.reference_phase)
