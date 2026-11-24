"""NeSVoR（IEEE TMI 2023）适配：直接调用 nesvor.inr.train.train。

与原 CLI 流程的差异：
- 每个训练层构造成一个 Slice（名义刚体变换 (0,0,z_k)、resolution_z = 层厚 FWHM），掩膜全为 True（不抠图）；
  不做 SVoRT 配准、分割与偏场校正。
- no_transformation_optimization=True（运动校正统一关闭）；n_iter = cfg.iters，其余取原 parser 默认值。
- 采样输出用原 sample_slice（层面 PSF），但掩膜换成覆盖整个栈的全 1 体：原训练点占据掩膜在层间可能出现空洞。
"""
import torch

from case_data import query_index


def _slice(image, z_mm, case, device):
    """构造单层 Slice：image [H,W]，层局部坐标与 C3DGS 像素中心约定一致。"""
    from nesvor.image import Slice
    from nesvor.transform import RigidTransform
    axisangle = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, float(z_mm)]], device=device)
    return Slice(image[None].to(device), torch.ones((1,) + image.shape, dtype=torch.bool, device=device),
                 RigidTransform(axisangle), case.spacing[0], case.spacing[1], case.thickness)


def _full_mask(case, device):
    """覆盖首末层外一个层厚的全 1 掩膜体，体素为面内 spacing。"""
    from nesvor.image import Volume
    from nesvor.transform import RigidTransform
    num_slices, height, width = case.shape
    resolution = case.spacing[0]
    depth = int(round(((num_slices - 1) * case.spacing[2] + 2 * case.thickness) / resolution)) + 1
    ones = torch.ones((depth, height, width), device=device)
    return Volume(ones, ones.bool(), RigidTransform(torch.zeros(1, 6, device=device)),
                  case.spacing[0], case.spacing[1], resolution)


def fit(case, cfg, method_cfg, out_dir, **_):
    """逐病例训练 NeSVoR，返回名义层位经层面 PSF 采样的预测。"""
    from nesvor.cli.parsers import build_parser_training
    from nesvor.inr.train import train
    from nesvor.inr.sample import sample_slice

    device = torch.device('cuda', torch.cuda.current_device())
    args = build_parser_training().parse_args([])
    for key, value in method_cfg.get('overrides', {}).items():
        setattr(args, key, value)
    args.n_iter, args.n_epochs = cfg.iters, None
    args.no_transformation_optimization = True
    args.device, args.dtype, args.debug = device, torch.float32, False

    images = case.frame(case.reference_phase)
    slices = [_slice(images[k], z, case, device)
              for k, z in zip(case.train_index, case.z_mm(case.train_index))]
    model, _, _ = train(slices, args)
    torch.save(dict(state_dict=model.state_dict(), args=vars(args)), out_dir / 'nesvor_inr.pt')

    # 同原 sample_slices：先切到 eval（train 模式下 INR 返回 (density, pe, z) 元组）
    model.eval()
    query = query_index(case, cfg.interp_step_size)
    mask = _full_mask(case, device)
    blank = torch.zeros_like(images[0])
    with torch.no_grad():
        volume = torch.stack([
            sample_slice(model, _slice(blank, z, case, device), mask, n_samples=args.n_samples).image[0]
            for z in case.z_mm(query)])
    return dict(query_index=query, volume=volume.cpu().numpy(), frame=case.reference_phase)
