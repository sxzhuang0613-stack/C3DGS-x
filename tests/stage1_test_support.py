"""Stage1 测试专用参考与显式临时扩展加载；生产代码没有后备渲染路径。"""
import importlib.util
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]


def load_test_extension():
    path = os.environ.get('SLICE_CUDA_EXTENSION')
    if not path:
        try:
            import diff_gaussian_rasterization_slice as package
        except ImportError:
            return False
        return getattr(package, 'API_VERSION', None) == 2 and getattr(package, 'RENDER_MODE', None) == 'alpha'

    name = 'diff_gaussian_rasterization_slice'
    native_name = name+'._C_G3D'
    spec = importlib.util.spec_from_file_location(native_name, path)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    sys.modules[native_name] = native
    package_path = ROOT/'diff-gaussian-rasterization_slice'/name
    spec = importlib.util.spec_from_file_location(name, package_path/'__init__.py',
                                                 submodule_search_locations=[str(package_path)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)
    return True


NATIVE_AVAILABLE = load_test_extension()


def unpack_covariance(packed):
    a, b, c, d, e, f = packed.unbind(-1)
    return torch.stack((a, b, c, b, d, e, c, e, f), -1).reshape(-1, 3, 3)


def activated_covariance(log_scales, rotations):
    from Gaussians.utils import covariance_from_scaling_rotation
    return unpack_covariance(covariance_from_scaling_rotation(log_scales.double().exp(), rotations.double()))


def reference_render(model, img_size, slice_idx, axis, num_slices, **kwargs):
    # 独立 CPU float64 alpha 合成；忽略 tile 包围近似，仅用于内部区域参考。
    means = kwargs.get('xyz', model.get_xyz)
    colors = kwargs.get('colors', model.get_intensity)
    covariance = kwargs.get('covariance')
    if covariance is None:
        from Gaussians.utils import covariance_from_scaling_rotation
        covariance = unpack_covariance(covariance_from_scaling_rotation(
            kwargs.get('scales', model.get_scaling).double(),
            kwargs.get('rotations', model._rotation).double()))
    elif covariance.ndim == 2:
        covariance = unpack_covariance(covariance)
    bounds = kwargs.get('bounds', model.bounds)
    height, width = img_size
    u, v = ((1, 2), (0, 2), (0, 1))[axis]
    y, x = torch.meshgrid(
        ((torch.arange(height, dtype=torch.float64)+.5)*2/height-1)*bounds[v],
        ((torch.arange(width, dtype=torch.float64)+.5)*2/width-1)*bounds[u], indexing='ij')
    plane = 0. if num_slices == 1 else (-1+2*slice_idx/(num_slices-1))*bounds[axis]
    coordinates = [x, y]
    coordinates.insert(axis, torch.full_like(x, plane))
    points = torch.stack(coordinates, -1)
    delta = points[None] - means.to(device='cpu', dtype=torch.float64)[:, None, None]
    covariance = covariance.to(device='cpu', dtype=torch.float64)
    power = -.5*torch.einsum('nhwi,nij,nhwj->nhw',delta,torch.linalg.inv(covariance),delta)
    alpha = power.exp().clamp_max(.99)
    alpha = torch.where(alpha >= 1/255, alpha, 0.)
    order = (means[:, axis].detach().cpu()-plane).abs().argsort(stable=True)
    transmittance = torch.ones_like(x)
    image = torch.zeros_like(x)
    done = torch.zeros_like(x, dtype=torch.bool)
    colors = colors.to(device='cpu', dtype=torch.float64)
    for index in order:
        next_transmittance = transmittance*(1-alpha[index])
        # 与 CUDA 一致：会使 T 低于阈值的当前项不参与，随后终止该像素。
        done = done | (next_transmittance < 1e-4)
        active_alpha = torch.where(done, 0., alpha[index])
        image = image+transmittance*active_alpha*colors[index, 0]
        transmittance = transmittance*(1-active_alpha)
    background = kwargs.get('raster_options', {}).get('background', 0.)
    image = image+background*transmittance
    visible = alpha.flatten(1).sum(-1) > 0
    return {'render':image.to(means), 'visibility_filter':visible.to(means.device)}


def alpha_train_config():
    """config_train.yaml 的旧 alpha 版本：旧模型契约测试不随主配置切换到加和模型而改变。"""
    from config_utils import load_config
    args = load_config(ROOT/'config_train.yaml')
    args.render_model = 'innergs_alpha_fixed_v1'
    args.structural_algorithm = 'batch_residual_v5'
    return args
