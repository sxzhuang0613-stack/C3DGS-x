"""MRI 切片渲染库验证：闭式解、极限、CUDA 前向/反向与参考实现对照。

运行：python tests/test_slice_mri.py（或 python -m pytest tests/test_slice_mri.py -q）；
需先编译安装，或 build_ext --inplace 后在仓库根目录设置 PYTHONPATH=.
"""
import itertools
import os
import sys

import torch

try:
    import pytest
except ImportError:  # 无 pytest 时用最小替身，配合文件末尾的直接运行入口
    class _Skip(Exception):
        pass

    class _Mark:
        @staticmethod
        def skipif(condition, reason=''):
            def wrap(fn):
                fn._skip = reason if condition else None
                return fn
            return wrap

        @staticmethod
        def parametrize(name, values):
            def wrap(fn):
                fn._params = [(name, values)] + getattr(fn, '_params', [])
                return fn
            return wrap

    class pytest:  # noqa: N801
        mark = _Mark()
        Skip = _Skip

        @staticmethod
        def importorskip(name):
            try:
                return __import__(name)
            except ImportError as error:
                raise _Skip(str(error))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reference as ref  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='需要 CUDA')


def random_rotation(generator, dtype=torch.float64):
    """随机右手正交矩阵 [3,3]。"""
    q, r = torch.linalg.qr(torch.randn(3, 3, generator=generator, dtype=dtype))
    q = q * torch.sign(torch.diagonal(r))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def random_scene(n, H, W, du, dv, R, o, generator, scale_range=(0.6, 3.0)):
    """在切片附近随机放置各向异性高斯（double，CPU）。"""
    u, v, nrm = R[0], R[1], R[2]
    px = torch.rand(n, 1, generator=generator, dtype=torch.float64) * (W - 1)
    py = torch.rand(n, 1, generator=generator, dtype=torch.float64) * (H - 1)
    pz = (torch.rand(n, 1, generator=generator, dtype=torch.float64) - 0.5) * 6.0
    means = o + px * du * u + py * dv * v + pz * nrm
    lo, hi = scale_range
    scales = lo + (hi - lo) * torch.rand(n, 3, generator=generator, dtype=torch.float64)
    rotations = torch.nn.functional.normalize(torch.randn(n, 4, generator=generator, dtype=torch.float64), dim=-1)
    intensities = torch.randn(n, 1, generator=generator, dtype=torch.float64)
    return means, scales, rotations, intensities


# ---------------------------------------------------------------- 数学（CPU, double）

@pytest.mark.parametrize('s', [0.0, 0.7, 2.5])
def test_closed_form_matches_numerical_integral(s):
    """闭式足迹 a·exp(power) 与沿法向数值积分一致；s=0 时即零厚度截面值。"""
    g = torch.Generator().manual_seed(1)
    R = random_rotation(g)
    o = torch.randn(3, generator=g, dtype=torch.float64)
    du, dv = 0.8, 1.1
    means, scales, rotations, _ = random_scene(6, 12, 14, du, dv, R, o, g)
    cov = ref.covariance(scales, rotations)
    pixels = torch.stack(torch.meshgrid(torch.arange(14.), torch.arange(12.), indexing='xy'), -1).reshape(-1, 2).double()

    mean_px, cov_px, amp = ref.footprint(means, cov, R, o, du, dv, s)
    d = mean_px[:, None] - pixels[None]
    closed = amp[:, None] * torch.exp(-0.5 * torch.einsum('nki,nij,nkj->nk', d, torch.linalg.inv(cov_px), d))
    numeric = ref.slab_integral_numeric(means, cov, R, o, du, dv, s, pixels)
    assert torch.allclose(closed, numeric, atol=1e-7, rtol=1e-5)


def test_large_thickness_limit():
    """s→∞：M→A，a·s→sqrt(c_cond)（线积分除以轮廓宽度）。"""
    g = torch.Generator().manual_seed(2)
    R = random_rotation(g)
    o = torch.zeros(3, dtype=torch.float64)
    means, scales, rotations, _ = random_scene(5, 8, 8, 1.0, 1.0, R, o, g)
    cov = ref.covariance(scales, rotations)
    Sp = R @ cov @ R.T
    A, b, c = Sp[:, :2, :2], Sp[:, :2, 2], Sp[:, 2, 2]
    cc = c - (b[:, None, :] @ torch.linalg.solve(A, b[:, :, None]))[:, 0, 0]
    s = 1e5
    _, cov_px, amp = ref.footprint(means, cov, R, o, 1.0, 1.0, s)
    assert torch.allclose(cov_px, A, rtol=1e-6)
    assert torch.allclose(amp * s, torch.sqrt(cc), rtol=1e-6)


# ---------------------------------------------------------------- CUDA

def cuda_render(means, intensities, R, o, du, dv, s, H, W, bg=0.0, cutoff=1 / 255, scales=None,
                rotations=None, cov6=None):
    """调用 CUDA 库渲染，返回 (图像 [H,W], radii, 幅值, means2D)。"""
    import diff_gaussian_rasterization_mri as mri
    settings = mri.SliceRasterizationSettings(
        image_height=H, image_width=W, slice_rotation=R.float().cuda(), slice_origin=o.float().cuda(),
        pixel_spacing=(du, dv), slice_sigma=s, bg=bg, amplitude_cutoff=cutoff)
    means2D = torch.zeros_like(means, requires_grad=True)
    image, radii, amps = mri.SliceRasterizer(settings)(
        means, means2D, intensities, scales=scales, rotations=rotations, cov3D_precomp=cov6)
    return image[0], radii, amps, means2D


def make_case(seed, n=300, H=45, W=52, du=0.9, dv=1.2, oblique=True):
    g = torch.Generator().manual_seed(seed)
    R = random_rotation(g) if oblique else torch.eye(3, dtype=torch.float64)
    o = torch.randn(3, generator=g, dtype=torch.float64) * 3
    means, scales, rotations, intensities = random_scene(n, H, W, du, dv, R, o, g)
    return dict(R=R, o=o, du=du, dv=dv, H=H, W=W, means=means, scales=scales, rotations=rotations,
                intensities=intensities)


def compare_images(a, b):
    """返回 (最大误差, 超过 1e-4 的像素比例)；截断边界的个别像素允许差一个 ε 量级。"""
    diff = (a - b).abs()
    return float(diff.max()), float((diff > 1e-4).double().mean())


@cuda
@pytest.mark.parametrize('s', [0.0, 1.5])
def test_cuda_forward_matches_reference(s):
    c = make_case(3)
    cov = ref.covariance(c['scales'], c['rotations'])
    expected = ref.render(c['means'], cov, c['intensities'], c['R'], c['o'], c['du'], c['dv'], s,
                          c['H'], c['W'], bg=0.1)
    f = lambda x: x.float().cuda().contiguous()
    img_cov, _, amps, _ = cuda_render(f(c['means']), f(c['intensities']), c['R'], c['o'], c['du'], c['dv'], s,
                                      c['H'], c['W'], bg=0.1, cov6=f(ref.mat_to_cov6(cov)))
    img_sr, _, _, _ = cuda_render(f(c['means']), f(c['intensities']), c['R'], c['o'], c['du'], c['dv'], s,
                                  c['H'], c['W'], bg=0.1, scales=f(c['scales']), rotations=f(c['rotations']))
    for img in (img_cov, img_sr):
        max_err, frac = compare_images(img.double().cpu(), expected)
        assert frac < 1e-3 and max_err < 1e-2, (max_err, frac)  # 实测 max_err≈5e-6；仅容许截断边界个别像素
    _, _, amp_ref = ref.footprint(c['means'], cov, c['R'], c['o'], c['du'], c['dv'], s)
    visible = amps.cpu() > 0
    assert torch.allclose(amps.double().cpu()[visible], amp_ref[visible], rtol=1e-4, atol=1e-6)


@cuda
def test_forward_is_deterministic():
    c = make_case(4)
    f = lambda x: x.float().cuda().contiguous()
    args = (f(c['means']), f(c['intensities']), c['R'], c['o'], c['du'], c['dv'], 1.0, c['H'], c['W'])
    kw = dict(scales=f(c['scales']), rotations=f(c['rotations']))
    assert torch.equal(cuda_render(*args, **kw)[0], cuda_render(*args, **kw)[0])


def relative_error(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


@cuda
@pytest.mark.parametrize('s', [0.0, 1.5])
@pytest.mark.parametrize('use_cov', [True, False])
def test_cuda_backward_matches_reference(s, use_cov):
    """CUDA 手写梯度与参考实现的 autograd 梯度一致（截断掩码视为常数）。"""
    c = make_case(5, n=120)
    weights = torch.randn(c['H'], c['W'], generator=torch.Generator().manual_seed(9), dtype=torch.float64)

    # 参考（double）
    means = c['means'].clone().requires_grad_()
    intens = c['intensities'].clone().requires_grad_()
    means2D_ref = torch.zeros_like(means, requires_grad=True)
    if use_cov:
        cov6 = ref.mat_to_cov6(ref.covariance(c['scales'], c['rotations'])).requires_grad_()
        leaves = [means, intens, means2D_ref, cov6]
        cov = ref.cov6_to_mat(cov6)
    else:
        scales = c['scales'].clone().requires_grad_()
        rots = c['rotations'].clone().requires_grad_()
        leaves = [means, intens, means2D_ref, scales, rots]
        cov = ref.covariance(scales, rots)
    img = ref.render(means, cov, intens, c['R'], c['o'], c['du'], c['dv'], s, c['H'], c['W'], means2D=means2D_ref)
    (img * weights).sum().backward()
    ref_grads = [leaf.grad for leaf in leaves]

    # CUDA（float32）
    f = lambda x: x.float().cuda().contiguous().requires_grad_()
    means_c, intens_c = f(c['means']), f(c['intensities'])
    if use_cov:
        cov_c = f(cov6.detach())
        extra = dict(cov6=cov_c)
        cu_leaves = [means_c, intens_c, None, cov_c]
    else:
        sc_c, rot_c = f(c['scales']), f(c['rotations'])
        extra = dict(scales=sc_c, rotations=rot_c)
        cu_leaves = [means_c, intens_c, None, sc_c, rot_c]
    img_c, _, _, means2D_c = cuda_render(means_c, intens_c, c['R'], c['o'], c['du'], c['dv'], s,
                                         c['H'], c['W'], **extra)
    (img_c * weights.float().cuda()).sum().backward()
    cu_leaves[2] = means2D_c
    names = ['means3D', 'intensities', 'means2D', 'cov3D'] if use_cov else \
        ['means3D', 'intensities', 'means2D', 'scales', 'rotations']
    for name, leaf, expected in zip(names, cu_leaves, ref_grads):
        got = leaf.grad.double().cpu()
        if name == 'means2D':
            got, expected = got[:, :2], expected[:, :2]
        err = relative_error(got, expected)
        assert err < 1e-4, (name, err)  # 实测 ≈3e-6


@cuda
def test_zero_intensity_still_gets_gradient():
    """剔除只看几何幅值：强度为 0 的可见高斯仍有强度梯度，不会“饿死”。"""
    R, o = torch.eye(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64)
    means = torch.tensor([[10.0, 10.0, 0.5]], device='cuda')
    intens = torch.zeros(1, 1, device='cuda', requires_grad=True)
    scales = torch.full((1, 3), 2.0, device='cuda')
    rots = torch.tensor([[1.0, 0, 0, 0]], device='cuda')
    img, radii, _, _ = cuda_render(means, intens, R, o, 1.0, 1.0, 1.0, 20, 20, scales=scales, rotations=rots)
    img.sum().backward()
    assert radii.item() > 0 and intens.grad.item() > 0


@cuda
def test_oblique_slice_equals_rotated_scene():
    """S6：斜切片渲染 = 把高斯变换到切片坐标后用单位位姿渲染。"""
    c = make_case(6, oblique=True)
    f = lambda x: x.float().cuda().contiguous()
    cov = ref.covariance(c['scales'], c['rotations'])
    img_oblique = cuda_render(f(c['means']), f(c['intensities']), c['R'], c['o'], c['du'], c['dv'], 1.2,
                              c['H'], c['W'], cov6=f(ref.mat_to_cov6(cov)))[0]
    means_local = (c['means'] - c['o']) @ c['R'].T
    cov_local = c['R'] @ cov @ c['R'].T
    eye, zero = torch.eye(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64)
    img_local = cuda_render(f(means_local), f(c['intensities']), eye, zero, c['du'], c['dv'], 1.2,
                            c['H'], c['W'], cov6=f(ref.mat_to_cov6(cov_local)))[0]
    max_err, frac = compare_images(img_oblique, img_local)
    assert frac < 1e-3 and max_err < 1e-2


@cuda
def test_zero_thickness_matches_existing_slice_library():
    """s=0 且单个孤立高斯时，与现有零厚度 alpha 切片库一致（α=G<0.99、bg=0 时 C = I·G）。"""
    slice_lib = pytest.importorskip('diff_gaussian_rasterization_slice')
    H = W = 64
    mean = torch.tensor([[0.1, -0.2, 0.05]], device='cuda')
    cov6 = torch.tensor([[0.02, 0.004, 0.003, 0.015, -0.002, 0.01]], device='cuda')
    # 切片库约定：像素中心归一化坐标 (2j+1)/W - 1，z 平面位于 campos[axis]
    settings = slice_lib.GaussianRasterizationSettings(
        image_height=H, image_width=W, tanfovx=1., tanfovy=1., bg=torch.zeros(3, device='cuda'),
        scale_modifier=1., viewmatrix=torch.eye(4, device='cuda'), projmatrix=torch.eye(4, device='cuda'),
        sh_degree=0, campos=torch.tensor([0., 0., 0.], device='cuda'), prefiltered=False, debug=False,
        antialiasing=False, axis=2)
    old, _, _ = slice_lib.GaussianRasterizer(settings)(
        means3D=mean, means2D=torch.zeros_like(mean), colors_precomp=torch.full((1, 3), 0.8, device='cuda'),
        opacities=torch.ones(1, 1, device='cuda'), cov3D_precomp=cov6)
    o = torch.tensor([-1 + 1 / W, -1 + 1 / H, 0.0], dtype=torch.float64)
    new = cuda_render(mean, torch.full((1, 1), 0.8, device='cuda'), torch.eye(3, dtype=torch.float64), o,
                      2 / W, 2 / H, 0.0, H, W, cov6=cov6)[0]
    both = (old[0] > 0) & (new > 0)
    assert both.sum() > 100
    assert torch.allclose(old[0][both], new[both], atol=1e-5)


if __name__ == '__main__':
    failed = 0
    for name, fn in list(globals().items()):
        if not (name.startswith('test_') and callable(fn)):
            continue
        if getattr(fn, '_skip', None):
            print(f'SKIP {name}: {fn._skip}')
            continue
        params = getattr(fn, '_params', [])
        keys = [k for k, _ in params]
        for combo in itertools.product(*[v for _, v in params]):
            label = name + (str(dict(zip(keys, combo))) if keys else '')
            try:
                fn(**dict(zip(keys, combo)))
                print(f'PASS {label}')
            except Exception as error:  # 汇报所有失败后统一返回非零
                if type(error).__name__ == '_Skip':
                    print(f'SKIP {label}: {error}')
                    continue
                failed += 1
                print(f'FAIL {label}: {type(error).__name__}: {error}')
    sys.exit(1 if failed else 0)
