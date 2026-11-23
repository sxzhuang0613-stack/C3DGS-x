"""纯 PyTorch 参考实现：逐像素稠密计算，供 CUDA 前向/反向对照；不追求速度。"""
import torch

MAX_POWER_CUTOFF = 4.5


def quat_to_rotmat(q):
    """四元数 (w, x, y, z) → 旋转矩阵 [N,3,3]，与 CUDA computeCov3D 一致（不归一化）。"""
    r, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y),
        2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x),
        2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y),
    ], -1).reshape(*q.shape[:-1], 3, 3)


def covariance(scales, rotations, scale_modifier=1.0):
    """Σ = R S² Rᵀ，[N,3,3]。"""
    R = quat_to_rotmat(rotations)
    S = scale_modifier * scales
    return R @ torch.diag_embed(S * S) @ R.transpose(-1, -2)


def cov6_to_mat(cov6):
    """[N,6]（xx, xy, xz, yy, yz, zz）→ [N,3,3]。"""
    xx, xy, xz, yy, yz, zz = cov6.unbind(-1)
    return torch.stack([xx, xy, xz, xy, yy, yz, xz, yz, zz], -1).reshape(-1, 3, 3)


def mat_to_cov6(m):
    return torch.stack([m[:, 0, 0], m[:, 0, 1], m[:, 0, 2], m[:, 1, 1], m[:, 1, 2], m[:, 2, 2]], -1)


def footprint(means, cov, R, o, du, dv, s):
    """闭式层厚积分足迹。返回像素中心 [N,2]、像素协方差 [N,2,2]、幅值 [N]。

    Σ' = RΣRᵀ = [[A, b], [bᵀ, c]]，μ' = R(μ-o)，p = μ'_n，q = c + s²：
    m = μ'_uv - b p/q，M = A - bbᵀ/q，a = sqrt(cc/(cc+s²))·exp(-p²/2q)，cc = c - bᵀA⁻¹b。
    """
    Sp = R @ cov @ R.T
    mu = (means - o) @ R.T
    A, b, c = Sp[:, :2, :2], Sp[:, :2, 2], Sp[:, 2, 2]
    p = mu[:, 2]
    q = c + s * s
    m = mu[:, :2] - b * (p / q)[:, None]
    M = A - b[:, :, None] * b[:, None, :] / q[:, None, None]
    cc = c - (b[:, None, :] @ torch.linalg.solve(A, b[:, :, None]))[:, 0, 0]
    amp = torch.sqrt(cc / (cc + s * s)) * torch.exp(-0.5 * p * p / q)
    D = torch.diag(torch.tensor([1.0 / du, 1.0 / dv], dtype=means.dtype, device=means.device))
    return m @ D, D @ M @ D, amp


def pixel_grid(H, W, dtype, device):
    """像素中心的像素坐标 [H,W,2]，(x=列, y=行)。"""
    ys, xs = torch.meshgrid(torch.arange(H, dtype=dtype, device=device),
                            torch.arange(W, dtype=dtype, device=device), indexing='ij')
    return torch.stack([xs, ys], -1)


def render(means, cov, intensities, R, o, du, dv, s, H, W, bg=0.0, cutoff=1.0 / 255.0,
           means2D=None, truncate=True):
    """稠密参考渲染 [H,W]；truncate=True 时使用与 CUDA 相同的截断支撑（作常数掩码处理）。"""
    mean_px, cov_px, amp = footprint(means, cov, R, o, du, dv, s)
    if means2D is not None:
        mean_px = mean_px + means2D[:, :2]
    conic = torch.linalg.inv(cov_px)
    d = mean_px[:, None, None, :] - pixel_grid(H, W, means.dtype, means.device)[None]
    power = -0.5 * torch.einsum('nhwi,nij,nhwj->nhw', d, conic, d)
    w = amp[:, None, None] * torch.exp(power)
    if truncate:
        with torch.no_grad():
            mask = (power >= -MAX_POWER_CUTOFF) & (w >= cutoff)
        w = w * mask
    return bg + (intensities[:, 0, None, None] * w).sum(0)


def slab_integral_numeric(means, cov, R, o, du, dv, s, pixels, num=4001, span=8.0):
    """数值积分 ∫ G3D(x + z n) N(z; 0, s²) dz，pixels 为 [K,2] 像素坐标；s=0 时直接取截面值。"""
    inv = torch.linalg.inv(cov)
    u, v, n = R[0], R[1], R[2]
    base = o + pixels[:, :1] * du * u + pixels[:, 1:] * dv * v  # [K,3]
    if s == 0:
        z = torch.zeros(1, dtype=means.dtype)
        weight = torch.ones(1, dtype=means.dtype)
    else:
        z = torch.linspace(-span * s, span * s, num, dtype=means.dtype)
        weight = torch.exp(-0.5 * (z / s) ** 2) / (s * (2 * torch.pi) ** 0.5) * (z[1] - z[0])
    pts = base[None, :, None, :] + z[None, None, :, None] * n  # [1,K,Z,3]
    d = pts - means[:, None, None, :]  # [N,K,Z,3]
    G = torch.exp(-0.5 * torch.einsum('nkzi,nij,nkzj->nkz', d, inv, d))
    return (G * weight).sum(-1)  # [N,K]
