/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_AUXILIARY_H_INCLUDED
#define CUDA_RASTERIZER_AUXILIARY_H_INCLUDED

#include "config.h"
#include "stdio.h"

#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define NUM_WARPS (BLOCK_SIZE/32)

// 切片位姿 pose[12]：pose[0..8] 为 R（行优先，三行依次为面内 u、面内 v、法向 n），
// pose[9..11] 为原点 o（像素 (0,0) 中心的坐标）。R 必须正交，所有长度同一单位。

// 覆盖像素坐标 [p - r, p + r] 的 tile 范围（右端不含）。
// 经典库用 (int)((p + r + B - 1) / B)，p 非整数时会漏掉 p + r 所在的最后一个 tile（最多差 1 像素），
// 这里改为 floor((p + r) / B) + 1，保证截断支撑（马氏半径 k 的椭圆）完整落在矩形内。
__forceinline__ __device__ void getRect(const float2 p, int max_radius, uint2& rect_min, uint2& rect_max, dim3 grid)
{
	// 先在浮点域截到 [0, grid] 再转整数，避免远离图像的高斯坐标溢出 int
	const float gx = (float)grid.x, gy = (float)grid.y;
	rect_min = {
		(unsigned int)fminf(gx, fmaxf(0.f, floorf((p.x - max_radius) / BLOCK_X))),
		(unsigned int)fminf(gy, fmaxf(0.f, floorf((p.y - max_radius) / BLOCK_Y)))
	};
	rect_max = {
		(unsigned int)fminf(gx, fmaxf(0.f, floorf((p.x + max_radius) / BLOCK_X) + 1.f)),
		(unsigned int)fminf(gy, fmaxf(0.f, floorf((p.y + max_radius) / BLOCK_Y) + 1.f))
	};
}

// 世界坐标点 → 切片坐标：μ' = R(μ - o)，分量依次为 (u, v, n)
__forceinline__ __device__ float3 worldToSlice(const float3& p, const float* pose)
{
	const float3 d = { p.x - pose[9], p.y - pose[10], p.z - pose[11] };
	return {
		pose[0] * d.x + pose[1] * d.y + pose[2] * d.z,
		pose[3] * d.x + pose[4] * d.y + pose[5] * d.z,
		pose[6] * d.x + pose[7] * d.y + pose[8] * d.z
	};
}

// 切片坐标中的向量梯度 → 世界坐标：dL/dμ = Rᵀ dL/dμ'
__forceinline__ __device__ float3 sliceToWorldVec(const float3& g, const float* pose)
{
	return {
		pose[0] * g.x + pose[3] * g.y + pose[6] * g.z,
		pose[1] * g.x + pose[4] * g.y + pose[7] * g.z,
		pose[2] * g.x + pose[5] * g.y + pose[8] * g.z
	};
}

// 对称矩阵 6 元存储 [xx, xy, xz, yy, yz, zz] 与 3x3 数组互转
__forceinline__ __device__ void sym6ToMat(const float* s, float m[3][3])
{
	m[0][0] = s[0]; m[0][1] = s[1]; m[0][2] = s[2];
	m[1][0] = s[1]; m[1][1] = s[3]; m[1][2] = s[4];
	m[2][0] = s[2]; m[2][1] = s[4]; m[2][2] = s[5];
}

// Y = R X Rᵀ（transpose=false）或 Y = Rᵀ X R（transpose=true），X 对称，结果写成 6 元
__forceinline__ __device__ void congruence(const float X[3][3], const float* pose, bool transpose, float* out6)
{
	float R[3][3];
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			R[i][j] = transpose ? pose[3 * j + i] : pose[3 * i + j];
	float T[3][3];
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			T[i][j] = R[i][0] * X[0][j] + R[i][1] * X[1][j] + R[i][2] * X[2][j];
	const int rows[6] = { 0, 0, 0, 1, 1, 2 };
	const int cols[6] = { 0, 1, 2, 1, 2, 2 };
	for (int k = 0; k < 6; k++)
	{
		const int i = rows[k], j = cols[k];
		out6[k] = T[i][0] * R[j][0] + T[i][1] * R[j][1] + T[i][2] * R[j][2];
	}
}

// ---------------------------------------------------------------------------
// 有限层厚切片足迹（前向与反向共用，反向据此重算中间量）
//
// 切片坐标下 Σ' = R Σ Rᵀ = [[A, b], [bᵀ, c]]，A 为面内 2x2，b 为面内-法向交叉项，c 为法向方差；
// 高斯中心 μ' = R(μ - o)，到切片平面的法向距离 p = μ'_n。
// 层面轮廓取归一化高斯 N(n; 0, s²)，沿法向积分
//     ∫ exp(-½ (x - μ')ᵀ Σ'^{-1} (x - μ')) N(n; 0, s²) dn
// 结果仍是面内二维高斯（等价于 Σ_nn → c + s² 后对 n=0 取条件分布）：
//     q      = c + s²
//     中心   m = μ'_uv - b p / q                      (秩 1 修正，S2)
//     协方差 M = A - b bᵀ / q
//     幅值   a = sqrt(c_cond / (c_cond + s²)) · exp(-p² / (2q)),   c_cond = c - bᵀ A^{-1} b
// 幅值无量纲（S1）：s→0 时 a→exp(-p²/2c)、M→条件协方差，正好是零厚度截面值；
// 离焦衰减 exp(-p²/2q) 只依赖中心深度 p。像素单位：mean_px = m / (du, dv)，
// P = D M D，D = diag(1/du, 1/dv)，conic = P^{-1}。
// ---------------------------------------------------------------------------
struct SliceFootprint
{
	float A00, A01, A11, b0, b1, c;  // Σ' 分块
	float3 mu;                        // μ' = (u, v, p)
	float q, detA, w0, w1, cc;        // w = A^{-1} b，cc = c_cond
	float M00, M01, M11;              // 面内协方差（长度单位）
	float P00, P01, P11, detP;        // 面内协方差（像素单位）
	float2 mean_px;
	float3 conic;                     // [P^{-1}]_00, _01, _11
	float amp;
	bool valid;
};

__forceinline__ __device__ SliceFootprint computeSliceFootprint(
	const float3& mean3D, const float* cov3D, const float* pose,
	const float du, const float dv, const float s2)
{
	SliceFootprint f;
	f.valid = false;

	float S[3][3];
	sym6ToMat(cov3D, S);
	float Sp[6];
	congruence(S, pose, false, Sp);
	f.A00 = Sp[0]; f.A01 = Sp[1]; f.b0 = Sp[2];
	f.A11 = Sp[3]; f.b1 = Sp[4]; f.c = Sp[5];
	f.mu = worldToSlice(mean3D, pose);

	// c_cond = c - bᵀA^{-1}b 是给定面内位置时的法向条件方差（Schur 补），Σ 正定时 > 0
	f.detA = f.A00 * f.A11 - f.A01 * f.A01;
	if (!(f.detA > 0.0f))
		return f;
	f.w0 = (f.A11 * f.b0 - f.A01 * f.b1) / f.detA;
	f.w1 = (f.A00 * f.b1 - f.A01 * f.b0) / f.detA;
	f.cc = f.c - (f.b0 * f.w0 + f.b1 * f.w1);
	if (!(f.cc > 0.0f))
		return f;

	f.q = f.c + s2;
	const float p = f.mu.z;
	f.M00 = f.A00 - f.b0 * f.b0 / f.q;
	f.M01 = f.A01 - f.b0 * f.b1 / f.q;
	f.M11 = f.A11 - f.b1 * f.b1 / f.q;
	f.mean_px = { (f.mu.x - f.b0 * p / f.q) / du, (f.mu.y - f.b1 * p / f.q) / dv };
	f.amp = sqrtf(f.cc / (f.cc + s2)) * expf(-0.5f * p * p / f.q);

	f.P00 = f.M00 / (du * du);
	f.P01 = f.M01 / (du * dv);
	f.P11 = f.M11 / (dv * dv);
	f.detP = f.P00 * f.P11 - f.P01 * f.P01;
	if (!(f.detP > 0.0f))
		return f;
	f.conic = { f.P11 / f.detP, -f.P01 / f.detP, f.P00 / f.detP };
	f.valid = true;
	return f;
}

#define CHECK_CUDA(A, debug) \
A; if(debug) { \
auto ret = cudaDeviceSynchronize(); \
if (ret != cudaSuccess) { \
std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
throw std::runtime_error(cudaGetErrorString(ret)); \
} \
}

#endif
