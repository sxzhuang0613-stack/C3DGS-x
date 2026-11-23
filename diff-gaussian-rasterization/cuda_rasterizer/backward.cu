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

#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// 梯度约定：对称矩阵的 6 元 / 3 元存储都把非对角元视为一个独立参数，
// 例如 power = -½(C00 dx² + C11 dy²) - C01 dx dy 时 ∂power/∂C01 = -dx dy；
// 对应的矩阵梯度为 [[g00, g01/2], [g01/2, g11]]。dL_dcov3D 与经典库约定一致。

// Backward pass for the conversion of scale and rotation to a
// 3D covariance matrix for each Gaussian.（与经典库相同）
__device__ void computeCov3D(int idx, const glm::vec3 scale, float mod, const glm::vec4 rot, const float* dL_dcov3Ds, glm::vec3* dL_dscales, glm::vec4* dL_drots)
{
	// Recompute (intermediate) results for the 3D covariance computation.
	glm::vec4 q = rot;
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 S = glm::mat3(1.0f);

	glm::vec3 s = mod * scale;
	S[0][0] = s.x;
	S[1][1] = s.y;
	S[2][2] = s.z;

	glm::mat3 M = S * R;

	const float* dL_dcov3D = dL_dcov3Ds + 6 * idx;

	// Convert per-element covariance loss gradients to matrix form
	glm::mat3 dL_dSigma = glm::mat3(
		dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
		0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
		0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]
	);

	// Compute loss gradient w.r.t. matrix M
	// dSigma_dM = 2 * M
	glm::mat3 dL_dM = 2.0f * M * dL_dSigma;

	glm::mat3 Rt = glm::transpose(R);
	glm::mat3 dL_dMt = glm::transpose(dL_dM);

	// Gradients of loss w.r.t. scale
	glm::vec3* dL_dscale = dL_dscales + idx;
	dL_dscale->x = glm::dot(Rt[0], dL_dMt[0]);
	dL_dscale->y = glm::dot(Rt[1], dL_dMt[1]);
	dL_dscale->z = glm::dot(Rt[2], dL_dMt[2]);

	dL_dMt[0] *= s.x;
	dL_dMt[1] *= s.y;
	dL_dMt[2] *= s.z;

	// Gradients of loss w.r.t. quaternion
	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
	dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
	dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
	dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

	float4* dL_drot = (float4*)(dL_drots + idx);
	*dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };
}

// 预处理的反向。前向链路（见 auxiliary.h）：
//   Σ → Σ' = RΣRᵀ → (A, b, c)；μ → μ' = R(μ-o) → (u, v, p)；q = c + s²
//   m = μ'_uv - b p/q,  M = A - bbᵀ/q,  cc = c - bᵀA^{-1}b,  a = sqrt(cc/(cc+s²))·exp(-p²/2q)
//   mean_px = m/(du,dv),  P = D M D,  conic = P^{-1}
// 反向逐级回传，最后由 dΣ = Rᵀ dΣ' R、dμ = Rᵀ dμ' 回到世界坐标。
__global__ void preprocessCUDA(
	int P,
	const float3* means,
	const int* radii,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* cov3Ds,
	const float* slice_pose,
	const float du, const float dv,
	const float slice_sigma,
	const float3* dL_dmean2D,
	const float3* dL_dconics,
	const float* dL_damps,
	glm::vec3* dL_dmeans,
	float* dL_dcov3D,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	// 重算前向中间量（与经典库一样不在前向中缓存）
	const float s2 = slice_sigma * slice_sigma;
	const SliceFootprint f = computeSliceFootprint(means[idx], cov3Ds + 6 * idx, slice_pose, du, dv, s2);
	const float q = f.q, p = f.mu.z, b0 = f.b0, b1 = f.b1;

	// 1) 像素中心 → 切片面内中心 m：mean_px = m / (du, dv)
	const float g_m0 = dL_dmean2D[idx].x / du;
	const float g_m1 = dL_dmean2D[idx].y / dv;

	// 2) conic → 像素协方差 P：C = P^{-1} ⇒ dL/dP = -C Ḡ C，Ḡ = [[gx, gy/2], [gy/2, gz]]
	const float ca = f.conic.x, cb = f.conic.y, cd = f.conic.z;
	const float3 gC = dL_dconics[idx];
	const float gx = gC.x, gy = 0.5f * gC.y, gz = gC.z;
	const float g_P00 = -(ca * ca * gx + 2.f * ca * cb * gy + cb * cb * gz);
	const float g_P01 = -2.f * (ca * cb * gx + (ca * cd + cb * cb) * gy + cb * cd * gz);
	const float g_P11 = -(cb * cb * gx + 2.f * cb * cd * gy + cd * cd * gz);

	// 3) P = D M D：P00 = M00/du², P01 = M01/(du dv), P11 = M11/dv²
	const float g_M00 = g_P00 / (du * du);
	const float g_M01 = g_P01 / (du * dv);
	const float g_M11 = g_P11 / (dv * dv);

	// 4) 幅值：ln a = ½ ln cc - ½ ln(cc+s²) - p²/(2q)
	//    ∂ln a/∂cc = s² / (2 cc (cc+s²)),  ∂ln a/∂p = -p/q,  ∂ln a/∂q = p²/(2q²)
	const float g_lna = dL_damps[idx] * f.amp;
	const float g_cc = g_lna * 0.5f * s2 / (f.cc * (f.cc + s2));
	float g_p = -g_lna * p / q;
	float g_q = g_lna * 0.5f * p * p / (q * q);

	// 5) cc = c - β，β = bᵀA^{-1}b，w = A^{-1}b：∂β/∂b = 2w，∂β/∂A = -w wᵀ
	float g_c = g_cc;
	const float g_beta = -g_cc;
	float g_b0 = 2.f * g_beta * f.w0;
	float g_b1 = 2.f * g_beta * f.w1;
	float g_A00 = -g_beta * f.w0 * f.w0;
	float g_A01 = -2.f * g_beta * f.w0 * f.w1;
	float g_A11 = -g_beta * f.w1 * f.w1;

	// 6) m = μ'_uv - b p / q
	const float g_mu0 = g_m0;
	const float g_mu1 = g_m1;
	g_b0 -= g_m0 * p / q;
	g_b1 -= g_m1 * p / q;
	g_p -= (g_m0 * b0 + g_m1 * b1) / q;
	g_q += (g_m0 * b0 + g_m1 * b1) * p / (q * q);

	// 7) M = A - b bᵀ / q
	g_A00 += g_M00;
	g_A01 += g_M01;
	g_A11 += g_M11;
	g_b0 -= (2.f * b0 * g_M00 + b1 * g_M01) / q;
	g_b1 -= (2.f * b1 * g_M11 + b0 * g_M01) / q;
	g_q += (b0 * b0 * g_M00 + b0 * b1 * g_M01 + b1 * b1 * g_M11) / (q * q);

	// 8) q = c + s²（本轮不回传 s 的梯度）
	g_c += g_q;

	// 9) Σ' 梯度（矩阵形式，非对角元取参数梯度的一半）→ dΣ = Rᵀ dΣ' R → 6 元参数梯度
	const float Gs[3][3] = {
		{ g_A00, 0.5f * g_A01, 0.5f * g_b0 },
		{ 0.5f * g_A01, g_A11, 0.5f * g_b1 },
		{ 0.5f * g_b0, 0.5f * g_b1, g_c }
	};
	float Gw[6];
	congruence(Gs, slice_pose, true, Gw);
	float* dcov = dL_dcov3D + 6 * idx;
	dcov[0] = Gw[0];
	dcov[1] = 2.f * Gw[1];
	dcov[2] = 2.f * Gw[2];
	dcov[3] = Gw[3];
	dcov[4] = 2.f * Gw[4];
	dcov[5] = Gw[5];

	// 10) μ' = R(μ - o) ⇒ dL/dμ = Rᵀ dL/dμ'
	const float3 g_mu = sliceToWorldVec({ g_mu0, g_mu1, g_p }, slice_pose);
	dL_dmeans[idx] = { g_mu.x, g_mu.y, g_mu.z };

	// 11) 协方差 → 尺度与旋转（仅在未直接给定协方差时）
	if (scales)
		computeCov3D(idx, scales[idx], scale_modifier, rotations[idx], dL_dcov3D, dL_dscale, dL_drot);
}

__device__ __forceinline__ float warpSum(float v)
{
	for (int offset = 16; offset > 0; offset >>= 1)
		v += __shfl_down_sync(0xffffffff, v, offset);
	return v;
}

// 加和渲染的反向。C(p) = bg + Σ_i I_i w_i(p)，w = a·G，G = exp(power)：
//   ∂C/∂I = w，∂C/∂a = I·G，∂C/∂G = I·a
//   ∂G/∂mean_px = G·(-conic·d)，∂G/∂conic = G·(-½dx², -dx dy, -½dy²)
// 加和与顺序无关，无需像经典库那样从后向前恢复透射率 T。
// 同一 warp 内所有线程同步遍历同一个高斯，先做 warp 归约再由 lane 0 原子累加，
// 把每像素一次的 atomicAdd 降为每 32 像素一次（没有提前终止时这是主要开销）。
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_amp,
	const float* __restrict__ colors,
	const float amp_cutoff,
	const float* __restrict__ dL_dpixels,
	float3* __restrict__ dL_dmean2D,
	float3* __restrict__ dL_dconic2D,
	float* __restrict__ dL_damp,
	float* __restrict__ dL_dcolors)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };

	const bool inside = pix.x < W && pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;
	const bool lane0 = (block.thread_rank() % 32) == 0;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_amp[BLOCK_SIZE];
	__shared__ float collected_colors[C * BLOCK_SIZE];

	float dL_dpixel[C] = { 0 };
	if (inside)
		for (int i = 0; i < C; i++)
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];

	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_amp[block.thread_rank()] = conic_amp[coll_id];
			for (int ch = 0; ch < C; ch++)
				collected_colors[ch * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + ch];
		}
		block.sync();

		// 所有线程（含越界像素）一起循环，保证 warp 归约时 32 个 lane 都在
		for (int j = 0; j < min(BLOCK_SIZE, toDo); j++)
		{
			const float2 xy = collected_xy[j];
			const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			const float4 con_a = collected_conic_amp[j];
			const float power = -0.5f * (con_a.x * d.x * d.x + con_a.z * d.y * d.y) - con_a.y * d.x * d.y;
			// 与前向完全相同的截断支撑；支撑外令 G=0，避免 expf 溢出后 inf·0 产生 NaN
			const bool in_support = inside && power <= 0.0f && power >= -MAX_POWER_CUTOFF;
			const float G = in_support ? expf(power) : 0.0f;
			const float w = con_a.w * G;
			const bool valid = in_support && w >= amp_cutoff;
			if (!__any_sync(0xffffffff, valid))
				continue;

			float g_I[C];
			float dL_dw = 0.0f;
			for (int ch = 0; ch < C; ch++)
			{
				g_I[ch] = valid ? w * dL_dpixel[ch] : 0.0f;
				dL_dw += collected_colors[ch * BLOCK_SIZE + j] * dL_dpixel[ch];
			}
			dL_dw = valid ? dL_dw : 0.0f;
			const float g_amp = G * dL_dw;
			const float dL_dG = con_a.w * dL_dw;
			const float gdx = G * d.x;
			const float gdy = G * d.y;
			// d = mean_px - pixel，∂d/∂mean_px = 1
			const float g_mx = dL_dG * (-gdx * con_a.x - gdy * con_a.y);
			const float g_my = dL_dG * (-gdy * con_a.z - gdx * con_a.y);
			const float g_cx = -0.5f * gdx * d.x * dL_dG;
			const float g_cy = -gdx * d.y * dL_dG;
			const float g_cz = -0.5f * gdy * d.y * dL_dG;

			float sum_I[C];
			for (int ch = 0; ch < C; ch++)
				sum_I[ch] = warpSum(g_I[ch]);
			const float sum_amp = warpSum(g_amp);
			const float sum_mx = warpSum(g_mx);
			const float sum_my = warpSum(g_my);
			const float sum_cx = warpSum(g_cx);
			const float sum_cy = warpSum(g_cy);
			const float sum_cz = warpSum(g_cz);
			if (lane0)
			{
				const int global_id = collected_id[j];
				for (int ch = 0; ch < C; ch++)
					atomicAdd(&dL_dcolors[global_id * C + ch], sum_I[ch]);
				atomicAdd(&dL_damp[global_id], sum_amp);
				atomicAdd(&dL_dmean2D[global_id].x, sum_mx);
				atomicAdd(&dL_dmean2D[global_id].y, sum_my);
				atomicAdd(&dL_dconic2D[global_id].x, sum_cx);
				atomicAdd(&dL_dconic2D[global_id].y, sum_cy);
				atomicAdd(&dL_dconic2D[global_id].z, sum_cz);
			}
		}
	}
}

void BACKWARD::preprocess(
	int P,
	const float3* means3D,
	const int* radii,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* cov3Ds,
	const float* slice_pose,
	const float pixel_du, const float pixel_dv,
	const float slice_sigma,
	const float3* dL_dmean2D,
	const float3* dL_dconic,
	const float* dL_damp,
	glm::vec3* dL_dmean3D,
	float* dL_dcov3D,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot)
{
	preprocessCUDA << < (P + 255) / 256, 256 >> > (
		P,
		means3D,
		radii,
		scales,
		rotations,
		scale_modifier,
		cov3Ds,
		slice_pose,
		pixel_du, pixel_dv,
		slice_sigma,
		dL_dmean2D,
		dL_dconic,
		dL_damp,
		dL_dmean3D,
		dL_dcov3D,
		dL_dscale,
		dL_drot);
}

void BACKWARD::render(
	const dim3 grid, const dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float4* conic_amp,
	const float* intensities,
	const float amp_cutoff,
	const float* dL_dpixels,
	float3* dL_dmean2D,
	float3* dL_dconic2D,
	float* dL_damp,
	float* dL_dintensities)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> >(
		ranges,
		point_list,
		W, H,
		means2D,
		conic_amp,
		intensities,
		amp_cutoff,
		dL_dpixels,
		dL_dmean2D,
		dL_dconic2D,
		dL_damp,
		dL_dintensities
		);
}
