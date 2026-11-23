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

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space.（与经典库相同，四元数不在此归一化）
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	glm::vec4 q = rot;
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// 逐高斯预处理。公式见 auxiliary.h 中 computeSliceFootprint 的说明。
__global__ void preprocessCUDA(int P,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* cov3D_precomp,
	const float* slice_pose,
	const int W, int H,
	const float pixel_du, const float pixel_dv,
	const float slice_sigma,
	const float amp_cutoff,
	int* radii,
	float2* points_xy_image,
	float* cov3Ds,
	float4* conic_amp,
	float* amplitudes,
	const dim3 grid,
	uint32_t* tiles_touched,
	uint4* tile_rects)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// radii=0 表示该高斯不参与本切片的渲染与反向
	radii[idx] = 0;
	tiles_touched[idx] = 0;
	amplitudes[idx] = 0.0f;

	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	const float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	const SliceFootprint f = computeSliceFootprint(
		p_orig, cov3D, slice_pose, pixel_du, pixel_dv, slice_sigma * slice_sigma);
	if (!f.valid)
		return;
	amplitudes[idx] = f.amp;

	// 剔除只看几何幅值 a（不乘强度），保证强度接近 0 的高斯仍能拿到强度梯度。
	// 足迹在像素上的权重 w = a·exp(power)；w ≥ ε 且 power ≥ -k_max²/2 才计入，
	// 对应马氏半径 k = min(k_max, sqrt(2 ln(a/ε)))（S8：幅值越小覆盖越少）。
	if (f.amp < amp_cutoff)
		return;
	const float k = sqrtf(min(2.0f * MAX_POWER_CUTOFF, 2.0f * logf(f.amp / amp_cutoff)));

	// 像素协方差的最大特征值给出包围半径（与经典库同样对判别式取 0.1 下限，只会略放大矩形）
	const float mid = 0.5f * (f.P00 + f.P11);
	const float lambda1 = mid + sqrtf(max(0.1f, mid * mid - f.detP));
	const float my_radius = ceilf(k * sqrtf(lambda1));
	// a 落在 ε 的 1 ulp 内时 a/ε 舍入为 1，k=0 → my_radius=0；但 getRect 对半径 0 仍给出 1 个 tile，
	// 于是 tiles_touched=1 而 radii=0，duplicateWithKeys 按 radii>0 跳过该点，留下未写入的键槽，
	// 显存残值被当作 tile 编号使 identifyTileRanges 越界（训练中实测的偶发非法访存）。
	// 这类点处处 w=a·G≤a≈ε，无可见贡献，直接剔除；!(≥1) 同时剔除 NaN。
	if (!(my_radius >= 1.0f))
		return;
	uint2 rect_min, rect_max;
	getRect(f.mean_px, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	radii[idx] = my_radius;
	points_xy_image[idx] = f.mean_px;
	conic_amp[idx] = { f.conic.x, f.conic.y, f.conic.z, f.amp };
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
	tile_rects[idx] = { rect_min.x, rect_min.y, rect_max.x, rect_max.y };
}

// 逐 tile 渲染，每个线程一个像素。MRI 信号线性叠加：
//     C(p) = bg + Σ_i I_i · a_i · exp(power_i(p)),
//     power = -½ dᵀ conic d,  d = mean_px - p
// 没有透射率、0.99 截断与提前终止，结果与 tile 内高斯的顺序无关。
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_amp,
	const float amp_cutoff,
	const float background,
	float* __restrict__ out_color)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// 越界线程只帮忙搬运数据，不累加
	const bool inside = pix.x < W && pix.y < H;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_amp[BLOCK_SIZE];

	float C[CHANNELS] = { 0 };

	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Collectively fetch per-Gaussian data from global to shared
		block.sync();
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_amp[block.thread_rank()] = conic_amp[coll_id];
		}
		block.sync();

		for (int j = 0; inside && j < min(BLOCK_SIZE, toDo); j++)
		{
			const float2 xy = collected_xy[j];
			const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			const float4 con_a = collected_conic_amp[j];
			const float power = -0.5f * (con_a.x * d.x * d.x + con_a.z * d.y * d.y) - con_a.y * d.x * d.y;
			// 截断支撑：马氏距离 ≤ k_max 且 w ≥ ε（反向使用完全相同的判据）
			if (power > 0.0f || power < -MAX_POWER_CUTOFF)
				continue;
			const float w = con_a.w * expf(power);
			if (w < amp_cutoff)
				continue;

			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * w;
		}
	}

	if (inside)
	{
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + background;
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* intensities,
	const float4* conic_amp,
	const float amp_cutoff,
	const float background,
	float* out_color)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		intensities,
		conic_amp,
		amp_cutoff,
		background,
		out_color);
}

void FORWARD::preprocess(int P,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* cov3D_precomp,
	const float* slice_pose,
	const int W, int H,
	const float pixel_du, const float pixel_dv,
	const float slice_sigma,
	const float amp_cutoff,
	int* radii,
	float2* means2D,
	float* cov3Ds,
	float4* conic_amp,
	float* amplitudes,
	const dim3 grid,
	uint32_t* tiles_touched,
	uint4* tile_rects)
{
	preprocessCUDA << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		scales,
		scale_modifier,
		rotations,
		cov3D_precomp,
		slice_pose,
		W, H,
		pixel_du, pixel_dv,
		slice_sigma,
		amp_cutoff,
		radii,
		means2D,
		cov3Ds,
		conic_amp,
		amplitudes,
		grid,
		tiles_touched,
		tile_rects
		);
}
