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

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace FORWARD
{
	// 逐高斯预处理：协方差 → 切片坐标 → 层厚积分足迹 → 覆盖半径与 tile 数
	void preprocess(int P,
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
		uint4* tile_rects);

	// 逐 tile 加和渲染
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float2* points_xy_image,
		const float* intensities,
		const float4* conic_amp,
		const float amp_cutoff,
		const float background,
		float* out_color);
}


#endif
