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

#ifndef CUDA_RASTERIZER_BACKWARD_H_INCLUDED
#define CUDA_RASTERIZER_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace BACKWARD
{
	// 加和渲染的反向：像素梯度 → 每个高斯的强度、幅值、像素中心与 conic 梯度
	void render(
		const dim3 grid, dim3 block,
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
		float* dL_dintensities);

	// 预处理的反向：(像素中心, conic, 幅值) → (μ, Σ) → (尺度, 旋转)
	void preprocess(
		int P,
		const float3* means,
		const int* radii,
		const glm::vec3* scales,
		const glm::vec4* rotations,
		const float scale_modifier,
		const float* cov3Ds,
		const float* slice_pose,
		const float pixel_du, const float pixel_dv,
		const float slice_sigma,
		const float3* dL_dmean2D,
		const float3* dL_dconics,
		const float* dL_damp,
		glm::vec3* dL_dmeans,
		float* dL_dcov3D,
		glm::vec3* dL_dscale,
		glm::vec4* dL_drot);
}

#endif
