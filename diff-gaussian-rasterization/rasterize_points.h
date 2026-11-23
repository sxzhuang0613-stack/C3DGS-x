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

#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>

// 前向返回 (num_rendered, 图像[1,H,W], radii[P], 幅值[P], geomBuffer, binningBuffer, imgBuffer)
std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeSliceCUDA(
	const torch::Tensor& means3D,
	const torch::Tensor& intensities,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& slice_pose,
	const float pixel_du,
	const float pixel_dv,
	const float slice_sigma,
	const float amp_cutoff,
	const float background,
	const int image_height,
	const int image_width,
	const bool debug);

// 反向返回 (dL_dmeans2D, dL_dintensities, dL_dmeans3D, dL_dcov3D, dL_dscales, dL_drotations)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeSliceBackwardCUDA(
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
	const torch::Tensor& intensities,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& slice_pose,
	const float pixel_du,
	const float pixel_dv,
	const float slice_sigma,
	const float amp_cutoff,
	const torch::Tensor& dL_dout_color,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool debug);
