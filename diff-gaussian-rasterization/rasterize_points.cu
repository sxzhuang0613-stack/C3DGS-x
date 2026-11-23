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

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include <fstream>
#include <string>
#include <functional>

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

// 空张量表示“未提供”，传 nullptr 给 CUDA 端
static float* optionalPtr(const torch::Tensor& t)
{
	return t.numel() == 0 ? nullptr : t.data_ptr<float>();
}

static void checkInput(const torch::Tensor& t, const char* name, int64_t P, int64_t cols)
{
	if (t.numel() == 0)
		return;
	TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kFloat32 && t.is_contiguous(),
		name, " 必须是连续的 CUDA float32 张量");
	TORCH_CHECK(t.dim() == 2 && t.size(0) == P && t.size(1) == cols,
		name, " 形状应为 (", P, ", ", cols, ")");
}

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
	const bool debug)
{
	TORCH_CHECK(means3D.dim() == 2 && means3D.size(1) == 3, "means3D 形状应为 (P, 3)");
	const int P = means3D.size(0);
	const int H = image_height;
	const int W = image_width;
	checkInput(means3D, "means3D", P, 3);
	TORCH_CHECK(intensities.numel() == (int64_t)P * NUM_CHANNELS, "intensities 形状应为 (P, 1)");
	checkInput(intensities, "intensities", P, NUM_CHANNELS);
	checkInput(scales, "scales", P, 3);
	checkInput(rotations, "rotations", P, 4);
	checkInput(cov3D_precomp, "cov3D_precomp", P, 6);
	TORCH_CHECK(slice_pose.is_cuda() && slice_pose.scalar_type() == torch::kFloat32
		&& slice_pose.is_contiguous() && slice_pose.numel() == 12, "slice_pose 应为 12 个元素的连续 CUDA float32 张量");

	auto float_opts = means3D.options().dtype(torch::kFloat32);
	torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, background, float_opts);
	torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
	torch::Tensor amplitudes = torch::zeros({P}, float_opts);

	torch::Device device(torch::kCUDA);
	torch::TensorOptions options(torch::kByte);
	torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
	torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
	torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
	std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
	std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
	std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);

	int rendered = 0;
	if (P != 0)
	{
		rendered = CudaRasterizer::Rasterizer::forward(
			geomFunc,
			binningFunc,
			imgFunc,
			P,
			background,
			W, H,
			means3D.data_ptr<float>(),
			intensities.data_ptr<float>(),
			optionalPtr(scales),
			scale_modifier,
			optionalPtr(rotations),
			optionalPtr(cov3D_precomp),
			slice_pose.data_ptr<float>(),
			pixel_du, pixel_dv,
			slice_sigma,
			amp_cutoff,
			out_color.data_ptr<float>(),
			radii.data_ptr<int>(),
			amplitudes.data_ptr<float>(),
			debug);
	}
	return std::make_tuple(rendered, out_color, radii, amplitudes, geomBuffer, binningBuffer, imgBuffer);
}

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
	const bool debug)
{
	const int P = means3D.size(0);
	const int H = dL_dout_color.size(1);
	const int W = dL_dout_color.size(2);

	torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
	// 与经典库一致：means2D 梯度为 (P, 3)，只有前两列（像素单位）有意义
	torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
	torch::Tensor dL_dintensities = torch::zeros({P, NUM_CHANNELS}, means3D.options());
	torch::Tensor dL_dconic = torch::zeros({P, 3}, means3D.options());
	torch::Tensor dL_damp = torch::zeros({P}, means3D.options());
	torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
	torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
	torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());

	if (P != 0)
	{
		const torch::Tensor dL_dpix = dL_dout_color.contiguous();
		CudaRasterizer::Rasterizer::backward(P, R,
			W, H,
			means3D.data_ptr<float>(),
			intensities.data_ptr<float>(),
			optionalPtr(scales),
			scale_modifier,
			optionalPtr(rotations),
			optionalPtr(cov3D_precomp),
			slice_pose.data_ptr<float>(),
			pixel_du, pixel_dv,
			slice_sigma,
			amp_cutoff,
			radii.data_ptr<int>(),
			reinterpret_cast<char*>(geomBuffer.data_ptr()),
			reinterpret_cast<char*>(binningBuffer.data_ptr()),
			reinterpret_cast<char*>(imageBuffer.data_ptr()),
			dL_dpix.data_ptr<float>(),
			dL_dmeans2D.data_ptr<float>(),
			dL_dconic.data_ptr<float>(),
			dL_damp.data_ptr<float>(),
			dL_dintensities.data_ptr<float>(),
			dL_dmeans3D.data_ptr<float>(),
			dL_dcov3D.data_ptr<float>(),
			dL_dscales.data_ptr<float>(),
			dL_drotations.data_ptr<float>(),
			debug);
	}

	return std::make_tuple(dL_dmeans2D, dL_dintensities, dL_dmeans3D, dL_dcov3D, dL_dscales, dL_drotations);
}
