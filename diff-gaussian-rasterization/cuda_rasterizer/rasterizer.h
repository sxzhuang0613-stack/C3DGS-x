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

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>

namespace CudaRasterizer
{
	// MRI 切片加和渲染器。slice_pose 为设备端 float[12]：R（行优先，行依次为 u、v、n）+ 原点 o。
	class Rasterizer
	{
	public:

		static int forward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P,
			const float background,
			const int width, int height,
			const float* means3D,
			const float* intensities,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* slice_pose,
			const float pixel_du, const float pixel_dv,
			const float slice_sigma,
			const float amp_cutoff,
			float* out_color,
			int* radii,
			float* amplitudes,
			bool debug = false);

		static void backward(
			const int P, int R,
			const int width, int height,
			const float* means3D,
			const float* intensities,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* slice_pose,
			const float pixel_du, const float pixel_dv,
			const float slice_sigma,
			const float amp_cutoff,
			const int* radii,
			char* geom_buffer,
			char* binning_buffer,
			char* image_buffer,
			const float* dL_dpix,
			float* dL_dmean2D,
			float* dL_dconic,
			float* dL_damp,
			float* dL_dintensity,
			float* dL_dmean3D,
			float* dL_dcov3D,
			float* dL_dscale,
			float* dL_drot,
			bool debug);
	};
};

#endif
