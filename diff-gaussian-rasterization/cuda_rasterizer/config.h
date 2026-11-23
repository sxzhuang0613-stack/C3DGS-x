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

#ifndef CUDA_RASTERIZER_CONFIG_H_INCLUDED
#define CUDA_RASTERIZER_CONFIG_H_INCLUDED

// MRI 信号是与视角无关的标量强度，只有一个通道
#define NUM_CHANNELS 1
#define BLOCK_X 16
#define BLOCK_Y 16

// 足迹截断的最大马氏距离 k_max=3，即 power >= -k_max^2/2；与经典 3DGS 的 3σ 一致
#define MAX_POWER_CUTOFF 4.5f

#endif
