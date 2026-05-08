#pragma once

#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <torch/library.h>
#include <torch/torch.h>

// FP8 GEMV with per-tensor scale: scale is fp32 scalar, N/K inferred from weight.
at::Tensor esimd_gemv_fp8_pert(
    at::Tensor input, at::Tensor weight, at::Tensor weight_scale,
    at::Tensor output);