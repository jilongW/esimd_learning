#pragma once

#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <torch/library.h>
#include <torch/torch.h>


// FP8 weight GEMV with per-N scale: output = input @ dequant(weight_fp8) * scale
// FP32 accumulation, element-wise acc + deferred scale. Optimized for decode (M=1).
at::Tensor esimd_gemv_fp8_pern(
    at::Tensor input, at::Tensor weight, at::Tensor weight_scale,
    at::Tensor output,
    int64_t N, int64_t K);

// FP8 GEMV with per-tensor scale: scale is fp32 scalar, N/K inferred from weight.
at::Tensor esimd_gemv_fp8_pert(
    at::Tensor input, at::Tensor weight, at::Tensor weight_scale,
    at::Tensor output);

at::Tensor esimd_fused_add_rms_norm_batched(
    at::Tensor hidden_states, at::Tensor residual,
    at::Tensor weight, double eps);

// FP8 GEMM per-tensor scale: input [M, K] fp16, weight [N, K] fp8, output [M, N] fp16
// Auto-dispatches: M<=3 → batched GEMV, M>=2 E4M3 → DPAS V9, else → WS
at::Tensor esimd_gemm_fp8_pert(
    at::Tensor input, at::Tensor weight, at::Tensor weight_scale,
    at::Tensor output);