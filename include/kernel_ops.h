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
    int64_t N, int64_t K, int64_t vl, int64_t ks);

at::Tensor esimd_gemv_fp8(
    at::Tensor input, at::Tensor weight, at::Tensor weight_scale,
    at::Tensor output);

// FP8 GEMV with per-tensor scale: scale is fp32 scalar, N/K inferred from weight.
at::Tensor esimd_gemv_fp8_pert(
    at::Tensor input, at::Tensor weight, at::Tensor weight_scale,
    at::Tensor output,
    int64_t N, int64_t K, int64_t vl, int64_t ks);

at::Tensor esimd_fused_add_rms_norm_batched(
    at::Tensor hidden_states, at::Tensor residual,
    at::Tensor weight, double eps);

at::Tensor esimd_rms_norm(
    at::Tensor hidden_states, at::Tensor weight,
    double eps, at::Tensor output, int64_t vl, int64_t ks);

at::Tensor esimd_norm_gemv_fp8_pert(
    at::Tensor hidden_states, at::Tensor norm_weight,
    at::Tensor gemv_weight, at::Tensor gemv_scale,
    at::Tensor output, double eps, int64_t vl, int64_t ks);

at::Tensor esimd_gelu_tanh_and_mul(
    at::Tensor input,
    at::Tensor output,
    int64_t vl,
    int64_t ks);

std::tuple<at::Tensor, at::Tensor> esimd_norm_gemv2_fp8_pert(
    at::Tensor hidden_states,
    at::Tensor norm_weight,
    at::Tensor gemv_weight0,
    at::Tensor gemv_scale0,
    at::Tensor gemv_weight1,
    at::Tensor gemv_scale1,
    double eps,
    int64_t vl,
    int64_t ks);

// FP8 GEMM per-tensor scale: input/output [M, K]/[M, N] fp16 or bf16 (matching dtype),
// weight [N, K] fp8.
// Auto-dispatches: M<=3 → batched GEMV, M>=2 E4M3 → DPAS V9, else → WS
at::Tensor esimd_gemm_fp8_pert(
    at::Tensor input, at::Tensor weight, at::Tensor weight_scale,
    at::Tensor output);

// Fused ResidualAdd + RMSNorm + FP8 GEMV
at::Tensor esimd_resadd_norm_gemv_fp8_pert(
    at::Tensor hidden_states, at::Tensor residual, at::Tensor norm_weight,
    at::Tensor gemv_weight, at::Tensor gemv_scale, at::Tensor output, at::Tensor normed_out,
    double eps);