"""Python wrappers for the retained ESIMD GEMV kernel."""

import torch

_ops = torch.ops.custom_esimd_kernels_vllm

def esimd_gemv_fp8_pern(
    input: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor,
    output: torch.Tensor,
    N: int, K: int,
    vl: int = 0, ks: int = 0,
) -> torch.Tensor:
    """FP8 weight GEMV with per-N scale, FP32 accumulation, deferred scale.

    input: [1, K] fp16, weight: [N, K] fp8_e4m3, scale: [N] fp16, output: [1, N] fp16.
    K must be 256-aligned. N must be 8-aligned.
    """
    return _ops.esimd_gemv_fp8_pern(input, weight, weight_scale, output, N, K, vl, ks)

# ---- Per-tensor scale variants (N/K auto-detected from weight shape) ----

def esimd_gemv_fp8_pert(
    input: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    """FP8 weight GEMV with per-tensor scale (fp32 scalar).

    input: [1, K] fp16, weight: [N, K] fp8_e4m3, scale: fp32 scalar, output: [1, N] fp16.
    N and K are inferred from weight shape.
    """
    return _ops.esimd_gemv_fp8_pert(input, weight, weight_scale, output)