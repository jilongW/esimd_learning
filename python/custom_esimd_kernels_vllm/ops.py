"""Python wrappers for the retained ESIMD GEMV kernel."""

import torch

_ops = torch.ops.custom_esimd_kernels_vllm

def esimd_gemv_fp8(
    input: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    """Unified FP8 GEMV with automatic scale-shape dispatch.

    input/output: [1, K]/[1, N] fp16 or bf16 with matching dtype,
    weight: [N, K] fp8, scale: scalar fp32 or [N] fp16/bf16.
    """
    return _ops.esimd_gemv_fp8(input, weight, weight_scale, output)

def esimd_gemv_fp8_pern(
    input: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor,
    output: torch.Tensor,
    N: int, K: int,
    vl: int, ks: int,
) -> torch.Tensor:
    """FP8 weight GEMV with per-N scale, FP32 accumulation, deferred scale.

    input/output: [1, K]/[1, N] fp16 or bf16 with matching dtype,
    weight: [N, K] fp8_e4m3, scale: [N] fp16.
    K must be divisible by both ks and vl, and (K // ks) must be divisible by vl.
    """
    return _ops.esimd_gemv_fp8_pern(input, weight, weight_scale, output, N, K, vl, ks)

# ---- Per-tensor scale variants (N/K auto-detected from weight shape) ----

def esimd_gemv_fp8_pert(
    input: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor,
    output: torch.Tensor,
    N: int, K: int,
    vl: int, ks: int,
) -> torch.Tensor:
    """FP8 weight GEMV with per-tensor scale (fp32 scalar).

    input/output: [1, K]/[1, N] fp16 or bf16 with matching dtype,
    weight: [N, K] fp8_e4m3, scale: fp32 scalar.
    K must be divisible by both ks and vl, and (K // ks) must be divisible by vl.
    """
    return _ops.esimd_gemv_fp8_pert(input, weight, weight_scale, output, N, K, vl, ks)

def esimd_fused_add_rms_norm_batched(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Batched fused residual add + RMSNorm (Gemma-style).

    residual[i] = hidden_states[i] + residual[i]  (in-place)
    hidden_states[i] = rmsnorm(residual[i]) * weight  (output)
    weight must be pre-adjusted (w+1.0). Works for any number of rows.
    """
    return _ops.esimd_fused_add_rms_norm_batched(hidden_states, residual, weight, eps)


def esimd_rms_norm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    output: torch.Tensor,
    vl: int,
    ks: int,
) -> torch.Tensor:
    """Batched RMSNorm.

    hidden_states: [..., K] fp16 or bf16, where K is a multiple of 128.
    weight: [K] with the same dtype as hidden_states.
    output: preallocated output tensor with the same shape and dtype as hidden_states.
    vl: vector length candidate, one of 128, 256, 512, 1024.
    ks: per-row thread split, one of 1, 2, 5, 8, 10.
    """
    return _ops.esimd_rms_norm(hidden_states, weight, eps, output, vl, ks)


def esimd_gemm_fp8_pert(
    input: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    """FP8 GEMM with per-tensor scale — handles any M (auto-dispatches).

    input/output: [M, K]/[M, N] fp16 or bf16 with matching dtype,
    weight: [N, K] fp8, scale: fp32 scalar.
    N and K are inferred from weight shape. M from input shape.

    Auto-dispatch:
      M=1-3  → batched GEMV (BW-bound, K-split SLM reduction)
      M>=2   → DPAS V9 (E4M3, K%64==0) or DPAS V7 (E5M2) or WS fallback
    """
    return _ops.esimd_gemm_fp8_pert(
        input,
        weight,
        weight_scale,
        output,
    )

def esimd_resadd_norm_gemv_fp8_pert(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    gemv_weight: torch.Tensor,
    gemv_scale: torch.Tensor,
    output: torch.Tensor,
    normed_out: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Fused ResidualAdd + RMSNorm + FP8 GEMV.
    
    """
    return _ops.esimd_resadd_norm_gemv_fp8_pert(
        hidden_states, residual, norm_weight,
        gemv_weight, gemv_scale, output, normed_out, eps)