"""Python wrappers for the retained ESIMD GEMV kernel."""

import torch

_ops = torch.ops.custom_esimd_kernels_vllm
_GELU_TANH_AND_MUL_VL_CANDIDATES = (128, 256, 512)
_GELU_TANH_AND_MUL_KS_CANDIDATES = (1, 2, 4, 5, 8, 10, 16, 20, 32, 40, 64)


def _gelu_tanh_and_mul_step_down_ks(value: int) -> int:
    if value == 64:
        return 40
    if value == 40:
        return 32
    if value == 32:
        return 20
    if value == 20:
        return 16
    if value == 16:
        return 10
    if value == 10:
        return 8
    if value == 8:
        return 5
    if value == 5:
        return 4
    if value == 4:
        return 2
    if value == 2:
        return 1
    return value

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


def esimd_norm_gemv_fp8_pert(
    hidden_states: torch.Tensor,
    norm_weight: torch.Tensor,
    gemv_weight: torch.Tensor,
    gemv_scale: torch.Tensor,
    output: torch.Tensor,
    eps: float,
    vl: int | None = None,
    ks: int | None = None,
) -> torch.Tensor:
    """Fused RMSNorm + FP8 GEMV with per-tensor scale.

    hidden_states: [1, K] fp16.
    norm_weight: [K] fp16.
    gemv_weight: [N, K] fp8.
    gemv_scale: scalar fp32.
    output: [1, N] or [N] fp16.
    """
    if (vl is None) != (ks is None):
        raise ValueError("vl and ks must both be provided or both be omitted")
    if vl is None and ks is None:
        vl, ks = 0, 0
    return _ops.esimd_norm_gemv_fp8_pert(
        hidden_states,
        norm_weight,
        gemv_weight,
        gemv_scale,
        output,
        eps,
        vl,
        ks,
    )


def esimd_norm_gemv2_fp8_pert(
    hidden_states: torch.Tensor,
    norm_weight: torch.Tensor,
    gemv_weight0: torch.Tensor,
    gemv_scale0: torch.Tensor,
    gemv_weight1: torch.Tensor,
    gemv_scale1: torch.Tensor,
    eps: float,
    vl: int,
    ks: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused RMSNorm + 2-matrix FP8 GEMV with per-tensor scales.

    hidden_states: [1, K] fp16.
    norm_weight: [K] fp16.
    gemv_weight0/gemv_weight1: [N0, K]/[N1, K] fp8.
    gemv_scale0/gemv_scale1: scalar fp32.
    returns: ([1, N0], [1, N1]) fp16.
    """
    return _ops.esimd_norm_gemv2_fp8_pert(
        hidden_states,
        norm_weight,
        gemv_weight0,
        gemv_scale0,
        gemv_weight1,
        gemv_scale1,
        eps,
        vl,
        ks,
    )


def _valid_gelu_tanh_and_mul_vl_ks(cols: int) -> list[tuple[int, int]]:
    half_cols = cols // 2
    valid = []
    for vl in _GELU_TANH_AND_MUL_VL_CANDIDATES:
        if half_cols % vl != 0:
            continue
        chunks_per_row = half_cols // vl
        for ks in _GELU_TANH_AND_MUL_KS_CANDIDATES:
            if 1 <= ks <= min(chunks_per_row, 64):
                valid.append((vl, ks))
    return valid


def _normalize_gelu_tanh_and_mul_vl_ks(cols: int, vl: int, ks: int) -> tuple[int, int]:
    half_cols = cols // 2
    while vl > 128 and (half_cols % vl != 0 or vl > half_cols):
        vl //= 2

    if half_cols % vl != 0 or vl > half_cols:
        raise ValueError(f"no valid vl for input width {cols}")

    chunks_per_row = half_cols // vl
    max_ks = min(chunks_per_row, 64)
    while ks > max_ks:
        next_ks = _gelu_tanh_and_mul_step_down_ks(ks)
        if next_ks == ks:
            break
        ks = next_ks
    return vl, ks


def select_gelu_tanh_and_mul_vl_ks(
    input: torch.Tensor,
) -> tuple[int, int]:
    """Select a default vl/ks pair for GeGLU activation.

    The selector is intentionally simple and tuned for Gemma4-like shapes.
    It favors higher ks for small-token batches and reduces ks as rows grow.
    """
    if input.dim() != 2:
        raise ValueError("esimd_gelu_tanh_and_mul selector expects a 2D [M, 2D] tensor")
    cols = int(input.shape[1])
    valid = _valid_gelu_tanh_and_mul_vl_ks(cols)
    if not valid:
        raise ValueError(f"no valid vl/ks for input width {cols}")

    rows = int(input.shape[0])
    if rows <= 1:
        vl, ks = 256, 20
    elif rows <= 2:
        vl, ks = (256, 40) if input.dtype == torch.bfloat16 else (128, 32)
    elif rows <= 8:
        vl, ks = 128, 40
    elif rows <= 16:
        vl, ks = (256, 20) if input.dtype == torch.bfloat16 else (128, 40)
    elif rows <= 24:
        vl, ks = (128, 40) if input.dtype == torch.bfloat16 else (256, 20)
    else:
        vl, ks = (256, 16) if input.dtype == torch.bfloat16 else (256, 20)

    normalized = _normalize_gelu_tanh_and_mul_vl_ks(cols, vl, ks)
    if normalized in valid:
        return normalized
    return valid[0]


def esimd_gelu_tanh_and_mul(
    input: torch.Tensor,
    output: torch.Tensor,
    vl: int | None = None,
    ks: int | None = None,
) -> torch.Tensor:
    """GeGLU activation with GELU(tanh approximation) on the gate half.

    input: [M, 2D] fp16 or bf16.
    output: [M, D] with the same dtype/device.
    """
    if (vl is None) != (ks is None):
        raise ValueError("vl and ks must both be provided or both be omitted")
    if vl is None and ks is None:
        vl, ks = 0, 0
    return _ops.esimd_gelu_tanh_and_mul(input, output, vl, ks)


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