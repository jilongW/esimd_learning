"""Python wrappers for the retained ESIMD GEMV kernel."""

import torch

_ops = torch.ops.custom_esimd_kernels_vllm

def esimd_gemv_fp8_pert(
    input: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    """FP8 weight GEMV with per-tensor scale (fp32 scalar).

    input: [1, K] fp16, weight: [N, K] fp8_e4m3, scale: fp32 scalar, output: [1, N] fp16.
    N and K are inferred from weight shape.
    """
    return _ops.esimd_gemv_fp8_pert(input, weight, weight_scale, output)