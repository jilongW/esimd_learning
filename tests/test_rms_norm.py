import sys
import time
from pathlib import Path

import torch

device = torch.device("xpu")
HIDDEN_SHAPES = (
    (1, 512),
    (1, 2048),
    (1, 2560),
    (1, 5120),
    (128, 512),
    (128, 2048),
    (128, 2560),
    (128, 5120),
)


def select_vl_ks(rows: int, hidden_size: int) -> tuple[int, int]:
    if hidden_size <= 512:
        vl, ks = 256, 1
    elif hidden_size <= 2048:
        vl, ks = (512, 2) if rows >= 64 else (512, 1)
    elif hidden_size <= 2560:
        vl, ks = (256, 5) if rows >= 64 else (512, 2)
    else:
        vl, ks = (256, 5) if rows >= 64 else (512, 8)

    while hidden_size % vl != 0 and vl > 128:
        vl //= 2

    while vl * ks >= hidden_size:
        if ks == 10:
            ks = 8
        elif ks == 8:
            ks = 5
        elif ks == 5:
            ks = 2
        else:
            ks = 1
            break

    return vl, ks


def ref_rms_norm(hidden_states, weight, eps):
    hidden = hidden_states.cpu().float()
    weight_f = weight.cpu().float()
    variance = hidden.pow(2).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(variance + eps)
    return hidden * inv_rms * weight_f


def test_rms_norm_correctness():
    from custom_esimd_kernels_vllm import esimd_rms_norm

    torch.manual_seed(42)
    eps = 1e-6

    for dtype in (torch.float16, torch.bfloat16):
        for rows, hidden_size in HIDDEN_SHAPES:
            hidden = torch.randn(rows, hidden_size, dtype=dtype, device=device)
            weight = torch.randn(hidden_size, dtype=dtype, device=device) * 0.1
            out = torch.empty_like(hidden)

            esimd_rms_norm(hidden, weight, eps, out)
            torch.xpu.synchronize()

            ref = ref_rms_norm(hidden, weight, eps)
            diff = (out.cpu().float() - ref).abs()
            assert diff.max().item() < 0.1, (
                f"dtype={dtype}, shape={tuple(hidden.shape)}, diff={diff.max().item():.4f}"
            )


def _effective_rms_norm_bytes(hidden_states: torch.Tensor, weight: torch.Tensor) -> int:
    # Compare on the same algorithmic workload: two reads of x, one read of weight, one write of output.
    return (
        2 * hidden_states.numel() * hidden_states.element_size()
        + weight.numel() * weight.element_size()
        + hidden_states.numel() * hidden_states.element_size()
    )


def _benchmark_one(fn, iters: int) -> float:
    for _ in range(20):
        fn()
    torch.xpu.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) * 1e6 / iters


def _benchmark_iters(hidden_shape: tuple[int, int]) -> int:
    _, hidden_size = hidden_shape
    if hidden_size <= 512:
        return 4000
    if hidden_size <= 2560:
        return 2000
    return 1000


def _rms_norm_flops(hidden_states: torch.Tensor) -> int:
    return 4 * hidden_states.numel()


def benchmark_rms_norm():
    from custom_esimd_kernels_vllm import esimd_rms_norm

    if not hasattr(torch.ops, "_C") or not hasattr(torch.ops._C, "rms_norm"):
        vllm_repo = Path(__file__).resolve().parents[2] / "applications.ai.gpu.vllm-xpu"
        if vllm_repo.exists():
            sys.path.insert(0, str(vllm_repo))
            from vllm.platforms import current_platform

            current_platform.import_kernels()

    if not hasattr(torch.ops, "_C") or not hasattr(torch.ops._C, "rms_norm"):
        raise RuntimeError("torch.ops._C.rms_norm is not available in this environment")

    eps = 1e-6
    print("\n--- RMSNorm Benchmark ---")
    print(
        f"\n{'Case':<30} {'Config':>20} | {'Indiv us':>10} {'vllm us':>10} {'Indiv TF':>10} {'vllm TF':>10} {'Indiv GB/s':>12} {'vllm GB/s':>11} {'Speedup':>8}"
    )
    print("-" * 136)

    for rows, hidden_size in HIDDEN_SHAPES:
        iters = _benchmark_iters((rows, hidden_size))
        for dtype in (torch.float16, torch.bfloat16):
            hidden = torch.randn(rows, hidden_size, dtype=dtype, device=device)
            weight = torch.randn(hidden_size, dtype=dtype, device=device) * 0.1
            out_torch = torch.empty_like(hidden)
            out_esimd = torch.empty_like(hidden)
            heuristic_vl, heuristic_ks = select_vl_ks(rows, hidden_size)
            total_bytes = _effective_rms_norm_bytes(hidden, weight)
            total_flops = _rms_norm_flops(hidden)

            esimd_us = _benchmark_one(
                lambda: esimd_rms_norm(hidden, weight, eps, out_esimd),
                iters,
            )
            esimd_bw = total_bytes / (esimd_us * 1e-6) / 1e9
            esimd_tflops = total_flops / (esimd_us * 1e6)

            torch_us = _benchmark_one(
                lambda: torch.ops._C.rms_norm(out_torch, hidden, weight, eps),
                iters,
            )
            torch_bw = total_bytes / (torch_us * 1e-6) / 1e9
            torch_tflops = total_flops / (torch_us * 1e6)

            case_name = f"rms_norm {str(dtype).split('.')[-1]}"
            cfg_str = f"shape=({rows},{hidden_size}) auto={heuristic_vl}:{heuristic_ks}"
            speedup = torch_us / esimd_us if esimd_us > 0 else 0.0
            print(
                f"{case_name:<30} {cfg_str:>20} | {esimd_us:>9.2f} {torch_us:>9.2f} "
                f"{esimd_tflops:>9.4f} {torch_tflops:>9.4f} {esimd_bw:>11.2f} {torch_bw:>11.2f} {speedup:>7.2f}x"
            )


if __name__ == "__main__":
    test_rms_norm_correctness()
    benchmark_rms_norm()