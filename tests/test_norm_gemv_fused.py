import time

import torch

from custom_esimd_kernels_vllm import (
    esimd_gemv_fp8,
    esimd_norm_gemv_fp8_pert,
    esimd_rms_norm,
)

DEVICE = torch.device("xpu")
EPS = 1e-6
WARMUP_ITERS = 10
BENCHMARK_ITERS = 1000
TARGET_BW = 112.0  # GB/s PTL
SHAPES = [
    ("gate_up_proj", 20480, 2560),
]

def _select_rms_norm_vl_ks(rows: int, hidden_size: int) -> tuple[int, int]:
    if hidden_size <= 256:
        vl, ks = 256, 1
    elif hidden_size <= 512:
        vl, ks = (256, 1) if rows >= 64 else (128, 1)
    elif hidden_size <= 2048:
        vl, ks = (512, 2) if rows >= 64 else (1024, 1)
    elif hidden_size <= 2560:
        vl, ks = (256, 5) if rows >= 64 else (256, 8)
    else:
        vl, ks = (256, 5) if rows >= 64 else (256, 8)

    while hidden_size % vl != 0 and vl > 128:
        if vl == 1024:
            vl = 512
        else:
            vl //= 2

    while vl * ks >= hidden_size and not (hidden_size == 256 and vl == 256 and ks == 1):
        if ks == 10:
            ks = 8
        elif ks == 8:
            ks = 5
        elif ks == 5:
            ks = 2
        elif ks == 2:
            ks = 1
        elif vl > 128:
            vl //= 2
            ks = 1
        else:
            raise ValueError(f"No valid vl/ks for hidden_size={hidden_size}")

    return vl, ks


def _ref_norm_gemv_fp8(hidden, norm_weight, weight_fp8, scale, eps):
    hidden_f = hidden.cpu().float()
    norm_weight_f = norm_weight.cpu().float()
    variance = hidden_f.pow(2).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(variance + eps)
    normed = hidden_f * inv_rms * norm_weight_f

    scale_f = float(scale.cpu().float().reshape(-1)[0].item())
    dequant_weight = weight_fp8.cpu().float() * scale_f
    output = normed @ dequant_weight.T
    return output, normed.half()


def _benchmark_xpu_callable(fn, warmup_iters=WARMUP_ITERS, benchmark_iters=BENCHMARK_ITERS):
    for _ in range(warmup_iters):
        fn()

    torch.xpu.synchronize()
    start = torch.xpu.Event(enable_timing=True)
    end = torch.xpu.Event(enable_timing=True)
    start.record()
    for _ in range(benchmark_iters):
        fn()
    end.record()
    torch.xpu.synchronize()
    return start.elapsed_time(end) * 1000.0 / benchmark_iters


def _norm_gemv_fused_bytes(n_size: int, k_size: int) -> int:
    hidden_bytes = k_size * 2
    norm_weight_bytes = k_size * 2
    gemv_weight_bytes = n_size * k_size
    scale_bytes = 4
    output_bytes = n_size * 2
    return hidden_bytes * 2 + norm_weight_bytes + gemv_weight_bytes + scale_bytes + output_bytes


def _norm_gemv_split_bytes(n_size: int, k_size: int) -> int:
    rms_norm_bytes = k_size * 2 * 2 + k_size * 2 + k_size * 2
    gemv_bytes = k_size * 2 + n_size * k_size + 4 + n_size * 2
    return rms_norm_bytes + gemv_bytes


def test_norm_gemv_fp8_pert_correctness():
    torch.manual_seed(42)

    for name, n_size, k_size in SHAPES:
        hidden = torch.randn(1, k_size, dtype=torch.float16, device=DEVICE) * 0.1
        norm_weight = torch.randn(k_size, dtype=torch.float16, device=DEVICE) * 0.1
        weight_fp16 = torch.randn(n_size, k_size, dtype=torch.float16, device=DEVICE) * 0.1
        weight_fp8 = weight_fp16.to(torch.float8_e4m3fn)
        scale = torch.tensor([0.0008], dtype=torch.float32, device=DEVICE)
        output = torch.empty(1, n_size, dtype=torch.float16, device=DEVICE)

        esimd_norm_gemv_fp8_pert(
            hidden,
            norm_weight,
            weight_fp8,
            scale,
            output,
            EPS,
        )
        torch.xpu.synchronize()

        ref_output, _ = _ref_norm_gemv_fp8(
            hidden,
            norm_weight,
            weight_fp8,
            scale,
            EPS,
        )
        out_diff = (output.cpu().float() - ref_output).abs()
        rel_err = out_diff.mean().item() / (ref_output.abs().mean().item() + 1e-6)

        assert rel_err < 0.3, f"{name} relative error too large: {rel_err:.4f}"


def benchmark_norm_gemv_fp8_pert_vs_split():
    torch.manual_seed(42)

    print(
        f"\n{'Shape':<20} {'N':>6} {'K':>6} | {'Mode':>6} {'GB/s':>8} {'BW%':>7} {'us':>8}"
    )
    print("-" * 74)

    for name, n_size, k_size in SHAPES:
        hidden = torch.randn(1, k_size, dtype=torch.float16, device=DEVICE) * 0.1
        norm_weight = torch.randn(k_size, dtype=torch.float16, device=DEVICE) * 0.1
        weight_fp16 = torch.randn(n_size, k_size, dtype=torch.float16, device=DEVICE) * 0.1
        weight_fp8 = weight_fp16.to(torch.float8_e4m3fn)
        scale = torch.tensor([0.0008], dtype=torch.float32, device=DEVICE)

        weight_bytes = n_size * k_size
        target_mem = 32 * 1024 * 1024
        num_copies = max(16, target_mem // max(weight_bytes, 1))
        num_copies = min(num_copies, 512)

        weights = [weight_fp8]
        for _ in range(1, num_copies):
            extra_weight = torch.randn(n_size, k_size, dtype=torch.float16, device=DEVICE) * 0.1
            weights.append(extra_weight.to(torch.float8_e4m3fn))

        fused_output = torch.empty(1, n_size, dtype=torch.float16, device=DEVICE)
        split_normed = torch.empty_like(hidden)
        split_output = torch.empty(1, n_size, dtype=torch.float16, device=DEVICE)

        rows = hidden.numel() // hidden.shape[-1]
        vl, ks = _select_rms_norm_vl_ks(rows, k_size)

        run_state = {"index": 0}

        def run_fused():
            current_weight = weights[run_state["index"] % num_copies]
            esimd_norm_gemv_fp8_pert(
                hidden,
                norm_weight,
                current_weight,
                scale,
                fused_output,
                EPS,
            )
            run_state["index"] += 1

        def run_split():
            current_weight = weights[run_state["index"] % num_copies]
            esimd_rms_norm(hidden, norm_weight, EPS, split_normed, vl, ks)
            esimd_gemv_fp8(split_normed, current_weight, scale, split_output)
            run_state["index"] += 1

        run_state["index"] = 0
        fused_latency_us = _benchmark_xpu_callable(run_fused)
        run_state["index"] = 0
        split_latency_us = _benchmark_xpu_callable(run_split)

        fused_bw = (_norm_gemv_fused_bytes(n_size, k_size) / 1e9) / (fused_latency_us / 1e6)
        split_bw = (_norm_gemv_split_bytes(n_size, k_size) / 1e9) / (split_latency_us / 1e6)

        print(f"{name:<20} {n_size:>6} {k_size:>6} | {'fused':>6} {fused_bw:>7.1f} {fused_bw / TARGET_BW * 100:>6.1f}% {fused_latency_us:>7.2f}")
        print(f"{name:<20} {n_size:>6} {k_size:>6} | {'split':>6} {split_bw:>7.1f} {split_bw / TARGET_BW * 100:>6.1f}% {split_latency_us:>7.2f}")


if __name__ == "__main__":
    test_norm_gemv_fp8_pert_correctness()
    benchmark_norm_gemv_fp8_pert_vs_split()