import time

import torch

from custom_esimd_kernels_vllm import (
    esimd_gemv_gelu_tanh_mul_fp8_pert,
    esimd_gemv_fp8,
    esimd_gelu_tanh_and_mul,
    esimd_norm_gemv2_geglu_fp8_pert,
    esimd_norm_gemv_fp8_pert,
    esimd_rms_norm,
)

DEVICE = torch.device("xpu")
EPS = 1e-6
WARMUP_ITERS = 10
BENCHMARK_ITERS = 500
TARGET_BW = 112.0
SHAPES = [
    (10240, 10240, 2560),
]
NORM_GEMV2_VL_CANDIDATES = (128, 256, 512)
NORM_GEMV2_KS_CANDIDATES = (1, 2, 4, 8, 10)


def _normalize_vl_ks(k_size: int, vl: int, ks: int) -> tuple[int, int]:
    k_per_thread = k_size // ks
    while vl > k_per_thread or k_per_thread % vl != 0:
        if vl > 128:
            vl //= 2
        elif ks == 10:
            ks = 8
        elif ks == 8:
            ks = 4
        elif ks == 4:
            ks = 2
        elif ks == 2:
            ks = 1
        else:
            break
        k_per_thread = k_size // ks
    return vl, ks


def _select_norm_gemv_vl_ks(n_size: int, k_size: int) -> tuple[int, int]:
    if k_size < 256:
        vl, ks = 128, 1
    elif k_size == 256:
        vl, ks = 256, 1
    elif k_size >= 10240:
        vl, ks = 256, 8
    elif k_size >= 4096:
        vl, ks = 256, 4
    elif k_size >= 2560 and n_size >= 10240:
        vl, ks = 256, 1
    elif k_size >= 2560:
        vl, ks = 128, 10
    elif k_size >= 2048:
        vl, ks = 256, 8
    else:
        vl, ks = 256, 1
    return _normalize_vl_ks(k_size, vl, ks)


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
            raise ValueError(f"No valid rms_norm vl/ks for hidden_size={hidden_size}")

    return vl, ks


def _valid_norm_gemv2_configs(k_size: int) -> list[tuple[int, int]]:
    valid: list[tuple[int, int]] = []
    supported_by_kernel = {
        (512, 1), (512, 2),
        (256, 1), (256, 2), (256, 4), (256, 8),
        (128, 1), (128, 2), (128, 4), (128, 8), (128, 10),
    }
    for vl in NORM_GEMV2_VL_CANDIDATES:
        if k_size % vl != 0:
            continue
        for ks in NORM_GEMV2_KS_CANDIDATES:
            if (vl, ks) not in supported_by_kernel:
                continue
            if k_size % ks != 0:
                continue
            if (k_size // ks) % vl != 0:
                continue
            valid.append((vl, ks))
    return valid


def _search_best_norm_gemv2_vl_ks(
    hidden: torch.Tensor,
    norm_weight: torch.Tensor,
    weight0: torch.Tensor,
    scale0: torch.Tensor,
    weight1: torch.Tensor,
    scale1: torch.Tensor,
    eps: float,
) -> tuple[int, int, float]:
    k_size = int(hidden.shape[1])
    candidates = _valid_norm_gemv2_configs(k_size)
    best_vl, best_ks = candidates[0]
    best_latency = float("inf")
    output = torch.empty(1, weight0.shape[0], dtype=torch.float16, device=weight0.device)

    for vl, ks in candidates:
        def run_once() -> None:
            esimd_norm_gemv2_geglu_fp8_pert(
                hidden,
                norm_weight,
                weight0,
                scale0,
                weight1,
                scale1,
                output,
                eps,
                vl,
                ks,
            )

        latency = _benchmark_xpu_callable(run_once, warmup_iters=4, benchmark_iters=40)
        if latency < best_latency:
            best_latency = latency
            best_vl, best_ks = vl, ks

    return best_vl, best_ks, best_latency


def _benchmark_xpu_callable(fn, warmup_iters: int = WARMUP_ITERS, benchmark_iters: int = BENCHMARK_ITERS) -> float:
    for _ in range(warmup_iters):
        fn()

    torch.xpu.synchronize()
    start = time.perf_counter()
    for _ in range(benchmark_iters):
        fn()
    torch.xpu.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed * 1e6 / benchmark_iters


def _norm_gemv2_fused_bytes(n0: int, n1: int, k_size: int) -> int:
    hidden_bytes = k_size * 2
    norm_weight_bytes = k_size * 2
    gemv_weight_bytes = (n0 + n1) * k_size
    scale_bytes = 8
    output_bytes = n0 * 2
    return hidden_bytes + norm_weight_bytes + gemv_weight_bytes + scale_bytes + output_bytes


def _norm_gemv2_combined_bytes(n0: int, n1: int, k_size: int) -> int:
    hidden_bytes = k_size * 2
    norm_weight_bytes = k_size * 2
    gemv_weight_bytes = (n0 + n1) * k_size
    scale_bytes = 8
    output_bytes = n0 * 2
    return hidden_bytes + norm_weight_bytes + gemv_weight_bytes + scale_bytes + output_bytes


def _norm_gemv2_split_bytes(n0: int, n1: int, k_size: int) -> int:
    rms_norm_bytes = k_size * 2 + k_size * 2 + k_size * 2
    gemv0_bytes = k_size * 2 + n0 * k_size + 4 + n0 * 2
    gemv1_bytes = k_size * 2 + n1 * k_size + 4 + n1 * 2
    geglu_bytes = (n0 + n1) * 2 + n0 * 2
    return rms_norm_bytes + gemv0_bytes + gemv1_bytes + geglu_bytes


def test_norm_gemv2_matches_three_paths() -> None:
    torch.manual_seed(42)

    for n0, n1, k_size in SHAPES:
        hidden = torch.randn(1, k_size, dtype=torch.float16, device=DEVICE) * 0.1
        norm_weight = torch.randn(k_size, dtype=torch.float16, device=DEVICE) * 0.1
        weight0_fp16 = torch.randn(n0, k_size, dtype=torch.float16, device=DEVICE) * 0.1
        weight1_fp16 = torch.randn(n1, k_size, dtype=torch.float16, device=DEVICE) * 0.1
        weight0_fp8 = weight0_fp16.to(torch.float8_e4m3fn)
        weight1_fp8 = weight1_fp16.to(torch.float8_e4m3fn)
        scale0 = torch.tensor([0.0008], dtype=torch.float32, device=DEVICE)
        scale1 = torch.tensor([0.0008], dtype=torch.float32, device=DEVICE)

        total_n = n0 + n1
        vl, ks = _select_norm_gemv_vl_ks(total_n, k_size)
        best_vl, best_ks, _ = _search_best_norm_gemv2_vl_ks(
            hidden,
            norm_weight,
            weight0_fp8,
            scale0,
            weight1_fp8,
            scale1,
            EPS,
        )
        fused_output = torch.empty(1, n0, dtype=torch.float16, device=DEVICE)
        esimd_norm_gemv2_geglu_fp8_pert(
            hidden,
            norm_weight,
            weight0_fp8,
            scale0,
            weight1_fp8,
            scale1,
            fused_output,
            EPS,
            best_vl,
            best_ks,
        )

        combined_weight = torch.cat([weight0_fp8, weight1_fp8], dim=0)
        combined_scale = torch.tensor([scale0.item(), scale1.item()], dtype=torch.float32, device=DEVICE)
        combined_output = torch.empty(1, n0, dtype=torch.float16, device=DEVICE)
        combined_normed = torch.empty_like(hidden)
        esimd_rms_norm(hidden, norm_weight, EPS, combined_normed, 256, 1)
        esimd_gemv_gelu_tanh_mul_fp8_pert(
            combined_normed,
            combined_weight,
            combined_scale,
            combined_output,
        )

        rms_vl, rms_ks = _select_rms_norm_vl_ks(hidden.shape[0], k_size)
        normed = torch.empty_like(hidden)
        esimd_rms_norm(hidden, norm_weight, EPS, normed, rms_vl, rms_ks)

        split_out0 = torch.empty(1, n0, dtype=torch.float16, device=DEVICE)
        split_out1 = torch.empty(1, n1, dtype=torch.float16, device=DEVICE)
        split_logits = torch.empty(1, total_n, dtype=torch.float16, device=DEVICE)
        split_output = torch.empty(1, n0, dtype=torch.float16, device=DEVICE)
        esimd_gemv_fp8(normed, weight0_fp8, scale0, split_out0)
        esimd_gemv_fp8(normed, weight1_fp8, scale1, split_out1)
        split_logits[:, :n0] = split_out0
        split_logits[:, n0:] = split_out1
        esimd_gelu_tanh_and_mul(split_logits, split_output)
        torch.xpu.synchronize()

        fused_vs_combined = (fused_output.float() - combined_output.float()).abs()
        fused_vs_split = (fused_output.float() - split_output.float()).abs()
        combined_vs_split = (combined_output.float() - split_output.float()).abs()

        assert fused_vs_combined.max().item() <= 5e-2
        assert fused_vs_combined.mean().item() <= 5e-3
        assert fused_vs_split.max().item() <= 5e-2
        assert fused_vs_split.mean().item() <= 5e-3
        assert combined_vs_split.max().item() <= 5e-2
        assert combined_vs_split.mean().item() <= 5e-3


def benchmark_norm_gemv2_three_paths() -> None:
    torch.manual_seed(42)

    print(
        f"\n{'N0':>6} {'N1':>6} {'K':>6} | {'Method':>8} {'Config':>11} {'GB/s':>8} {'BW%':>7} {'us':>9} {'vs_best':>8}"
    )
    print("-" * 78)

    for n0, n1, k_size in SHAPES:
        hidden = torch.randn(1, k_size, dtype=torch.float16, device=DEVICE) * 0.1
        norm_weight = torch.randn(k_size, dtype=torch.float16, device=DEVICE) * 0.1
        scale0 = torch.tensor([0.0008], dtype=torch.float32, device=DEVICE)
        scale1 = torch.tensor([0.0008], dtype=torch.float32, device=DEVICE)
        combined_scale = torch.tensor([scale0.item(), scale1.item()], dtype=torch.float32, device=DEVICE)

        weight_bytes = (n0 + n1) * k_size
        target_mem = 32 * 1024 * 1024
        num_copies = max(16, target_mem // max(weight_bytes, 1))
        num_copies = min(num_copies, 512)

        weight0_pool = []
        weight1_pool = []
        combined_weight_pool = []
        for _ in range(num_copies):
            weight0_fp16 = torch.randn(n0, k_size, dtype=torch.float16, device=DEVICE) * 0.1
            weight1_fp16 = torch.randn(n1, k_size, dtype=torch.float16, device=DEVICE) * 0.1
            weight0_fp8 = weight0_fp16.to(torch.float8_e4m3fn)
            weight1_fp8 = weight1_fp16.to(torch.float8_e4m3fn)
            weight0_pool.append(weight0_fp8)
            weight1_pool.append(weight1_fp8)
            combined_weight_pool.append(torch.cat([weight0_fp8, weight1_fp8], dim=0))

        total_n = n0 + n1
        fused_vl, fused_ks, _ = _search_best_norm_gemv2_vl_ks(
            hidden,
            norm_weight,
            weight0_pool[0],
            scale0,
            weight1_pool[0],
            scale1,
            EPS,
        )
        combined_vl, combined_ks = _select_norm_gemv_vl_ks(total_n, k_size)
        rms_vl, rms_ks = _select_rms_norm_vl_ks(hidden.shape[0], k_size)

        fused_output = torch.empty(1, n0, dtype=torch.float16, device=DEVICE)
        combined_output = torch.empty(1, n0, dtype=torch.float16, device=DEVICE)
        normed = torch.empty_like(hidden)
        split_out0 = torch.empty(1, n0, dtype=torch.float16, device=DEVICE)
        split_out1 = torch.empty(1, n1, dtype=torch.float16, device=DEVICE)
        split_logits = torch.empty(1, total_n, dtype=torch.float16, device=DEVICE)
        split_output = torch.empty(1, n0, dtype=torch.float16, device=DEVICE)
        run_state = {"index": 0}

        def run_fused() -> None:
            current_idx = run_state["index"] % num_copies
            esimd_norm_gemv2_geglu_fp8_pert(
                hidden,
                norm_weight,
                weight0_pool[current_idx],
                scale0,
                weight1_pool[current_idx],
                scale1,
                fused_output,
                EPS,
                fused_vl,
                fused_ks,
            )
            run_state["index"] += 1

        def run_combined() -> None:
            current_idx = run_state["index"] % num_copies
            esimd_rms_norm(hidden, norm_weight, EPS, normed, rms_vl, rms_ks)
            esimd_gemv_gelu_tanh_mul_fp8_pert(
                normed,
                combined_weight_pool[current_idx],
                combined_scale,
                combined_output,
            )
            run_state["index"] += 1

        def run_split() -> None:
            current_idx = run_state["index"] % num_copies
            esimd_rms_norm(hidden, norm_weight, EPS, normed, rms_vl, rms_ks)
            esimd_gemv_fp8(normed, weight0_pool[current_idx], scale0, split_out0)
            esimd_gemv_fp8(normed, weight1_pool[current_idx], scale1, split_out1)
            split_logits[:, :n0] = split_out0
            split_logits[:, n0:] = split_out1
            esimd_gelu_tanh_and_mul(split_logits, split_output)
            run_state["index"] += 1

        run_state["index"] = 0
        fused_latency_us = _benchmark_xpu_callable(run_fused)
        run_state["index"] = 0
        combined_latency_us = _benchmark_xpu_callable(run_combined)
        run_state["index"] = 0
        split_latency_us = _benchmark_xpu_callable(run_split)

        best_latency_us = min(fused_latency_us, combined_latency_us, split_latency_us)
        fused_bw = (_norm_gemv2_fused_bytes(n0, n1, k_size) / 1e9) / (fused_latency_us / 1e6)
        combined_bw = (_norm_gemv2_combined_bytes(n0, n1, k_size) / 1e9) / (combined_latency_us / 1e6)
        split_bw = (_norm_gemv2_split_bytes(n0, n1, k_size) / 1e9) / (split_latency_us / 1e6)

        print(
            f"{n0:>6} {n1:>6} {k_size:>6} | {'fused2':>8} {fused_vl:>3}:{fused_ks:<7} {fused_bw:>7.1f} {fused_bw / TARGET_BW * 100:>6.1f}% {fused_latency_us:>8.2f} {fused_latency_us / best_latency_us:>8.3f}"
        )
        print(
            f"{n0:>6} {n1:>6} {k_size:>6} | {'combined':>8} {combined_vl:>3}:{combined_ks:<7} {combined_bw:>7.1f} {combined_bw / TARGET_BW * 100:>6.1f}% {combined_latency_us:>8.2f} {combined_latency_us / best_latency_us:>8.3f}"
        )
        print(
            f"{n0:>6} {n1:>6} {k_size:>6} | {'split':>8} {rms_vl:>3}:{rms_ks:<7} {split_bw:>7.1f} {split_bw / TARGET_BW * 100:>6.1f}% {split_latency_us:>8.2f} {split_latency_us / best_latency_us:>8.3f}"
        )


if __name__ == "__main__":
    test_norm_gemv2_matches_three_paths()
    benchmark_norm_gemv2_three_paths()