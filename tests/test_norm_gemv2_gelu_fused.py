import time

import torch

from custom_esimd_kernels_vllm import (
    esimd_gemv_gelu_tanh_mul_fp8_pert,
    esimd_gemv_fp8,
    esimd_gelu_tanh_and_mul,
    esimd_norm_gemv2_geglu_fp8_pert,
    esimd_rms_norm,
)

DEVICE = torch.device("xpu")
EPS = 1e-6
WARMUP_ITERS = 10
BENCHMARK_ITERS = 500
TARGET_BW = 112.0
SHAPES = [
    # (256, 256, 2048),
    # (256, 256, 2560),
    # (1024, 1024, 2560),
    (10240, 10240, 2560),
    # (10240, 10240, 5120),
]
NORM_GEMV2_VL_CANDIDATES = (128, 256, 512)
NORM_GEMV2_KS_CANDIDATES = (1, 2, 4, 8, 10)
RMS_VL_CANDIDATES = (128, 256, 512, 1024)
RMS_KS_CANDIDATES = (1, 2, 4, 5, 8, 10)


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


def _valid_norm_gemv2_configs(k_size: int) -> list[tuple[int, int]]:
    valid: list[tuple[int, int]] = []
    supported_by_kernel = {
        (512, 1),
        (512, 2),
        (256, 1),
        (256, 2),
        (256, 4),
        (256, 8),
        (128, 1),
        (128, 2),
        (128, 4),
        (128, 8),
        (128, 10),
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

    if not valid:
        raise RuntimeError(f"No valid norm_gemv2 vl/ks for K={k_size}")
    return valid


def _select_combined_heuristic_vl_ks(total_n: int, k_size: int) -> tuple[int, int]:
    if k_size == 256:
        return 256, 1
    if k_size == 2048:
        return 256, 8
    if k_size == 4096:
        return 256, 4
    if k_size == 10240:
        return 256, 8
    if k_size == 2560:
        if total_n <= 256:
            return 128, 10
        if total_n >= 10240:
            return 256, 1
        return 128, 4
    if k_size >= 4096:
        return 256, 4
    if k_size >= 2048:
        return 256, 8
    return 128, 1


def _valid_combined_configs(total_n: int, k_size: int) -> list[tuple[int, int]]:
    del total_n
    valid: list[tuple[int, int]] = []
    for vl in NORM_GEMV2_VL_CANDIDATES:
        if k_size % vl != 0:
            continue
        for ks in NORM_GEMV2_KS_CANDIDATES:
            if k_size % ks != 0:
                continue
            if (k_size // ks) % vl != 0:
                continue
            if (vl * ks >= k_size) and not (k_size == 256 and vl == 256 and ks == 1):
                continue
            valid.append((vl, ks))

    if not valid:
        raise RuntimeError(f"No valid combined vl/ks for K={k_size}")
    return valid


def _select_rms_norm_vl_ks(rows: int, k_size: int) -> tuple[int, int]:
    del rows
    if k_size <= 256:
        return 256, 1
    if k_size <= 512:
        return 256, 1
    if k_size <= 2048:
        return 512, 2
    if k_size <= 2560:
        return 256, 8
    if k_size <= 5120:
        return 512, 8
    return 1024, 1


def _valid_rms_configs(k_size: int) -> list[tuple[int, int]]:
    valid: list[tuple[int, int]] = []
    for vl in RMS_VL_CANDIDATES:
        if k_size % vl != 0:
            continue
        for ks in RMS_KS_CANDIDATES:
            if k_size % ks != 0:
                continue
            if (k_size // ks) % vl != 0:
                continue
            if (vl * ks >= k_size) and not (k_size == 256 and vl == 256 and ks == 1):
                continue
            valid.append((vl, ks))

    if not valid:
        raise RuntimeError(f"No valid rms vl/ks for K={k_size}")
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


def _search_best_combined_vl_ks(
    hidden: torch.Tensor,
    combined_weight: torch.Tensor,
    combined_scale: torch.Tensor,
    eps: float,
) -> tuple[int, int, float]:
    del eps
    k_size = int(hidden.shape[1])
    total_n = int(combined_weight.shape[0])
    n_out = total_n // 2
    candidates = _valid_combined_configs(total_n, k_size)

    heuristic_vl, heuristic_ks = _select_combined_heuristic_vl_ks(total_n, k_size)
    latency_tolerance_us = 0.2
    best_cfg = None
    best_us = None
    rule_us = None
    candidate_records: list[tuple[tuple[int, int], float]] = []

    output = torch.empty(1, n_out, dtype=hidden.dtype, device=hidden.device)

    for vl, ks in candidates:
        def run_once(vl=vl, ks=ks) -> None:
            esimd_gemv_gelu_tanh_mul_fp8_pert(
                hidden,
                combined_weight,
                combined_scale,
                output,
                vl,
                ks,
            )

        candidate_us = _benchmark_xpu_callable(run_once, warmup_iters=4, benchmark_iters=40)
        candidate_records.append(((vl, ks), candidate_us))
        if (vl, ks) == (heuristic_vl, heuristic_ks):
            rule_us = candidate_us
        if best_us is None or candidate_us < (best_us - latency_tolerance_us):
            best_cfg = (vl, ks)
            best_us = candidate_us
        # print(f"Tested combined config K={k_size} N={total_n}: VL={vl} KS={ks} -> {candidate_us:.2f} us")
    if best_cfg is None:
        best_cfg = (heuristic_vl, heuristic_ks)
        best_us = rule_us if rule_us is not None else float("inf")

    return best_cfg[0], best_cfg[1], float(best_us)


def _search_best_rms_vl_ks(
    hidden: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
) -> tuple[int, int, float]:
    rows = int(hidden.shape[0])
    k_size = int(hidden.shape[1])
    candidates = _valid_rms_configs(k_size)

    heuristic_vl, heuristic_ks = _select_rms_norm_vl_ks(rows, k_size)
    latency_tolerance_us = 0.2
    best_cfg = None
    best_us = None
    rule_us = None
    candidate_records: list[tuple[tuple[int, int], float]] = []

    output = torch.empty_like(hidden)

    for vl, ks in candidates:
        def run_once(vl=vl, ks=ks) -> None:
            esimd_rms_norm(hidden, norm_weight, eps, output, vl, ks)

        candidate_us = _benchmark_xpu_callable(run_once, warmup_iters=4, benchmark_iters=40)
        candidate_records.append(((vl, ks), candidate_us))
        if (vl, ks) == (heuristic_vl, heuristic_ks):
            rule_us = candidate_us
        if best_us is None or candidate_us < (best_us - latency_tolerance_us):
            best_cfg = (vl, ks)
            best_us = candidate_us

    if best_cfg is None:
        best_cfg = (heuristic_vl, heuristic_ks)
        best_us = rule_us if rule_us is not None else float("inf")

    return best_cfg[0], best_cfg[1], float(best_us)


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
        combined_vl, combined_ks, _ = _search_best_combined_vl_ks(hidden, combined_weight, combined_scale, EPS)
        combined_output = torch.empty(1, n0, dtype=torch.float16, device=DEVICE)
        combined_normed = torch.empty_like(hidden)
        esimd_rms_norm(hidden, norm_weight, EPS, combined_normed, 256, 1)
        esimd_gemv_gelu_tanh_mul_fp8_pert(
            combined_normed,
            combined_weight,
            combined_scale,
            combined_output,
            combined_vl,
            combined_ks,
        )

        rms_vl, rms_ks, _ = _search_best_rms_vl_ks(hidden, norm_weight, EPS)
        normed = torch.empty_like(hidden)
        esimd_rms_norm(hidden, norm_weight, EPS, normed, rms_vl, rms_ks)

        total_n = n0 + n1
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
        combined_scale = torch.tensor([scale0.item()], dtype=torch.float32, device=DEVICE)

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
        combined_vl, combined_ks, _ = _search_best_combined_vl_ks(
            hidden,
            combined_weight_pool[0],
            combined_scale,
            EPS,
        )
        rms_vl, rms_ks, _ = _search_best_rms_vl_ks(hidden, norm_weight, EPS)

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
                combined_vl,
                combined_ks,
            )
            run_state["index"] += 1

        def run_split() -> None:
            current_idx = run_state["index"] % num_copies
            esimd_rms_norm(hidden, norm_weight, EPS, normed, rms_vl, rms_ks)
            esimd_gemv_fp8(normed, combined_weight_pool[current_idx], combined_scale, split_logits)
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
