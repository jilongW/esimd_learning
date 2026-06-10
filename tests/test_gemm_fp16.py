import gc
import time

import torch


device = torch.device("xpu")
GEMM_VL_CANDIDATES = (128, 256, 512)
GEMM_KS_CANDIDATES = (1, 2, 4, 5)
GEMM_SHAPES = [
    # (3072, 2560),
    # (6144, 2560),
    # (2560, 2048),
    # (20480, 2560),
    (262144, 2560),
]


def _valid_gemm_vl_ks(K: int) -> list[tuple[int, int]]:
    valid = []
    for vl in GEMM_VL_CANDIDATES:
        if K % vl != 0:
            continue
        for ks in GEMM_KS_CANDIDATES:
            if K % ks != 0:
                continue
            if (K // ks) % vl != 0:
                continue
            valid.append((vl, ks))
    return valid


def _benchmark_iters(total_bytes: int) -> int:
    if total_bytes < 512 * 1024:
        return 3000
    if total_bytes < 2 * 1024 * 1024:
        return 1200
    return 300


def _benchmark_latency_only(run_fn, iters: int) -> float:
    for _ in range(8):
        run_fn()
    torch.xpu.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        run_fn()
    torch.xpu.synchronize()

    us = (time.perf_counter() - t0) * 1e6 / iters
    gc.collect()
    if hasattr(torch.xpu, "empty_cache"):
        torch.xpu.empty_cache()
    return us


def _benchmark_pair(esimd_run, torch_run, iters: int) -> tuple[float, float]:
    esimd_us = _benchmark_latency_only(esimd_run, iters)
    time.sleep(0.2)
    torch_us = _benchmark_latency_only(torch_run, iters)
    return esimd_us, torch_us


def _autotune_gemm_vl_ks(input_t, weight_t, output_t, iters: int) -> tuple[tuple[int, int], float]:
    from custom_esimd_kernels_vllm import esimd_gemm_fp16

    K = int(input_t.shape[1])
    candidates = _valid_gemm_vl_ks(K)
    if not candidates:
        return (0, 0), float("inf")

    best_cfg = candidates[0]
    best_us = float("inf")

    for vl, ks in candidates:
        us = _benchmark_latency_only(
            lambda: esimd_gemm_fp16(input_t, weight_t, output_t, vl=vl, ks=ks),
            iters,
        )
        if us < best_us:
            best_us = us
            best_cfg = (vl, ks)
        print(f"  autotune vl/ks={vl}/{ks} -> {us:.2f} us")

    return best_cfg, best_us


def _run_correctness_case(N: int, K: int, M: int) -> None:
    from custom_esimd_kernels_vllm import esimd_gemm_fp16

    input_t = torch.randn(M, K, dtype=torch.float16, device=device) * 0.1
    weight_t = torch.randn(N, K, dtype=torch.float16, device=device) * 0.1

    out_auto = torch.zeros(M, N, dtype=torch.float16, device=device)
    out_manual = torch.zeros_like(out_auto)

    esimd_gemm_fp16(input_t, weight_t, out_auto)

    valid_cfg = _valid_gemm_vl_ks(K)
    if valid_cfg:
        vl, ks = valid_cfg[0]
        esimd_gemm_fp16(input_t, weight_t, out_manual, vl=vl, ks=ks)
    else:
        out_manual.copy_(out_auto)

    ref = torch.nn.functional.linear(input_t.float(), weight_t.float(), None)

    auto_diff = (out_auto.float() - ref).abs()
    auto_max = auto_diff.max().item()
    auto_ref_max = ref.abs().max().item()
    auto_rel = auto_max / max(auto_ref_max, 1e-6)

    manual_diff = (out_manual.float() - ref).abs()
    manual_max = manual_diff.max().item()
    manual_ref_max = ref.abs().max().item()
    manual_rel = manual_max / max(manual_ref_max, 1e-6)

    print(
        f"[CORR] M={M:>3} N={N:>6} K={K:>5} "
        f"auto(max={auto_max:.4f}, rel={auto_rel:.4f}) "
        f"manual(max={manual_max:.4f}, rel={manual_rel:.4f})"
    )

    assert auto_max < 1.0 or auto_rel < 0.05
    assert manual_max < 1.0 or manual_rel < 0.05


def test_correctness():
    m_values = [1]
    for N, K in GEMM_SHAPES:
        for M in m_values:
            _run_correctness_case(N, K, M)


def test_auto_vs_manual_match():
    from custom_esimd_kernels_vllm import esimd_gemm_fp16

    N, K = 6144, 2560
    M = 8
    input_t = torch.randn(M, K, dtype=torch.float16, device=device) * 0.1
    weight_t = torch.randn(N, K, dtype=torch.float16, device=device) * 0.1

    out_auto = torch.zeros(M, N, dtype=torch.float16, device=device)
    out_manual = torch.zeros_like(out_auto)

    esimd_gemm_fp16(input_t, weight_t, out_auto)

    valid_cfg = _valid_gemm_vl_ks(K)
    assert valid_cfg, f"No valid vl/ks for K={K}"
    vl, ks = valid_cfg[0]
    esimd_gemm_fp16(input_t, weight_t, out_manual, vl=vl, ks=ks)

    diff = (out_auto.float() - out_manual.float()).abs()
    max_diff = diff.max().item()
    ref_max = out_auto.float().abs().max().item()
    rel = max_diff / max(ref_max, 1e-6)

    print(f"[AUTO-MANUAL] max_diff={max_diff:.6f} rel={rel:.6f} vl={vl} ks={ks}")
    assert max_diff < 1.0 or rel < 0.05


def benchmark_fp16_gemm_vs_torch() -> None:
    from custom_esimd_kernels_vllm import esimd_gemm_fp16

    print(
        f"\n{'Case':<24} {'Config':>22} | {'ESIMD us':>10} {'Torch us':>10} {'ESIMD TF':>9} {'Torch TF':>9}"
    )
    print("-" * 96)

    for N, K in GEMM_SHAPES:
        for M in (1, 2):
            input_t = torch.randn(M, K, dtype=torch.float16, device=device) * 0.1
            weight_t = torch.randn(N, K, dtype=torch.float16, device=device) * 0.1
            output = torch.zeros(M, N, dtype=torch.float16, device=device)

            element_bytes = torch.tensor([], dtype=torch.float16).element_size()
            total_bytes = M * K * element_bytes + N * K * element_bytes + M * N * element_bytes
            total_flops = 2 * M * N * K
            iters = _benchmark_iters(total_bytes)
            tune_iters = max(20, min(100, iters // 4))

            (vl, ks), best_us = _autotune_gemm_vl_ks(input_t, weight_t, output, tune_iters)

            esimd_us, torch_us = _benchmark_pair(
                lambda: esimd_gemm_fp16(input_t, weight_t, output, vl=vl, ks=ks),
                lambda: torch.nn.functional.linear(input_t, weight_t, None),
                iters,
            )

            esimd_tf = total_flops / (esimd_us * 1e6) if esimd_us > 0 else 0.0
            torch_tf = total_flops / (torch_us * 1e6) if torch_us > 0 else 0.0

            print(
                f"{'fp16_gemm':<24} {f'M={M} N={N} K={K}':>22} | "
                f"{esimd_us:>9.2f} {torch_us:>9.2f} {esimd_tf:>8.4f} {torch_tf:>8.4f}"
            )
            print(f"  best vl/ks={vl}/{ks}, tune_us={best_us:.2f}, iters={iters}")


if __name__ == "__main__":
    if not torch.xpu.is_available():
        raise RuntimeError("XPU is not available")

    print("=" * 60)
    print("custom-esimd-kernels-vllm: GEMM FP16 Tests")
    print("=" * 60)

    test_correctness()
    test_auto_vs_manual_match()
    benchmark_fp16_gemm_vs_torch()
