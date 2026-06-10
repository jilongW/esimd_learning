import gc
import torch
import time
import sys
from vllm.platforms import current_platform

device = torch.device("xpu")
VL_CANDIDATES = (128, 256, 512)
KS_CANDIDATES = (1, 2, 4, 8, 10)
GEMM_VL_CANDIDATES = (64, 128, 256, 512)
GEMM_KS_CANDIDATES = (1, 2, 4, 5)
SUPPORTED_GEMV_CONFIGS = {
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

GEMV_SHAPES = [
    # ("qkv_proj", 3072, 2560),
    # ("qkv_proj", 6144, 2560),
    # ("Attn o_proj", 2560, 2048),
    # ("Attn o_proj", 2560, 4096),
    # ("gate_up_proj", 20480, 2560),
    # ("down_proj", 2560, 10240),
    # ("per_layer_input_gate", 256, 2560),
    # ("per_layer_input_gate_out", 2560, 256),
    ("embedding", 262144, 2560),
]


def _normalize_vl_ks(K: int, vl: int, ks: int) -> tuple[int, int]:
    kpt = K // ks
    while vl > kpt or kpt % vl != 0:
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
        kpt = K // ks
    return vl, ks


def _select_vl_ks_impl(N: int, K: int, *, k256_vl: int, k256_ks: int) -> tuple[int, int]:
    if K < 256:
        vl, ks = 128, 1
    elif K == 256:
        vl, ks = k256_vl, k256_ks
    elif K >= 10240:
        vl, ks = 512, 2
    elif K >= 4096:
        vl, ks = 512, 2
    elif K >= 2560 and N >= 10240:
        vl, ks = 512, 1
    elif K >= 2560:
        vl, ks = 128, 10
    elif K >= 2048:
        vl, ks = 256, 8
    else:
        vl, ks = 512, 1

    return _normalize_vl_ks(K, vl, ks)


def select_vl_ks_pert(N: int, K: int) -> tuple[int, int]:
    vl, ks = _select_vl_ks_impl(N, K, k256_vl=256, k256_ks=1)
    if (vl, ks) not in _valid_vl_ks(K):
        raise ValueError(f"No valid pert vl/ks for N={N}, K={K}")
    return vl, ks


def _valid_vl_ks(K: int) -> list[tuple[int, int]]:
    valid = []
    for vl in VL_CANDIDATES:
        if K % vl != 0:
            continue
        for ks in KS_CANDIDATES:
            if (vl, ks) not in SUPPORTED_GEMV_CONFIGS:
                continue
            if K % ks != 0:
                continue
            if (K // ks) % vl != 0:
                continue
            valid.append((vl, ks))
    return valid


def _benchmark_one(run_fn, output_fn, iters: int, sample_count: int = 64) -> tuple[float, list[float]]:
    for _ in range(10):
        run_fn(0)
    torch.xpu.synchronize()

    # Timed region: kernel execution only.
    t0 = time.perf_counter()
    for index in range(iters):
        run_fn(index)
    torch.xpu.synchronize()
    elapsed_us = (time.perf_counter() - t0) * 1e6 / iters

    # Untimed region: collect a small deterministic sample for parity checks.
    outputs = []
    collect_iters = min(iters, sample_count)
    for index in range(collect_iters):
        run_fn(index)
        output = output_fn()
        sample_index = [0] * output.dim()
        if output.dim() >= 2:
            sample_index[1] = index % output.size(1)
        else:
            sample_index[0] = index % output.size(0)
        outputs.append(float(output[tuple(sample_index)]))

    torch.xpu.synchronize()
    gc.collect()
    if hasattr(torch.xpu, "empty_cache"):
        torch.xpu.empty_cache()
    return elapsed_us, outputs


def _benchmark_latency_only(run_fn, iters: int) -> float:
    for _ in range(8):
        run_fn(0)
    torch.xpu.synchronize()

    t0 = time.perf_counter()
    for index in range(iters):
        run_fn(index)
    torch.xpu.synchronize()
    elapsed_us = (time.perf_counter() - t0) * 1e6 / iters
    gc.collect()
    if hasattr(torch.xpu, "empty_cache"):
        torch.xpu.empty_cache()
    return elapsed_us


def _assert_output_lists_close(lhs_outputs: list[float], rhs_outputs: list[float]) -> None:
    assert len(lhs_outputs) == len(rhs_outputs), (
        f"output list length mismatch: lhs={len(lhs_outputs)} rhs={len(rhs_outputs)}"
    )
    for index, (lhs_out, rhs_out) in enumerate(zip(lhs_outputs, rhs_outputs)):
        max_diff = abs(lhs_out - rhs_out)
        scale = max(abs(lhs_out), abs(rhs_out), 1e-6)
        assert max_diff < 1.0 or (max_diff / scale) < 0.05, (
            f"benchmark output mismatch at iter={index}, diff={max_diff:.4f}"
        )


def _benchmark_iters(total_bytes: int) -> int:
    if total_bytes < 512 * 1024:
        return 4000
    if total_bytes < 2 * 1024 * 1024:
        return 1000
    return 300


def _valid_gemm_vl_ks(K: int) -> list[tuple[int, int]]:
    valid = []
    for vl in GEMM_VL_CANDIDATES:
        for ks in GEMM_KS_CANDIDATES:
            if ks <= 0:
                continue
            if K % ks != 0:
                continue
            if (K // ks) % vl != 0:
                continue
            valid.append((vl, ks))
    return valid


def _autotune_gemm_vl_ks(
    input_t: torch.Tensor,
    weights: list[torch.Tensor],
    scale_scalar: torch.Tensor,
    out_gemm: torch.Tensor,
    K: int,
    nc: int,
    iters: int,
) -> tuple[tuple[int, int], float, list[tuple[int, int]], float]:
    from custom_esimd_kernels_vllm import esimd_gemm_fp8_pert

    candidates = _valid_gemm_vl_ks(K)
    if not candidates:
        return (0, 0), float("inf"), [(0, 0)], float("inf")

    best_us = float("inf")
    records: list[tuple[float, tuple[int, int]]] = []
    for vl, ks in candidates:
        us = _benchmark_latency_only(
            lambda index, vl=vl, ks=ks: esimd_gemm_fp8_pert(
                input_t,
                weights[index % nc],
                scale_scalar,
                out_gemm,
                vl,
                ks,
            ),
            iters,
        )
        records.append((us, (vl, ks)))
        if us < best_us:
            best_us = us
    threshold_us = best_us + 1.0
    sorted_records = sorted(records, key=lambda x: x[0])
    best_pick_us, best_pick = sorted_records[0]
    best_cfg = [cfg for us, cfg in sorted_records if us <= threshold_us]
    if not best_cfg:
        best_cfg = [best_pick]

    return best_pick, best_pick_us, best_cfg, best_us


def benchmark_gemm_vs_gemv_vs_vllm():
    from custom_esimd_kernels_vllm import esimd_gemm_fp8_pert, esimd_gemv_fp8

    if not hasattr(torch.ops, "_xpu_C") or not hasattr(torch.ops._xpu_C, "fp8_gemm_w8a16"):
        raise RuntimeError("torch.ops._xpu_C.fp8_gemm_w8a16 is unavailable")

    print(
        f"\n{'Case':<30} {'Config':>20} | {'GEMM us':>9} {'GEMV us':>9} {'vllm us':>9} {'GEMM TF':>9} {'GEMV TF':>9} {'vllm TF':>9} {'GEMM GB/s':>11} {'GEMV GB/s':>11} {'vllm GB/s':>11}"
    )
    print("-" * 160)

    for name, N, K in GEMV_SHAPES:
        weight_ref = torch.randn(N, K, dtype=torch.float16, device=device) * 0.1
        weight_fp8 = weight_ref.to(torch.float8_e4m3fn)
        scale_val = 0.05 + torch.rand(1).item() * 0.1
        scale_scalar = torch.tensor(scale_val, dtype=torch.float32, device=device)
        scale_pern = torch.full((N,), scale_val, dtype=torch.float16, device=device)

        wb = N * K
        target_mem = 32 * 1024 * 1024
        nc = max(16, target_mem // max(wb, 1))
        nc = min(nc, 512)

        weights = [weight_fp8]
        for _ in range(1, nc):
            w = torch.randn(N, K, dtype=torch.float16, device=device) * 0.1
            weights.append(w.to(torch.float8_e4m3fn))

        for dtype_name, io_dtype in (("fp16", torch.float16), ("bf16", torch.bfloat16)):
            input_t = torch.randn(1, K, dtype=io_dtype, device=device) * 0.1
            out_gemm = torch.zeros(1, N, dtype=io_dtype, device=device)
            out_gemv = torch.zeros(1, N, dtype=io_dtype, device=device)
            vllm_output = [torch.zeros(1, N, dtype=io_dtype, device=device)]
            vl, ks = select_vl_ks_pert(N, K)

            element_bytes = torch.tensor([], dtype=io_dtype).element_size()
            total_bytes = K * element_bytes + N * K + N * 2 + N * element_bytes
            total_flops = 2 * N * K
            ni = _benchmark_iters(total_bytes)
            tune_iters = max(20, min(120, ni // 4))
            config = f"N={N} K={K} {dtype_name}"

            vllm_us, vllm_outputs = _benchmark_one(
                lambda index: vllm_output.__setitem__(
                    0,
                    torch.ops._xpu_C.fp8_gemm_w8a16(
                        input_t,
                        weights[index % nc].t(),
                        scale_pern,
                        None,
                    ),
                ),
                lambda: vllm_output[0],
                ni,
            )

            time.sleep(1)

            best_pick, best_pick_us, best_cfg, best_us = _autotune_gemm_vl_ks(
                input_t=input_t,
                weights=weights,
                scale_scalar=scale_scalar,
                out_gemm=out_gemm,
                K=K,
                nc=nc,
                iters=tune_iters,
            )
            gemm_vl, gemm_ks = best_pick

            gemm_us, gemm_outputs = _benchmark_one(
                lambda index: esimd_gemm_fp8_pert(
                    input_t,
                    weights[index % nc],
                    scale_scalar,
                    out_gemm,
                    gemm_vl,
                    gemm_ks,
                ),
                lambda: out_gemm,
                ni,
            )
            time.sleep(1)

            gemv_us, gemv_outputs = _benchmark_one(
                lambda index: esimd_gemv_fp8(
                    input_t,
                    weights[index % nc],
                    scale_scalar,
                    out_gemv,

                ),
                lambda: out_gemv,
                ni,
            )
            time.sleep(1)

            

            _assert_output_lists_close(gemm_outputs, gemv_outputs)
            _assert_output_lists_close(gemm_outputs, vllm_outputs)

            gemm_tflops = total_flops / (gemm_us * 1e6) if gemm_us > 0 else 0
            gemv_tflops = total_flops / (gemv_us * 1e6) if gemv_us > 0 else 0
            vllm_tflops = total_flops / (vllm_us * 1e6) if vllm_us > 0 else 0
            gemm_bw = (total_bytes / 1e9) / (gemm_us / 1e6) if gemm_us > 0 else 0
            gemv_bw = (total_bytes / 1e9) / (gemv_us / 1e6) if gemv_us > 0 else 0
            vllm_bw = (total_bytes / 1e9) / (vllm_us / 1e6) if vllm_us > 0 else 0

            print(
                f"{name:<30} {config:>20} | {gemm_us:>8.2f} {gemv_us:>8.2f} {vllm_us:>8.2f} "
                f"{gemm_tflops:>8.4f} {gemv_tflops:>8.4f} {vllm_tflops:>8.4f} "
                f"{gemm_bw:>10.2f} {gemv_bw:>10.2f} {vllm_bw:>10.2f}"
            )
            print(
                f"  gemm autotune pick: vl={gemm_vl}, ks={gemm_ks}; "
                f"pick_us={best_pick_us:.2f}; best_us={best_us:.2f}; "
                f"near_best_cfgs={best_cfg}"
            )
            time.sleep(1)


def _run_correctness_case(weight_dtype, io_dtype):
    from custom_esimd_kernels_vllm import esimd_gemm_fp8_pert

    weight_name = "E5M2" if weight_dtype == torch.float8_e5m2 else "E4M3"
    print(f"\n--- GEMM {weight_name} Correctness ({str(io_dtype).split('.')[-1]}) ---")
    shapes = [
        (3072, 2560),
        (6144, 2560),
        (2560, 2048),
        (2560, 4096),
        (20480, 2560),
        (2560, 10240),
        (256, 2560),
        (2560, 256),
    ]
    m_values = [1, 2, 4, 8, 16, 32, 64]

    for N, K in shapes:
        for M in m_values:
            weight_ref = torch.randn(N, K, dtype=torch.float16, device=device) * 0.1
            weight_fp8 = weight_ref.to(weight_dtype)
            scale_val = 1.0
            scale_t = torch.tensor(scale_val, dtype=torch.float32, device=device)

            input_t = torch.randn(M, K, dtype=io_dtype, device=device) * 0.1
            output = torch.zeros(M, N, dtype=io_dtype, device=device)

            esimd_gemm_fp8_pert(input_t, weight_fp8, scale_t, output)

            weight_dequant = weight_fp8.to(torch.float16)
            ref = (input_t.float() @ weight_dequant.float().T) * scale_val

            max_diff = (output.float() - ref.float()).abs().max().item()
            ref_max = ref.float().abs().max().item()
            rel_err = (max_diff / ref_max) if ref_max > 1e-6 else 0
            ok = max_diff < 1.0 or rel_err < 0.05
            status = "PASS" if ok else "FAIL"
            print(
                f"  [{status}] M={M:>3} N={N:>5} K={K:>5} {weight_name} {str(io_dtype).split('.')[-1]}"
                f"  max_diff={max_diff:.4f}  rel={rel_err:.4f}"
            )
            assert ok, (
                f"Correctness failed for M={M}, N={N}, K={K}, "
                f"weight={weight_name}, io_dtype={io_dtype}"
            )


def test_correctness():
    """Correctness across M=1..64 for key shapes and fp16/bf16 IO."""
    _run_correctness_case(torch.float8_e4m3fn, torch.float16)
    _run_correctness_case(torch.float8_e4m3fn, torch.bfloat16)


def test_e5m2_correctness():
    """E5M2 correctness across M values and fp16/bf16 IO."""
    _run_correctness_case(torch.float8_e5m2, torch.float16)
    _run_correctness_case(torch.float8_e5m2, torch.bfloat16)


def test_gemm_vs_gemv_m1():
    """M=1: GEMM dispatch should produce same result as dedicated GEMV."""
    from custom_esimd_kernels_vllm import esimd_gemv_fp8, esimd_gemm_fp8_pert

    for io_dtype in [torch.float16, torch.bfloat16]:
        print(f"\n--- GEMM vs GEMV at M=1 ({str(io_dtype).split('.')[-1]}) ---")
        for _, N, K in GEMV_SHAPES:
            weight_ref = torch.randn(N, K, dtype=torch.float16, device=device) * 0.1
            weight_fp8 = weight_ref.to(torch.float8_e4m3fn)
            scale_t = torch.tensor(0.073, dtype=torch.float32, device=device)
            input_t = torch.randn(1, K, dtype=io_dtype, device=device) * 0.1
            vl, ks = select_vl_ks_pert(N, K)

            out_gemv = torch.zeros(1, N, dtype=io_dtype, device=device)
            out_gemm = torch.zeros(1, N, dtype=io_dtype, device=device)

            esimd_gemv_fp8(input_t, weight_fp8, scale_t, out_gemv)
            esimd_gemm_fp8_pert(input_t, weight_fp8, scale_t, out_gemm)

            # Both use batched GEMV internally for M=1, should be close.
            max_diff = (out_gemm.float() - out_gemv.float()).abs().max().item()
            ref_max = out_gemv.float().abs().max().item()
            rel_err = (max_diff / ref_max) if ref_max > 1e-6 else 0
            ok = rel_err < 0.01
            status = "PASS" if ok else "FAIL"
            print(
                f"  [{status}] N={N:>5} K={K:>5} {str(io_dtype).split('.')[-1]}"
                f"  max_diff={max_diff:.6f}  rel={rel_err:.6f}"
            )
            assert ok, f"GEMM vs GEMV mismatch at M=1 for N={N}, K={K}, io_dtype={io_dtype}"


if __name__ == "__main__":
    print("=" * 60)
    print("custom-esimd-kernels-vllm: GEMM FP8 Per-tensor Tests")
    print("=" * 60)

    test_correctness()
    test_e5m2_correctness()
    test_gemm_vs_gemv_m1()
    benchmark_gemm_vs_gemv_vs_vllm()

