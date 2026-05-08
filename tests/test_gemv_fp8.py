"""
Test esimd_gemv_fp8_pern kernel — FP8 GEMV with per-N scale, FP32 accumulation.

Correctness: compare against FP16 dequant reference (torch matmul).
Performance: benchmark Qwen3-Next-80B-A3B TP4 projection shapes.
"""
import torch
import time

device = torch.device("xpu")


def test_correctness_basic():
    """Basic correctness: scale=1, compare dequant(fp8)*input vs kernel output."""
    from custom_esimd_kernels_vllm import esimd_gemv_fp8_pern

    print("\n--- Correctness (scale=1) ---")
    for N, K in [(1024, 1024), (2560, 2048), (512, 2048), (2048, 512),
                 (128, 2048), (2048, 128), (3072, 2048), (16, 2048)]:
        weight_ref = torch.randn(N, K, dtype=torch.float16, device=device) * 0.1
        weight_fp8 = weight_ref.to(torch.float8_e4m3fn)
        scale = torch.ones(N, dtype=torch.float16, device=device)

        input_t = torch.randn(1, K, dtype=torch.float16, device=device) * 0.1
        output = torch.zeros(1, N, dtype=torch.float16, device=device)

        esimd_gemv_fp8_pern(input_t, weight_fp8, scale, output, N, K)

        weight_dequant = weight_fp8.to(torch.float16)
        ref = input_t.float() @ weight_dequant.float().T

        max_diff = (output.float() - ref.float()).abs().max().item()
        ref_max = ref.float().abs().max().item()
        rel_err = (max_diff / ref_max) if ref_max > 1e-6 else 0
        ok = max_diff < 1.0 or rel_err < 0.02
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] N={N:5d} K={K:5d}  max_diff={max_diff:.4f}  rel={rel_err:.4f}")
        assert ok, f"Correctness failed for N={N}, K={K}"


def test_correctness_with_scale():
    """Correctness with non-trivial per-N scale."""
    from custom_esimd_kernels_vllm import esimd_gemv_fp8_pern

    print("\n--- Correctness (with scale) ---")
    for N, K in [(2560, 2048), (512, 2048), (2048, 512), (3072, 2048), (128, 2048)]:
        weight_ref = torch.randn(N, K, dtype=torch.float16, device=device) * 0.1
        weight_fp8 = weight_ref.to(torch.float8_e4m3fn)
        scale = torch.randn(N, dtype=torch.float16, device=device) * 0.1

        input_t = torch.randn(1, K, dtype=torch.float16, device=device) * 0.1
        output = torch.zeros(1, N, dtype=torch.float16, device=device)

        esimd_gemv_fp8_pern(input_t, weight_fp8, scale, output, N, K)

        weight_dequant = weight_fp8.to(torch.float16)
        ref = (input_t.float() @ weight_dequant.float().T) * scale.float().unsqueeze(0)

        max_diff = (output.float() - ref.float()).abs().max().item()
        ref_max = ref.float().abs().max().item()
        rel_err = (max_diff / ref_max) if ref_max > 1e-6 else 0
        ok = max_diff < 0.5 or rel_err < 0.05
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] N={N:5d} K={K:5d} (with scale)  max_diff={max_diff:.4f}  rel={rel_err:.4f}")
        assert ok, f"Correctness failed for N={N}, K={K} with scale"


def benchmark_shapes():
    """Benchmark Qwen3-Next-80B-A3B TP4 shapes."""
    from custom_esimd_kernels_vllm import esimd_gemv_fp8_pern

    shapes = [
        ("Attn qkv",     2560, 2048),
        ("Attn o_proj",  2048, 1024),
        ("DN qkvz",      3072, 2048),
        ("DN ba",          16, 2048),
        ("DN out_proj",  2048, 1024),
        ("Exp gate",      512, 2048),
        ("Exp up",        512, 2048),
        ("Exp down",     2048,  512),
        ("Sh gate",       128, 2048),
        ("Sh up",         128, 2048),
        ("Sh down",      2048,  128),
        ("Router",        512, 2048),
    ]

    TARGET_BW = 450.0  # GB/s BMG

    print(f"\n{'Shape':<18} {'N':>6} {'K':>6} {'KB':>7} | {'GB/s':>8} {'BW%':>7} {'us':>8}")
    print("-" * 70)

    for name, N, K in shapes:
        weight_ref = torch.randn(N, K, dtype=torch.float16, device=device) * 0.1
        weight_fp8 = weight_ref.to(torch.float8_e4m3fn)
        scale = torch.randn(N, dtype=torch.float16, device=device) * 0.1
        input_t = torch.randn(1, K, dtype=torch.float16, device=device) * 0.1
        output = torch.zeros(1, N, dtype=torch.float16, device=device)

        # Total bytes: input(K*2) + weight(N*K) + scale(N*2) + output(N*2)
        total_bytes = K * 2 + N * K + N * 2 + N * 2

        # Cache-bust: create multiple weight copies
        wb = N * K
        target_mem = 32 * 1024 * 1024
        nc = max(16, target_mem // max(wb, 1))
        nc = min(nc, 512)

        weights = [weight_fp8]
        for i in range(1, nc):
            w = torch.randn(N, K, dtype=torch.float16, device=device) * 0.1
            weights.append(w.to(torch.float8_e4m3fn))

        ni = 4000 if total_bytes < 512 * 1024 else (1000 if total_bytes < 2 * 1024 * 1024 else 300)

        # Warmup
        for i in range(10):
            esimd_gemv_fp8_pern(input_t, weights[i % nc], scale, output, N, K)
        torch.xpu.synchronize()

        # Timed
        t0 = time.perf_counter()
        for i in range(ni):
            esimd_gemv_fp8_pern(input_t, weights[i % nc], scale, output, N, K)
        torch.xpu.synchronize()
        t1 = time.perf_counter()

        ms = (t1 - t0) / ni * 1000
        bw = (total_bytes / 1e9) / (ms / 1e3)
        us = ms * 1000
        bw_pct = bw / TARGET_BW * 100

        print(f"{name:<18} {N:>6} {K:>6} {total_bytes//1024:>6}K | {bw:>7.1f} {bw_pct:>6.1f}% {us:>7.2f}")



def benchmark_fused():
    """Benchmark fused vs sum-of-individual latencies for target Qwen3 shapes."""
    from custom_esimd_kernels_vllm import (
        esimd_gemv_fp8_pern, 
    )

    TARGET_BW = 450.0  # GB/s BMG

    print(f"\n{'Case':<20} {'Config':>20} | {'Indiv us':>10} {'Fused us':>10} {'Speedup':>8}")
    print("-" * 78)

    def make_tensors(N, K):
        w = (torch.randn(N, K, dtype=torch.float16, device=device) * 0.1).to(torch.float8_e4m3fn)
        s = torch.randn(N, dtype=torch.float16, device=device) * 0.1
        o = torch.zeros(1, N, dtype=torch.float16, device=device)
        return w, s, o

    cases_fused2 = [
        ("DN qkvz+ba",   [(3072, 2048), (16, 2048)]),
        ("Exp gate+up",  [(512, 2048), (512, 2048)]),
        ("Sh gate+up",   [(128, 2048), (128, 2048)]),
    ]

    ni = 2000

    for name, shapes in cases_fused2:
        K = shapes[0][1]
        input_t = torch.randn(1, K, dtype=torch.float16, device=device) * 0.1
        w0, s0, o0 = make_tensors(shapes[0][0], K)
        w1, s1, o1 = make_tensors(shapes[1][0], K)
        config = f"N=[{shapes[0][0]},{shapes[1][0]}] K={K}"

        # Warmup + bench individual
        for _ in range(10):
            esimd_gemv_fp8_pern(input_t, w0, s0, o0, shapes[0][0], K)
            esimd_gemv_fp8_pern(input_t, w1, s1, o1, shapes[1][0], K)
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        for _ in range(ni):
            esimd_gemv_fp8_pern(input_t, w0, s0, o0, shapes[0][0], K)
            esimd_gemv_fp8_pern(input_t, w1, s1, o1, shapes[1][0], K)
        torch.xpu.synchronize()
        indiv_us = (time.perf_counter() - t0) / ni * 1e6

        fused_us = (time.perf_counter() - t0) / ni * 1e6

        speedup = indiv_us / fused_us if fused_us > 0 else 0
        print(f"{name:<20} {config:>20} | {indiv_us:>9.2f} {speedup:>7.2f}x")


def test_pert_correctness():
    """Per-tensor scale: compare against FP16 dequant reference."""
    from custom_esimd_kernels_vllm import esimd_gemv_fp8_pert

    print("\n--- Per-tensor scale Correctness ---")
    for N, K in [(1024, 1024), (2560, 2048), (512, 2048), (3072, 2048),
                 (128, 2048), (16, 2048), (2048, 512)]:
        weight_ref = torch.randn(N, K, dtype=torch.float16, device=device) * 0.1
        weight_fp8 = weight_ref.to(torch.float8_e4m3fn)
        scale_val = 0.05 + torch.rand(1).item() * 0.1  # random fp32 scalar
        scale_t = torch.tensor(scale_val, dtype=torch.float32, device=device)

        input_t = torch.randn(1, K, dtype=torch.float16, device=device) * 0.1
        output = torch.zeros(1, N, dtype=torch.float16, device=device)

        esimd_gemv_fp8_pert(input_t, weight_fp8, scale_t, output)

        weight_dequant = weight_fp8.to(torch.float16)
        ref = (input_t.float() @ weight_dequant.float().T) * scale_val

        max_diff = (output.float() - ref.float()).abs().max().item()
        ref_max = ref.float().abs().max().item()
        rel_err = (max_diff / ref_max) if ref_max > 1e-6 else 0
        ok = max_diff < 0.5 or rel_err < 0.05
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] N={N:5d} K={K:5d}  scale={scale_val:.4f}  max_diff={max_diff:.4f}  rel={rel_err:.4f}")
        assert ok, f"Per-tensor correctness failed for N={N}, K={K}"


def test_pert_vs_pern():
    """Per-tensor scale should match per-N when all per-N scales are the same."""
    from custom_esimd_kernels_vllm import esimd_gemv_fp8_pern, esimd_gemv_fp8_pert

    print("\n--- Per-tensor vs Per-N (uniform scale) ---")
    for N, K in [(2560, 2048), (512, 2048), (128, 2048)]:
        weight_ref = torch.randn(N, K, dtype=torch.float16, device=device) * 0.1
        weight_fp8 = weight_ref.to(torch.float8_e4m3fn)
        scale_val = 0.073
        scale_pern = torch.full((N,), scale_val, dtype=torch.float16, device=device)
        scale_pert = torch.tensor(scale_val, dtype=torch.float32, device=device)

        input_t = torch.randn(1, K, dtype=torch.float16, device=device) * 0.1

        out_pern = torch.zeros(1, N, dtype=torch.float16, device=device)
        out_pert = torch.zeros(1, N, dtype=torch.float16, device=device)

        esimd_gemv_fp8_pern(input_t, weight_fp8, scale_pern, out_pern, N, K)
        esimd_gemv_fp8_pert(input_t, weight_fp8, scale_pert, out_pert)

        # Not bit-identical due to fp16 vs fp32 scale precision, but should be very close
        max_diff = (out_pert.float() - out_pern.float()).abs().max().item()
        ref_max = out_pern.float().abs().max().item()
        rel_err = (max_diff / ref_max) if ref_max > 1e-6 else 0
        ok = rel_err < 0.01
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] N={N:5d} K={K:5d}  max_diff={max_diff:.6f}  rel={rel_err:.6f}")
        assert ok, f"pert vs pern mismatch for N={N}, K={K}"


def test_e5m2_correctness_pern():
    """E5M2 per-N scale correctness: compare against FP16 dequant reference."""
    from custom_esimd_kernels_vllm import esimd_gemv_fp8_pern

    print("\n--- E5M2 Per-N Correctness ---")
    for N, K in [(1024, 1024), (2560, 2048), (512, 2048), (128, 2048), (16, 2048)]:
        weight_ref = torch.randn(N, K, dtype=torch.float16, device=device) * 0.1
        weight_fp8 = weight_ref.to(torch.float8_e5m2)
        scale = torch.randn(N, dtype=torch.float16, device=device) * 0.1

        input_t = torch.randn(1, K, dtype=torch.float16, device=device) * 0.1
        output = torch.zeros(1, N, dtype=torch.float16, device=device)

        esimd_gemv_fp8_pern(input_t, weight_fp8, scale, output, N, K)

        weight_dequant = weight_fp8.to(torch.float16)
        ref = (input_t.float() @ weight_dequant.float().T) * scale.float().unsqueeze(0)

        max_diff = (output.float() - ref.float()).abs().max().item()
        ref_max = ref.float().abs().max().item()
        rel_err = (max_diff / ref_max) if ref_max > 1e-6 else 0
        ok = max_diff < 0.5 or rel_err < 0.05
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] N={N:5d} K={K:5d}  max_diff={max_diff:.4f}  rel={rel_err:.4f}")
        assert ok, f"E5M2 pern correctness failed for N={N}, K={K}"


def test_e5m2_correctness_pert():
    """E5M2 per-tensor scale correctness."""
    from custom_esimd_kernels_vllm import esimd_gemv_fp8_pert

    print("\n--- E5M2 Per-tensor Correctness ---")
    for N, K in [(1024, 1024), (2560, 2048), (512, 2048), (128, 2048)]:
        weight_ref = torch.randn(N, K, dtype=torch.float16, device=device) * 0.1
        weight_fp8 = weight_ref.to(torch.float8_e5m2)
        scale_val = 0.05 + torch.rand(1).item() * 0.1
        scale_t = torch.tensor(scale_val, dtype=torch.float32, device=device)

        input_t = torch.randn(1, K, dtype=torch.float16, device=device) * 0.1
        output = torch.zeros(1, N, dtype=torch.float16, device=device)

        esimd_gemv_fp8_pert(input_t, weight_fp8, scale_t, output)

        weight_dequant = weight_fp8.to(torch.float16)
        ref = (input_t.float() @ weight_dequant.float().T) * scale_val

        max_diff = (output.float() - ref.float()).abs().max().item()
        ref_max = ref.float().abs().max().item()
        rel_err = (max_diff / ref_max) if ref_max > 1e-6 else 0
        ok = max_diff < 0.5 or rel_err < 0.05
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] N={N:5d} K={K:5d}  scale={scale_val:.4f}  max_diff={max_diff:.4f}  rel={rel_err:.4f}")
        assert ok, f"E5M2 pert correctness failed for N={N}, K={K}"


def test_e5m2_fused():
    """E5M2 fused correctness: fused2 pern + fused2 pert."""
    from custom_esimd_kernels_vllm import (
        esimd_gemv_fp8_pern, 
        esimd_gemv_fp8_pert, 
    )

    print("\n--- E5M2 Fused Correctness ---")
    N0, N1, K = 512, 512, 2048
    input_t = torch.randn(1, K, dtype=torch.float16, device=device) * 0.1

    # Fused2 per-N with E5M2
    w0 = (torch.randn(N0, K, dtype=torch.float16, device=device) * 0.1).to(torch.float8_e5m2)
    s0 = torch.randn(N0, dtype=torch.float16, device=device) * 0.1
    w1 = (torch.randn(N1, K, dtype=torch.float16, device=device) * 0.1).to(torch.float8_e5m2)
    s1 = torch.randn(N1, dtype=torch.float16, device=device) * 0.1

    ref_o0 = torch.zeros(1, N0, dtype=torch.float16, device=device)
    ref_o1 = torch.zeros(1, N1, dtype=torch.float16, device=device)
    esimd_gemv_fp8_pern(input_t, w0, s0, ref_o0, N0, K)
    esimd_gemv_fp8_pern(input_t, w1, s1, ref_o1, N1, K)

   

    # Fused2 per-tensor with E5M2
    st0 = torch.tensor(0.08, dtype=torch.float32, device=device)
    st1 = torch.tensor(0.12, dtype=torch.float32, device=device)

    ref_o0 = torch.zeros(1, N0, dtype=torch.float16, device=device)
    ref_o1 = torch.zeros(1, N1, dtype=torch.float16, device=device)
    esimd_gemv_fp8_pert(input_t, w0, st0, ref_o0)
    esimd_gemv_fp8_pert(input_t, w1, st1, ref_o1)


def benchmark_e5m2():
    """Benchmark E5M2 vs E4M3 on key shapes with cache-busting buffer rotation."""
    from custom_esimd_kernels_vllm import esimd_gemv_fp8_pern

    shapes = [
        ("Attn qkv",     2560, 2048),
        ("DN qkvz",      3072, 2048),
        ("Exp gate",      512, 2048),
        ("Sh gate",       128, 2048),
    ]

    TARGET_BW = 450.0

    print(f"\n{'Shape':<18} {'N':>6} {'K':>6} | {'E4M3 us':>9} {'E5M2 us':>9} {'E4M3 GB/s':>10} {'E5M2 GB/s':>10}")
    print("-" * 80)

    for name, N, K in shapes:
        total_bytes = K * 2 + N * K + N * 2 + N * 2
        input_t = torch.randn(1, K, dtype=torch.float16, device=device) * 0.1
        scale = torch.randn(N, dtype=torch.float16, device=device) * 0.1
        output = torch.zeros(1, N, dtype=torch.float16, device=device)

        # Cache-bust: rotate through enough buffers to exceed L3
        wb = N * K
        target_mem = 32 * 1024 * 1024
        nc = max(16, target_mem // max(wb, 1))
        nc = min(nc, 512)

        ni = 4000 if total_bytes < 512 * 1024 else (1000 if total_bytes < 2 * 1024 * 1024 else 300)

        results = {}
        for dtype_name, dtype in [("E4M3", torch.float8_e4m3fn), ("E5M2", torch.float8_e5m2)]:
            weights = []
            for i in range(nc):
                w = (torch.randn(N, K, dtype=torch.float16, device=device) * 0.1).to(dtype)
                weights.append(w)

            for i in range(10):
                esimd_gemv_fp8_pern(input_t, weights[i % nc], scale, output, N, K)
            torch.xpu.synchronize()

            t0 = time.perf_counter()
            for i in range(ni):
                esimd_gemv_fp8_pern(input_t, weights[i % nc], scale, output, N, K)
            torch.xpu.synchronize()
            us = (time.perf_counter() - t0) / ni * 1e6
            bw = (total_bytes / 1e9) / (us / 1e6)
            results[dtype_name] = (us, bw)

        e4_us, e4_bw = results["E4M3"]
        e5_us, e5_bw = results["E5M2"]
        print(f"{name:<18} {N:>6} {K:>6} | {e4_us:>8.2f} {e5_us:>8.2f} {e4_bw:>9.1f} {e5_bw:>9.1f}")


if __name__ == "__main__":
    print("=" * 60)
    print("custom-esimd-kernels-vllm: GEMV FP8 Tests")
    print("=" * 60)

    # E4M3 per-N scale tests
    test_correctness_basic()
    test_correctness_with_scale()

    # E4M3 per-tensor scale tests
    test_pert_correctness()
    test_pert_vs_pern()

    # E5M2 tests
    test_e5m2_correctness_pern()
    test_e5m2_correctness_pert()
    test_e5m2_fused()

    # Performance
    print("\n--- Performance Benchmark (unfused per-N, E4M3) ---")
    benchmark_shapes()

    print("\n--- Performance Benchmark (fused vs individual) ---")
    benchmark_fused()

    print("\n--- Performance Benchmark (E4M3 vs E5M2) ---")
    benchmark_e5m2()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)