"""
Test esimd_gemv_fp8_pern kernel — FP8 GEMV with per-N scale, FP32 accumulation.

Correctness: compare against FP16 dequant reference (torch matmul).
Performance: benchmark Qwen3-Next-80B-A3B TP4 projection shapes.
"""
import torch
import time
from vllm.platforms import current_platform

device = torch.device("xpu")
DUMP_PATH = "/home/edgeai/applications.ai.gpu.vllm-xpu/xpu_fp8_assert_dump_1778481474998.pt"


def test_correctness_basic():
    """Basic correctness: scale=1, compare dequant(fp8)*input vs kernel output."""
    from custom_esimd_kernels_vllm import esimd_gemv_fp8_pern

    print("\n--- Correctness (scale=1) ---")
    for N, K in [(3072, 2560), (2560, 2048), (20480, 2560), (2560, 10240),
                 (256, 2560), (2048, 128), (3072, 2048), (16, 2048)]:
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
    for N, K in [(3072, 2560), (2560, 2048), (20480, 2560), (2560, 10240), (128, 2048), (2560,10752)]:
        weight_ref = torch.randn(N, K, dtype=torch.float16, device=device) * 0.1
        weight_fp8 = weight_ref.to(torch.float8_e4m3fn)
        scale = torch.randn(N, dtype=torch.float16, device=device) * 0.1
        #print(scale.shape, scale.dtype)
        input_t = torch.rand(1, K, dtype=torch.float16, device=device) * 0.1
        output = torch.zeros(1, N, dtype=torch.float16, device=device)

        esimd_gemv_fp8_pern(input_t, weight_fp8, scale, output, N, K)

        weight_dequant = weight_fp8.to(torch.float16)
        ref = (input_t.float() @ weight_dequant.float().T) * scale.float().unsqueeze(0)
        output = output.to(torch.bfloat16)
        max_diff = (output.float() - ref.float()).abs().max().item()
        ref_max = ref.float().abs().max().item()
        rel_err = (max_diff / ref_max) if ref_max > 1e-6 else 0
        ok = max_diff < 0.5 or rel_err < 0.05
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] N={N:5d} K={K:5d} (with scale)  max_diff={max_diff:.4f}  rel={rel_err:.4f}")
        assert ok, f"Correctness failed for N={N}, K={K} with scale"



def test_pern_replay_from_assert_dump():
    """Replay a captured failure case for esimd_gemv_fp8_pern."""
    from custom_esimd_kernels_vllm import esimd_gemv_fp8_pern

    dump = torch.load(DUMP_PATH, map_location="cpu")
    input_t = dump["x"].to(device=device).to(torch.float16)
    weight_fp8 = dump["weight"].to(device=device)
    scale = dump["weight_scale"].reshape(-1).to(device=device).to(torch.float16)
    ref = dump["output"].to(device=device)

    N, K = weight_fp8.shape
    if scale.numel() == 1:
        scale = scale.repeat(N)

    output = torch.zeros_like(ref, device=device).to(torch.float16)
    print("input_t shape:", input_t.shape, "dtype:", input_t.dtype)
    print("input_t min/max:", input_t.float().min().item(), input_t.float().max().item())
    print("weight_fp8 shape:", weight_fp8.shape, "dtype:", weight_fp8.dtype)
    print(
        "weight_fp8 min/max:",
        weight_fp8.float().min().item(),
        weight_fp8.float().max().item(),
    )
    print("scale shape:", scale.shape, "dtype:", scale.dtype)
    print("output shape:", output.shape, "dtype:", output.dtype)
    print("ref shape:", ref.shape, "dtype:", ref.dtype)
    esimd_gemv_fp8_pern(input_t, weight_fp8, scale, output, N, K)

    max_diff = (output.float() - ref.float()).abs().max().item()
    ref_max = ref.float().abs().max().item()
    rel_err = (max_diff / ref_max) if ref_max > 1e-6 else 0
    ok = max_diff < 0.5 or rel_err < 0.05
    status = "PASS" if ok else "FAIL"

    print(
        f"  [{status}] replay dump N={N:5d} K={K:5d} "
        f"scale_numel={scale.numel()} max_diff={max_diff:.4f} rel={rel_err:.4f}"
    )
    assert ok, (
        "Replay from assert dump failed for esimd_gemv_fp8_pern: "
        f"max_diff={max_diff:.4f}, rel_err={rel_err:.4f}"
    )


def benchmark_shapes():
    """Benchmark gemma-4-E4B-it shape."""
    from custom_esimd_kernels_vllm import esimd_gemv_fp8_pern

    shapes = [
        ("qkv_proj",     3072, 2560),
        ("Attn o_proj",  2560, 2048),
        ("gate_up_proj", 20480, 2560),
        ("down_proj",    2560, 10240),
        ("per_layer_input_gate",  256, 2560),
        ("per_layer_input_gate_out",     2560, 256),
    ]

    TARGET_BW = 112.0  # GB/s PTL

    print(f"\n{'Shape':<30} {'N':>6} {'K':>6} {'KB':>7} | {'GB/s':>8} {'BW%':>7} {'us':>8}")
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

        print(f"{name:<30} {N:>6} {K:>6} {total_bytes//1024:>6}K | {bw:>7.1f} {bw_pct:>6.1f}% {us:>7.2f}")


def benchmark_best_vl_ks():
    """Search the best vl/ks configuration for each benchmark shape."""
    from custom_esimd_kernels_vllm import esimd_gemv_fp8_pern

    shapes = [
        ("qkv_proj", 3072, 2560),
        ("Attn o_proj", 2560, 2048),
        ("gate_up_proj", 20480, 2560),
        ("down_proj", 2560, 10240),
        ("per_layer_input_gate", 256, 2560),
        ("per_layer_input_gate_out", 2560, 256),
    ]
    candidates = [
        (512, 1),
        (512, 2),
        (512, 5),
        (256, 1),
        (256, 2),
        (256, 4),
        (256, 5),
        (128, 1),
        (128, 2),
        (128, 4),
        (128, 5),
        (128, 10),
    ]

    print(f"\n{'Shape':<30} {'N':>6} {'K':>6} | {'Best':>9} {'Auto us':>10} {'Best us':>10} {'Speedup':>8}")
    print("-" * 86)

    for name, N, K in shapes:
        weight_ref = torch.randn(N, K, dtype=torch.float16, device=device) * 0.1
        weight_fp8 = weight_ref.to(torch.float8_e4m3fn)
        scale = torch.randn(N, dtype=torch.float16, device=device) * 0.1
        input_t = torch.randn(1, K, dtype=torch.float16, device=device) * 0.1
        output = torch.zeros(1, N, dtype=torch.float16, device=device)

        total_bytes = K * 2 + N * K + N * 2 + N * 2
        ni = 4000 
        valid_candidates = [
            (vl, ks)
            for vl, ks in candidates
            if K % ks == 0 and (K // ks) >= vl and (K // ks) % vl == 0
        ]

        for _ in range(10):
            esimd_gemv_fp8_pern(input_t, weight_fp8, scale, output, N, K, vl=0, ks=0)
        torch.xpu.synchronize()

        t0 = time.perf_counter()
        for _ in range(ni):
            esimd_gemv_fp8_pern(input_t, weight_fp8, scale, output, N, K, vl=0, ks=0)
        torch.xpu.synchronize()
        auto_us = (time.perf_counter() - t0) / ni * 1e6

        best_vl = 0
        best_ks = 0
        best_us = float("inf")
        for vl, ks in valid_candidates:
            for _ in range(10):
                esimd_gemv_fp8_pern(input_t, weight_fp8, scale, output, N, K, vl=vl, ks=ks)
            torch.xpu.synchronize()

            t0 = time.perf_counter()
            for _ in range(ni):
                esimd_gemv_fp8_pern(input_t, weight_fp8, scale, output, N, K, vl=vl, ks=ks)
            torch.xpu.synchronize()
            tuned_us = (time.perf_counter() - t0) / ni * 1e6

            if tuned_us < best_us:
                best_us = tuned_us
                best_vl = vl
                best_ks = ks

        speedup = auto_us / best_us if best_us > 0 else 0.0
        print(f"{name:<30} {N:>6} {K:>6} | {best_vl:>3}/{best_ks:<5} {auto_us:>9.2f} {best_us:>9.2f} {speedup:>7.2f}x")

def test_esimd_vs_vllm():
    from custom_esimd_kernels_vllm import (
        esimd_gemv_fp8_pern, 
    )

    TARGET_BW = 112.0  # GB/s PTL

    print(f"\n{'Case':<30} {'Config':>20} | {'Indiv us':>10} {'vllm us':>10} {'Speedup':>8}")
    print("-" * 78)

    def make_tensors(N, K):
        w = (torch.randn(N, K, dtype=torch.float16, device=device) * 0.1).to(torch.float8_e4m3fn)
        s = torch.randn(N, dtype=torch.float16, device=device) * 0.1
        o = torch.zeros(1, N, dtype=torch.float16, device=device)
        return w, s, o

    shapes = [
        ("qkv_proj",     3072, 2560),
        ("Attn o_proj",  2560, 2048),
        ("gate_up_proj", 20480, 2560),
        ("down_proj",    2560, 10240),
        ("per_layer_input_gate",  256, 2560),
        ("per_layer_input_gate_out",     2560, 256),
    ]

    ni = 2000

    for name, N, K in shapes:
        input_t = torch.randn(1, K, dtype=torch.float16, device=device) * 0.1
        w0, s0, o0 = make_tensors(N, K)
        config = f"N={N} K={K}"

        # Warmup + bench individual
        for _ in range(10):
            esimd_gemv_fp8_pern(input_t, w0, s0, o0, N, K)
            # esimd_gemv_fp8_pern(input_t, w1, s1, o1, shapes[1][0], K)
            torch.xpu.synchronize()
        t0 = time.perf_counter()
        for _ in range(ni):
            esimd_gemv_fp8_pern(input_t, w0, s0, o0, N, K)
            torch.xpu.synchronize()
            # esimd_gemv_fp8_pern(input_t, w1, s1, o1, shapes[1][0], K)
        torch.xpu.synchronize()
        indiv_us = (time.perf_counter() - t0) / ni * 1e6

        # Warmup + bench individual vllm
        for _ in range(10):
            torch.ops._xpu_C.fp8_gemm_w8a16(input_t, w0.t(), s0, None)
            torch.xpu.synchronize()
            # esimd_gemv_fp8_pern(input_t, w1, s1, o1, shapes[1][0], K)
        
        t0 = time.perf_counter()
        for _ in range(ni):
            torch.ops._xpu_C.fp8_gemm_w8a16(input_t, w0.t(), s0, None)
            torch.xpu.synchronize()
            # esimd_gemv_fp8_pern(input_t, w1, s1, o1, shapes[1][0], K)
        
        vllm_us = (time.perf_counter() - t0) / ni * 1e6

        speedup = vllm_us / indiv_us if indiv_us > 0 else 0
        print(f"{name:<30} {config:>20} | {indiv_us:>9.2f} {vllm_us:>9.2f} {speedup:>7.2f}x")


def benchmark_fused():
    """Benchmark fused vs sum-of-individual latencies for target Qwen3 shapes."""
    from custom_esimd_kernels_vllm import (
        esimd_gemv_fp8_pern, 
    )

    TARGET_BW = 112.0  # GB/s PTL

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
        ("qkv_proj",     3072, 2560),
        ("Attn o_proj",  2560, 2048),
        ("gate_up_proj", 20480, 2560),
        ("down_proj",    2560, 10240),
        ("per_layer_input_gate",  256, 2560),
        ("per_layer_input_gate_out",     2560, 256),
    ]

    TARGET_BW = 450.0

    print(f"\n{'Shape':<30} {'N':>6} {'K':>6} | {'E4M3 us':>9} {'E5M2 us':>9} {'E4M3 GB/s':>10} {'E5M2 GB/s':>10}")
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
        print(f"{name:<30} {N:>6} {K:>6} | {e4_us:>8.2f} {e5_us:>8.2f} {e4_bw:>9.1f} {e5_bw:>9.1f}")


if __name__ == "__main__":
    print("=" * 60)
    print("custom-esimd-kernels-vllm: GEMV FP8 Tests")
    print("=" * 60)

    # E4M3 per-N scale tests
    # test_correctness_basic()
    # test_correctness_with_scale()
    # test_pern_replay_from_assert_dump()
    # test_esimd_vs_vllm()
    benchmark_best_vl_ks()
    # E4M3 per-tensor scale tests
    # test_pert_correctness()
    # test_pert_vs_pern()

    # # E5M2 tests
    # test_e5m2_correctness_pern()
    # test_e5m2_correctness_pert()

    # # Performance
    # print("\n--- Performance Benchmark (unfused per-N, E4M3) ---")
    # benchmark_shapes()


    # print("\n--- Performance Benchmark (E4M3 vs E5M2) ---")
    # benchmark_e5m2()

    # print("\n" + "=" * 60)
    # print("ALL TESTS PASSED")
    # print("=" * 60)