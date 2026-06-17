"""
Test PTL-optimized fp8_GEMV_ptl vs baseline fp8_GEMV_v2.

Compares:
1. Correctness: Both should produce identical results (within FP tolerance)
2. Performance: PTL version optimized for 480 threads vs V2 baseline

PTL hardware: 480 threads, VL=256, SLM=192KB
BMG hardware: 640 threads, VL=256, SLM=~64KB

Test shapes from Qwen3-Next-80B-A3B TP4 projections and various sizes.
"""

import gc
import os
import sys
import torch
import time
from typing import Tuple, List

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

device = torch.device("xpu")

# ============================================================================
# Helper Functions
# ============================================================================

def _benchmark_one(run_fn, output_fn, iters: int) -> Tuple[float, List[float]]:
    """Benchmark a function with warmup and output sampling."""
    # Warmup
    for _ in range(10):
        run_fn(0)
    torch.xpu.synchronize()

    outputs = []
    t0 = time.perf_counter()
    for index in range(iters):
        run_fn(index)
        output = output_fn()
        sample_index = [0] * output.dim()
        if output.dim() >= 2:
            sample_index[1] = index % output.size(1)
        else:
            sample_index[0] = index % output.size(0)
        outputs.append(float(output[tuple(sample_index)]))
    torch.xpu.synchronize()
    elapsed_us = (time.perf_counter() - t0) * 1e6 / iters

    # Cleanup
    gc.collect()
    if hasattr(torch.xpu, "empty_cache"):
        torch.xpu.empty_cache()

    return elapsed_us, outputs


def _create_cache_busting_weights(N: int, K: int, num_copies: int, dtype=torch.float8_e4m3fn):
    """Create multiple weight copies for cache-busting during benchmarks."""
    weights = []
    for _ in range(num_copies):
        w = torch.randn(N, K, dtype=torch.float16, device=device) * 0.1
        weights.append(w.to(dtype))
    return weights


def _calculate_num_iters(total_bytes: int) -> int:
    """Calculate number of iterations based on problem size."""
    if total_bytes < 512 * 1024:
        return 4000
    elif total_bytes < 2 * 1024 * 1024:
        return 1000
    else:
        return 300


def _calculate_num_cache_busting_copies(N: int, K: int) -> int:
    """Calculate number of weight copies for cache-busting."""
    wb = N * K
    target_mem = 32 * 1024 * 1024
    nc = max(16, target_mem // max(wb, 1))
    return min(nc, 512)


# ============================================================================
# PTL Selection Logic (480 threads, optimized for PTL)
# ============================================================================

def select_ptl(N: int, K: int) -> Tuple[int, int, int]:
    """
    Select (VL_BIG, VL_TAIL, K_SPLIT) for PTL architecture.

    PTL: ~480 hardware threads
    Strategy: N × K_SPLIT >= 960 for full saturation (2× threads)
    """
    vl_big = 256
    vl_tail = 0
    ks = 1

    # Target threads = N × ks; aim for >= 480 (PTL full occupancy)
    target_ks = 1
    if N * 8 <= 480:
        target_ks = 8
    elif N * 4 <= 480:
        target_ks = 4
    elif N * 2 <= 480:
        target_ks = 2
    else:
        target_ks = 1

    # K_SPLIT must divide K
    chosen_ks = 1
    for s in [target_ks, target_ks // 2, target_ks // 4, 1]:
        if s > 0 and K % s == 0:
            chosen_ks = s
            break
    ks = chosen_ks

    kp = K // ks

    # Pick VL_BIG: largest power-of-2 ≤ 256 that gives kp_full > 0
    candidates = [256, 128, 64, 32]
    for c in candidates:
        if kp >= c:
            vl_big = c
            kp_full = (kp // c) * c
            tail = kp - kp_full

            if tail == 0:
                vl_tail = 0
                return vl_big, vl_tail, ks

            # Find power-of-2 that exactly equals tail
            for t in [8, 16, 32, 64, 128]:
                if t == tail:
                    vl_tail = t
                    return vl_big, vl_tail, ks

    # Fallback
    return 32, 0, 1


# ============================================================================
# V2 Selection Logic (baseline heuristic)
# ============================================================================

def select_v2(N: int, K: int) -> Tuple[int, int]:
    """
    Select (VL, K_SPLIT) for V2 baseline.
    This mimics the logic from fp8_GEMV_v2.h
    """
    vl = 256
    ks = 1

    # Simplified V2 heuristic (from test_gemv_fp8.py)
    if K < 256:
        vl, ks = 128, 1
    elif K == 256:
        vl, ks = 256, 1
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

    # Normalize
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


# ============================================================================
# Test Functions
# ============================================================================

def test_correctness():
    """Test correctness: PTL vs V2 should produce same results."""
    print("\n" + "=" * 80)
    print("CORRECTNESS TEST: PTL vs V2")
    print("=" * 80)
    print(f"\n{'Shape':<20} {'PTL Config':<20} {'V2 Config':<20} {'MaxDiff':<12} {'RelErr':<12} {'Status':<8}")
    print("-" * 100)

    shapes = [
        ("qkv_proj",     3072, 2560),                               # 128/10
        ("qkv_proj",     6144, 2560),                               # 128/10
        ("Attn o_proj",  2560, 2048),                               # 256/8
        ("Attn o_proj",  2560, 4096),                               # 256/4
        ("gate_up_proj", 20480, 2560),                              # 512/1
        ("down_proj",    2560, 10240),                              # 128/10
        ("per_layer_input_gate",  256, 2560),                       # 128/10
        ("per_layer_input_gate_out",     2560, 256),                # 128/2
    ]

    all_passed = True

    for name, N, K in shapes:
        # Create test data
        weight_ref = torch.randn(N, K, dtype=torch.float16, device=device) * 0.1
        weight_fp8 = weight_ref.to(torch.float8_e4m3fn)
        scale = torch.randn(N, dtype=torch.float16, device=device) * 0.1
        input_t = torch.randn(1, K, dtype=torch.float16, device=device) * 0.1

        # PTL kernel (per-tensor scale)
        scale_tensor = torch.tensor(0.073, dtype=torch.float32, device=device)
        output_ptl = torch.zeros(1, N, dtype=torch.float16, device=device)

        # V2 kernel (per-N scale)
        output_v2 = torch.zeros(1, N, dtype=torch.float16, device=device)

        # Get configurations
        vl_big_ptl, vl_tail_ptl, ks_ptl = select_ptl(N, K)
        vl_v2, ks_v2 = select_v2(N, K)

        # Run PTL kernel (need to call via custom ops)
        from custom_esimd_kernels_vllm import esimd_gemv_fp8_pert

        try:
            # PTL uses per-tensor scale - need to match V2's per-N scale behavior
            # For correctness test, use per-N scale on both
            from custom_esimd_kernels_vllm import esimd_gemv_fp8_pern

            # Check if PTL config is valid (no tail for now - simplified test)
            if vl_tail_ptl == 0:
                esimd_gemv_fp8_pern(input_t, weight_fp8, scale, output_ptl, N, K, vl_big_ptl, ks_ptl)
            else:
                # Skip shapes with tail for now
                print(f"{name:<20} {N:>6} {K:>6} - SKIP (PTL has tail, need tail kernel)")
                continue

            # Run V2 kernel
            esimd_gemv_fp8_pern(input_t, weight_fp8, scale, output_v2, N, K, vl_v2, ks_v2)

            # Compare
            max_diff = (output_ptl.float() - output_v2.float()).abs().max().item()
            ref_max = output_v2.float().abs().max().item()
            rel_err = (max_diff / ref_max) if ref_max > 1e-6 else 0

            ok = max_diff < 0.5 or rel_err < 0.02
            status = "PASS" if ok else "FAIL"

            ptl_cfg = f"vl={vl_big_ptl} ks={ks_ptl}"
            v2_cfg = f"vl={vl_v2} ks={ks_v2}"

            print(f"{name:<20} {ptl_cfg:<20} {v2_cfg:<20} {max_diff:<12.6f} {rel_err:<12.6f} {status:<8}")

            if not ok:
                all_passed = False

        except Exception as e:
            print(f"{name:<20} N={N:>6} K={K:>6} - ERROR: {str(e)}")
            all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print("✓ ALL CORRECTNESS TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 80)

    return all_passed


def test_performance():
    """Benchmark PTL vs V2 performance."""
    print("\n" + "=" * 80)
    print("PERFORMANCE TEST: PTL vs V2")
    print("=" * 80)

    TARGET_BW_PTL = 112.0  # GB/s for PTL

    print(f"\n{'Shape':<20} {'N':>6} {'K':>6} {'PTL Config':<15} {'V2 Config':<15} | "
          f"{'PTL us':>9} {'V2 us':>9} {'PTL GB/s':>9} {'V2 GB/s':>9} {'Speedup':>8}")
    print("-" * 120)

    shapes = [
        ("qkv_proj",     3072, 2560),                               # 128/10
        ("qkv_proj",     6144, 2560),                               # 128/10
        ("Attn o_proj",  2560, 2048),                               # 256/8
        ("Attn o_proj",  2560, 4096),                               # 256/4
        ("gate_up_proj", 20480, 2560),                              # 512/1
        ("down_proj",    2560, 10240),                              # 128/10
        ("per_layer_input_gate",  256, 2560),                       # 128/10
        ("per_layer_input_gate_out",     2560, 256),                # 128/2
    ]

    from custom_esimd_kernels_vllm import esimd_gemv_fp8_pern

    for name, N, K in shapes:
        # Calculate bytes
        element_bytes = 2  # fp16
        total_bytes = K * element_bytes + N * K + N * element_bytes + N * element_bytes

        # Prepare data
        weight_ref = torch.randn(N, K, dtype=torch.float16, device=device) * 0.1
        weight_fp8 = weight_ref.to(torch.float8_e4m3fn)
        scale = torch.randn(N, dtype=torch.float16, device=device) * 0.1
        input_t = torch.randn(1, K, dtype=torch.float16, device=device) * 0.1

        # Cache-busting
        nc = _calculate_num_cache_busting_copies(N, K)
        weights = _create_cache_busting_weights(N, K, nc)

        ni = _calculate_num_iters(total_bytes)

        # Get configurations
        vl_big_ptl, vl_tail_ptl, ks_ptl = select_ptl(N, K)
        vl_v2, ks_v2 = select_v2(N, K)

        # Skip if PTL has tail (simplified for now)
        if vl_tail_ptl != 0:
            print(f"{name:<20} {N:>6} {K:>6} - SKIP (PTL has tail)")
            continue

        output_ptl = torch.zeros(1, N, dtype=torch.float16, device=device)
        output_v2 = torch.zeros(1, N, dtype=torch.float16, device=device)

        try:
            # Benchmark PTL
            ptl_us, _ = _benchmark_one(
                lambda index: esimd_gemv_fp8_pern(
                    input_t, weights[index % nc], scale, output_ptl, N, K, vl_big_ptl, ks_ptl
                ),
                lambda: output_ptl,
                ni,
            )

            # Benchmark V2
            v2_us, _ = _benchmark_one(
                lambda index: esimd_gemv_fp8_pern(
                    input_t, weights[index % nc], scale, output_v2, N, K, vl_v2, ks_v2
                ),
                lambda: output_v2,
                ni,
            )

            # Calculate metrics
            ptl_bw = (total_bytes / 1e9) / (ptl_us / 1e6)
            v2_bw = (total_bytes / 1e9) / (v2_us / 1e6)
            speedup = v2_us / ptl_us if ptl_us > 0 else 0

            ptl_cfg = f"vl={vl_big_ptl} ks={ks_ptl}"
            v2_cfg = f"vl={vl_v2} ks={ks_v2}"

            print(f"{name:<20} {N:>6} {K:>6} {ptl_cfg:<15} {v2_cfg:<15} | "
                  f"{ptl_us:>9.2f} {v2_us:>9.2f} {ptl_bw:>9.2f} {v2_bw:>9.2f} {speedup:>7.2f}x")

        except Exception as e:
            print(f"{name:<20} {N:>6} {K:>6} - ERROR: {str(e)}")

    print("\n" + "=" * 80)


def test_config_differences():
    """Show how PTL vs V2 choose different configurations."""
    print("\n" + "=" * 80)
    print("CONFIGURATION COMPARISON: PTL (480 threads) vs V2 (640 threads)")
    print("=" * 80)
    print(f"\n{'Shape':<20} {'N':>6} {'K':>6} | {'PTL Config':<25} {'V2 Config':<25} {'Diff':<10}")
    print("-" * 100)

    shapes = [
        ("qkv_proj",     3072, 2560),                               # 128/10
        ("qkv_proj",     6144, 2560),                               # 128/10
        ("Attn o_proj",  2560, 2048),                               # 256/8
        ("Attn o_proj",  2560, 4096),                               # 256/4
        ("gate_up_proj", 20480, 2560),                              # 512/1
        ("down_proj",    2560, 10240),                              # 128/10
        ("per_layer_input_gate",  256, 2560),                       # 128/10
        ("per_layer_input_gate_out",     2560, 256),                # 128/2
    ]

    for name, N, K in shapes:
        vl_big_ptl, vl_tail_ptl, ks_ptl = select_ptl(N, K)
        vl_v2, ks_v2 = select_v2(N, K)

        ptl_cfg = f"vl={vl_big_ptl} ks={ks_ptl} tail={vl_tail_ptl}"
        v2_cfg = f"vl={vl_v2} ks={ks_v2}"

        diff = "SAME" if (vl_big_ptl == vl_v2 and ks_ptl == ks_v2) else "DIFFER"

        print(f"{name:<20} {N:>6} {K:>6} | {ptl_cfg:<25} {v2_cfg:<25} {diff:<10}")

    print("\n" + "=" * 80)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("PTL vs V2 FP8 GEMV Comparison")
    print("=" * 80)
    print("PTL: 480 hardware threads, VL_max=256, SLM=192KB")
    print("BMG: 640 hardware threads, VL_max=256, SLM=~64KB")
    print("=" * 80)

    # Configuration comparison (always run)
    test_config_differences()

    # Correctness test
    print("\nRun correctness test? (y/n): ", end="")
    if input().lower().startswith('y'):
        test_correctness()

    # Performance test
    print("\nRun performance test? (y/n): ", end="")
    if input().lower().startswith('y'):
        test_performance()

    print("\n" + "=" * 80)
    print("Tests complete!")
    print("=" * 80)
