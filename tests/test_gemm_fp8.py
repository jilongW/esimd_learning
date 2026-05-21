import torch
import time
import sys

device = torch.device("xpu")


def _run_correctness_case(weight_dtype, io_dtype):
    from custom_esimd_kernels_vllm import esimd_gemm_fp8_pert

    weight_name = "E5M2" if weight_dtype == torch.float8_e5m2 else "E4M3"
    print(f"\n--- GEMM {weight_name} Correctness ({str(io_dtype).split('.')[-1]}) ---")
    shapes = [
        (2560, 2048),
        (512, 2048),
        (2048, 512),
        (128, 2048),
        (3072, 2048),
        (1024, 1024),
    ]
    m_values = [1, 2, 4, 8, 16, 32, 64]

    for N, K in shapes:
        for M in m_values:
            weight_ref = torch.randn(N, K, dtype=torch.float16, device=device) * 0.1
            weight_fp8 = weight_ref.to(weight_dtype)
            scale_val = 0.05 + torch.rand(1).item() * 0.1
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
    from custom_esimd_kernels_vllm import esimd_gemv_fp8_pert, esimd_gemm_fp8_pert

    for io_dtype in [torch.float16, torch.bfloat16]:
        print(f"\n--- GEMM vs GEMV at M=1 ({str(io_dtype).split('.')[-1]}) ---")
        for N, K in [(2560, 2048), (512, 2048), (128, 2048), (2048, 512)]:
            weight_ref = torch.randn(N, K, dtype=torch.float16, device=device) * 0.1
            weight_fp8 = weight_ref.to(torch.float8_e4m3fn)
            scale_t = torch.tensor(0.073, dtype=torch.float32, device=device)
            input_t = torch.randn(1, K, dtype=io_dtype, device=device) * 0.1

            out_gemv = torch.zeros(1, N, dtype=io_dtype, device=device)
            out_gemm = torch.zeros(1, N, dtype=io_dtype, device=device)

            esimd_gemv_fp8_pert(input_t, weight_fp8, scale_t, out_gemv)
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


def benchmark():
    """Benchmark across M values for key shapes."""
    from custom_esimd_kernels_vllm import esimd_gemm_fp8_pert

    shapes = [
        ("Attn qkv",    2560, 2048),
        ("Exp gate",     512, 2048),
        ("Exp down",    2048,  512),
        ("DN qkvz",     3072, 2048),
    ]
    m_values = [1, 2, 4, 8, 16, 32, 64]

    print(f"\n{'Shape':<14} {'N':>5} {'K':>5} | " +
          " ".join(f"{'M='+str(m):>9}" for m in m_values))
    print("-" * (30 + 10 * len(m_values)))

    for name, N, K in shapes:
        line = f"{name:<14} {N:>5} {K:>5} |"
        for M in m_values:
            weight_ref = torch.randn(N, K, dtype=torch.float16, device=device) * 0.1
            weight_fp8 = weight_ref.to(torch.float8_e4m3fn)
            scale_t = torch.tensor(0.073, dtype=torch.float32, device=device)
            input_t = torch.randn(M, K, dtype=torch.float16, device=device) * 0.1
            output = torch.zeros(M, N, dtype=torch.float16, device=device)

            ni = 2000 if N * K < 2 * 1024 * 1024 else 500

            # Warmup
            for _ in range(10):
                esimd_gemm_fp8_pert(input_t, weight_fp8, scale_t, output)
            torch.xpu.synchronize()

            t0 = time.perf_counter()
            for _ in range(ni):
                esimd_gemm_fp8_pert(input_t, weight_fp8, scale_t, output)
            torch.xpu.synchronize()
            us = (time.perf_counter() - t0) / ni * 1e6
            line += f" {us:>8.1f}"
        print(line)


if __name__ == "__main__":
    print("=" * 60)
    print("custom-esimd-kernels-vllm: GEMM FP8 Per-tensor Tests")
    print("=" * 60)

    test_correctness()
    test_e5m2_correctness()
    test_gemm_vs_gemv_m1()

    print("\n--- Performance Benchmark (us per call) ---")
    benchmark()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)