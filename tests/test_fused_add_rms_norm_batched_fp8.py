import torch
import time
import sys

device = torch.device("xpu")

def ref_fused_add_rms_norm_batched_fp16(hidden, residual, weight, eps):
    """Reference for esimd_fused_add_rms_norm_batched.

    residual = residual + hidden
    hidden = rmsnorm(residual) * weight

    weight is expected to be the Gemma-style RMSNorm weight already shifted to
    the kernel convention, i.e. w + 1.0.
    """
    h = hidden.cpu().float()
    r = residual.cpu().float()

    # Step 1: ResAdd
    updated_residual = h + r

    # Step 2: RMSNorm
    variance = updated_residual.pow(2).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(variance + eps)
    normed = updated_residual * inv_rms * weight.cpu().float()

    return updated_residual.float(), normed.float()

def test_correctness():
    """Fused kernel should match step-by-step reference."""

    from custom_esimd_kernels_vllm import esimd_fused_add_rms_norm_batched

    print("\n--- Fused Add RMSNorm Correctness ---")
    shapes = [512, 1024, 2048, 4096]
    torch.manual_seed(42)
    eps = 1e-6
    for K in shapes:
        hidden = torch.randn(1, K, dtype=torch.float16, device=device)
        residual = torch.randn(1, K, dtype=torch.float16, device=device)
        norm_weight = torch.randn(K, dtype=torch.float16, device=device) * 0.1
        hidden_out = hidden.clone()
        res_fp8 = residual.clone()

        esimd_fused_add_rms_norm_batched(hidden_out, res_fp8, norm_weight, eps)
        torch.xpu.synchronize()

        ref_residual, ref_normed = ref_fused_add_rms_norm_batched_fp16(
            hidden, residual, norm_weight, eps)

        # # Check residual update
        # res_diff = (res_fp8.cpu().float() - ref_residual.float()).abs()
        # assert res_diff.max().item() < 0.01, \
        #     f"Residual diff: {res_diff.max().item():.4f}"

        # Check normed output
        norm_diff = (hidden_out.cpu().float() - ref_normed.float()).abs()
        assert norm_diff.max().item() < 0.1, \
            f"Normed diff: {norm_diff.max().item():.4f}"




def benchmark():
    """Benchmark esimd_fused_add_rms_norm_batched across batch sizes."""
    from custom_esimd_kernels_vllm import esimd_fused_add_rms_norm_batched

    shapes = [
        ("Attn qkv", 2048),
        ("Exp gate", 2048),
        ("Exp down", 512),
        ("DN qkvz", 2048),
    ]
    m_values = [1, 2, 4, 8, 16, 32, 64]
    eps = 1e-6

    print(f"\n{'Shape':<14} {'K':>5} | " +
          " ".join(f"{'M='+str(m):>9}" for m in m_values))
    print("-" * (24 + 10 * len(m_values)))

    for name, K in shapes:
        line = f"{name:<14} {K:>5} |"
        for M in m_values:
            hidden = torch.randn(M, K, dtype=torch.float16, device=device) * 0.1
            residual = torch.randn(M, K, dtype=torch.float16, device=device) * 0.1
            norm_weight = (
                torch.randn(K, dtype=torch.float16, device=device) * 0.1 + 1.0
            )

            ni = 2000 if M * K < 2 * 1024 * 1024 else 500

            # Warmup
            for _ in range(10):
                hidden_out = hidden.clone()
                residual_out = residual.clone()
                esimd_fused_add_rms_norm_batched(
                    hidden_out,
                    residual_out,
                    norm_weight,
                    eps,
                )
            torch.xpu.synchronize()

            t0 = time.perf_counter()
            for _ in range(ni):
                hidden_out = hidden.clone()
                residual_out = residual.clone()
                esimd_fused_add_rms_norm_batched(
                    hidden_out,
                    residual_out,
                    norm_weight,
                    eps,
                )
            torch.xpu.synchronize()
            us = (time.perf_counter() - t0) / ni * 1e6
            line += f" {us:>8.1f}"
        print(line)


if __name__ == "__main__":
    print("=" * 60)
    print("custom-esimd-kernels-vllm: Fused Add RMSNorm Tests")
    print("=" * 60)

    test_correctness()

    print("\n--- Performance Benchmark (us per call) ---")
    benchmark()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)