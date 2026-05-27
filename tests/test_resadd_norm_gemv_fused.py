import torch

from custom_esimd_kernels_vllm import (
    esimd_fused_add_rms_norm_batched,
    esimd_gemm_fp8_pert,
    esimd_resadd_norm_gemv_fp8_pert,
)

DEVICE = torch.device("xpu")
WARMUP_ITERS = 20
BENCHMARK_ITERS = 200


def print_topk_output_errors(name, output, ref_output, topk=10):
    out_cpu = output.cpu().float().reshape(-1)
    ref_cpu = ref_output.reshape(-1)
    abs_diff = (out_cpu - ref_cpu).abs()
    top_vals, top_idx = torch.topk(abs_diff, k=min(topk, abs_diff.numel()))

    print(f"{name} top-{top_vals.numel()} output abs diff:")
    for rank, (diff_val, idx_val) in enumerate(zip(top_vals.tolist(), top_idx.tolist()), start=1):
        out_val = out_cpu[idx_val].item()
        ref_val = ref_cpu[idx_val].item()
        rel_to_ref = diff_val / max(abs(ref_val), 1e-6)
        print(
            f"  #{rank:02d} idx={idx_val:5d} "
            f"out={out_val:+.6f} ref={ref_val:+.6f} "
            f"abs_diff={diff_val:.6f} rel_to_ref={rel_to_ref:.6f}"
        )


def ref_resadd_norm_gemv_fp8(hidden, residual, norm_weight, weight_fp8, scale, eps):
    h = hidden.cpu().float()
    r = residual.cpu().float()

    updated_residual = h + r
    variance = updated_residual.pow(2).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(variance + eps)
    normed = updated_residual * inv_rms * norm_weight.cpu().float()

    scale_cpu = scale.cpu().float().reshape(-1)
    weight_fp8_cpu = weight_fp8.cpu().float()
    if scale_cpu.numel() == 1:
        dequant_weight = weight_fp8_cpu * float(scale_cpu[0])
    else:
        split_n = (weight_fp8_cpu.shape[0] + 1) // 2
        dequant_weight = weight_fp8_cpu.clone()
        dequant_weight[:split_n] *= float(scale_cpu[0])
        dequant_weight[split_n:] *= float(scale_cpu[1])

    result = normed @ dequant_weight.T
    return result, updated_residual.half(), normed.half()


def benchmark_xpu_callable(fn, warmup_iters=WARMUP_ITERS, benchmark_iters=BENCHMARK_ITERS):
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


def benchmark_fused_vs_split():
    torch.manual_seed(42)
    eps = 1e-6
    shapes = [
        ("gate_up_proj", 20480, 2560),
    ]

    for name, N, K in shapes:
        hidden = torch.randn(1, K, dtype=torch.float16, device=DEVICE) * 0.1
        residual = hidden.clone()
        norm_weight = torch.randn(K, dtype=torch.float16, device=DEVICE) * 0.1
        weight_fp16 = torch.randn(N, K, dtype=torch.float16, device=DEVICE) * 0.1
        weight_fp8 = weight_fp16.to(torch.float8_e4m3fn)
        scale = torch.tensor([0.0008, 0.0008], dtype=torch.float32, device=DEVICE)

        fused_output = torch.empty(1, N, dtype=torch.float16, device=DEVICE)
        fused_normed_out = torch.empty(1, K, dtype=torch.float16, device=DEVICE)
        fused_residual = residual.clone()

        split_normed = hidden.clone()
        split_residual = residual.clone()
        split_output = torch.empty(1, N, dtype=torch.float16, device=DEVICE)

        def run_fused():
            fused_residual.copy_(residual)
            esimd_resadd_norm_gemv_fp8_pert(
                hidden,
                fused_residual,
                norm_weight,
                weight_fp8,
                scale,
                fused_output,
                fused_normed_out,
                eps,
            )

        def run_split():
            split_normed.copy_(hidden)
            split_residual.copy_(residual)
            esimd_fused_add_rms_norm_batched(
                split_normed,
                split_residual,
                norm_weight,
                eps,
            )
            esimd_gemm_fp8_pert(
                split_normed,
                weight_fp8,
                scale[:1],
                split_output,
            )

        fused_latency_us = benchmark_xpu_callable(run_fused)
        split_latency_us = benchmark_xpu_callable(run_split)
        speedup = split_latency_us / fused_latency_us

        print(
            f"{name} benchmark: fused={fused_latency_us:.3f} us, "
            f"split={split_latency_us:.3f} us, speedup={speedup:.3f}x"
        )


def test_correctness():
    torch.manual_seed(42)
    eps = 1e-6
    shapes = [
        ("gate_up_proj", 20480, 2560),
    ]

    for name, N, K in shapes:
        hidden = torch.randn(1, K, dtype=torch.float16, device=DEVICE) * 0.1
        residual = hidden.clone()
        norm_weight = torch.randn(K, dtype=torch.float16, device=DEVICE) * 0.1
        weight_fp16 = torch.randn(N, K, dtype=torch.float16, device=DEVICE) * 0.1
        weight_fp8 = weight_fp16.to(torch.float8_e4m3fn)
        scale = torch.tensor([0.0008, 0.0008], dtype=torch.float32, device=DEVICE)

        output = torch.empty(1, N, dtype=torch.float16, device=DEVICE)
        normed_out = torch.empty(1, K, dtype=torch.float16, device=DEVICE)
        residual_copy = residual.clone()

        esimd_resadd_norm_gemv_fp8_pert(
            hidden,
            residual_copy,
            norm_weight,
            weight_fp8,
            scale,
            output,
            normed_out,
            eps,
        )
        torch.xpu.synchronize()

        ref_output, ref_residual, ref_normed = ref_resadd_norm_gemv_fp8(
            hidden, residual, norm_weight, weight_fp8, scale, eps
        )

        res_diff = (residual_copy.cpu().float() - ref_residual.float()).abs().max().item()
        norm_diff = (normed_out.cpu().float() - ref_normed.float()).abs().max().item()
        out_diff = (output.cpu().float() - ref_output).abs()
        rel_err = out_diff.mean().item() / (ref_output.abs().mean().item() + 1e-6)

        # print(f"{name}: res_diff={res_diff:.6f} norm_diff={norm_diff:.6f} rel_err={rel_err:.6f}")
        # print_topk_output_errors(name, output, ref_output)

        assert res_diff < 0.05, f"residual diff too large: {res_diff:.4f}"
        assert norm_diff < 0.1, f"normed diff too large: {norm_diff:.4f}"
        assert rel_err < 0.3, f"output relative error too large: {rel_err:.4f}"


if __name__ == "__main__":
    test_correctness()
    benchmark_fused_vs_split()
