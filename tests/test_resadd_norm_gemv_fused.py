import torch

from custom_esimd_kernels_vllm import esimd_resadd_norm_gemv_fp8_pert

DEVICE = torch.device("xpu")


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


def test_correctness():
    torch.manual_seed(0)
    eps = 1e-6
    shapes = [
        ("gate_up_proj", 20480, 2560),
    ]

    for name, N, K in shapes:
        hidden = torch.randn(1, K, dtype=torch.float16, device=DEVICE)
        residual = torch.randn(1, K, dtype=torch.float16, device=DEVICE)
        norm_weight = torch.randn(K, dtype=torch.float16, device=DEVICE) * 0.1
        weight_fp16 = torch.randn(N, K, dtype=torch.float16, device=DEVICE) * 0.1
        weight_fp8 = weight_fp16.to(torch.float8_e4m3fn)
        scale = torch.tensor([0.08, 0.08], dtype=torch.float32, device=DEVICE)

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

        assert res_diff < 0.05, f"residual diff too large: {res_diff:.4f}"
        assert norm_diff < 0.1, f"normed diff too large: {norm_diff:.4f}"
        assert rel_err < 0.2, f"output relative error too large: {rel_err:.4f}"


if __name__ == "__main__":
    test_correctness()
