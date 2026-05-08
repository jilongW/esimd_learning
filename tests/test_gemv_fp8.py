"""Focused tests for the retained esimd_gemv_fp8_pert kernel."""

import pytest
import torch


DEVICE = torch.device("xpu")


def _require_xpu():
    if not torch.xpu.is_available():
        pytest.skip("XPU is not available in the current environment")


def _run_reference_case(weight_dtype: torch.dtype, n: int, k: int):
    from custom_esimd_kernels_vllm import esimd_gemv_fp8_pert

    _require_xpu()

    weight_ref = torch.randn(n, k, dtype=torch.float16, device=DEVICE) * 0.1
    weight_fp8 = weight_ref.to(weight_dtype)
    scale_value = 0.05 + torch.rand(1).item() * 0.1
    scale_tensor = torch.tensor(scale_value, dtype=torch.float32, device=DEVICE)

    input_tensor = torch.randn(1, k, dtype=torch.float16, device=DEVICE) * 0.1
    output = torch.zeros(1, n, dtype=torch.float16, device=DEVICE)

    esimd_gemv_fp8_pert(input_tensor, weight_fp8, scale_tensor, output)

    reference = (input_tensor.float() @ weight_fp8.to(torch.float16).float().T) * scale_value
    max_diff = (output.float() - reference).abs().max().item()
    reference_max = reference.abs().max().item()
    rel_err = (max_diff / reference_max) if reference_max > 1e-6 else 0.0
    assert max_diff < 0.5 or rel_err < 0.05


@pytest.mark.parametrize("n,k", [(1024, 1024), (2560, 2048), (512, 2048), (128, 2048)])
def test_pert_correctness_e4m3(n: int, k: int):
    _run_reference_case(torch.float8_e4m3fn, n, k)


@pytest.mark.parametrize("n,k", [(1024, 1024), (2560, 2048), (512, 2048), (128, 2048)])
def test_pert_correctness_e5m2(n: int, k: int):
    _run_reference_case(torch.float8_e5m2, n, k)