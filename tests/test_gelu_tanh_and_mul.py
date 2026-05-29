import sys
import time
from pathlib import Path

import pytest
import torch

from custom_esimd_kernels_vllm import esimd_gelu_tanh_and_mul, select_gelu_tanh_and_mul_vl_ks

DEVICE = torch.device("xpu")
TARGET_BW = 112.0
WARMUP_ITERS = 20
BENCHMARK_ITERS = 1000
VL_CANDIDATES = (128, 256, 512)
KS_CANDIDATES = (1, 2, 4, 5, 8, 10, 16, 20, 32, 40, 64)
SHAPES = (
    (1, 20480),
    # (2, 20480),
    # (4, 20480),
    # (8, 20480),
    # (16, 20480),
    # (17, 20480),
    # (32, 20480),
)
DTYPES = (torch.float16, torch.bfloat16)
MAX_ABS_TOL = {
    torch.float16: 5e-2,
    torch.bfloat16: 8e-2,
}
MEAN_ABS_TOL = {
    torch.float16: 5e-3,
    torch.bfloat16: 8e-3,
}


def _ensure_vllm_repo_on_path() -> Path:
    vllm_repo = Path(__file__).resolve().parents[2] / "applications.ai.gpu.vllm-xpu"
    if not vllm_repo.exists():
        raise RuntimeError(f"vllm repo not found: {vllm_repo}")
    if str(vllm_repo) not in sys.path:
        sys.path.insert(0, str(vllm_repo))
    return vllm_repo


def _build_gemma4_act_fn():
    _ensure_vllm_repo_on_path()
    from vllm.platforms import current_platform

    current_platform.import_kernels()

    if not hasattr(torch.ops, "_C") or not hasattr(torch.ops._C, "gelu_tanh_and_mul"):
        raise RuntimeError("torch.ops._C.gelu_tanh_and_mul is not available")

    def _act_fn(input_tensor: torch.Tensor) -> torch.Tensor:
        output = torch.empty(
            input_tensor.shape[0],
            input_tensor.shape[1] // 2,
            dtype=input_tensor.dtype,
            device=input_tensor.device,
        )
        torch.ops._C.gelu_tanh_and_mul(output, input_tensor)
        return output

    return _act_fn


def _benchmark_xpu_callable(fn, total_bytes: int) -> tuple[float, float, float]:
    for _ in range(WARMUP_ITERS):
        fn()

    torch.xpu.synchronize()
    start = time.perf_counter()
    for _ in range(BENCHMARK_ITERS):
        fn()
    torch.xpu.synchronize()
    elapsed = time.perf_counter() - start

    avg_latency_ms = elapsed / BENCHMARK_ITERS * 1000.0
    avg_latency_us = avg_latency_ms * 1000.0
    bandwidth_gbps = (total_bytes / 1e9) / (avg_latency_ms / 1e3)
    utilization_pct = bandwidth_gbps / TARGET_BW * 100.0
    return avg_latency_us, bandwidth_gbps, utilization_pct


def _activation_bytes(input_tensor: torch.Tensor) -> int:
    output_cols = input_tensor.shape[-1] // 2
    output_bytes = input_tensor.shape[0] * output_cols * input_tensor.element_size()
    return input_tensor.numel() * input_tensor.element_size() + output_bytes


def _valid_vl_ks(cols: int) -> list[tuple[int, int]]:
    half_cols = cols // 2
    valid = []
    for vl in VL_CANDIDATES:
        if half_cols % vl != 0:
            continue
        chunks_per_row = half_cols // vl
        for ks in KS_CANDIDATES:
            if 1 <= ks <= min(chunks_per_row, 64):
                valid.append((vl, ks))
    return valid


def _select_default_vl_ks(cols: int) -> tuple[int, int]:
    valid = _valid_vl_ks(cols)
    if not valid:
        raise RuntimeError(f"no valid vl/ks for cols={cols}")
    if (256, 8) in valid:
        return (256, 8)
    if (128, 8) in valid:
        return (128, 8)
    return valid[0]


def test_gelu_tanh_and_mul_matches_gemma4_act_fn() -> None:
    if not torch.xpu.is_available():
        pytest.skip("xpu is not available")

    act_fn = _build_gemma4_act_fn()
    torch.manual_seed(42)

    for dtype in DTYPES:
        for rows, cols in SHAPES:
            input_tensor = torch.randn(rows, cols, dtype=dtype, device=DEVICE) * 0.1
            esimd_output = torch.empty(rows, cols // 2, dtype=dtype, device=DEVICE)
            vl, ks = select_gelu_tanh_and_mul_vl_ks(input_tensor)

            reference_output = act_fn(input_tensor)
            esimd_gelu_tanh_and_mul(input_tensor, esimd_output, vl, ks)
            torch.xpu.synchronize()

            diff = (esimd_output.float() - reference_output.float()).abs()
            assert diff.max().item() <= MAX_ABS_TOL[dtype], (
                f"dtype={dtype}, shape={(rows, cols)}, max_diff={diff.max().item():.6f}"
            )
            assert diff.mean().item() <= MEAN_ABS_TOL[dtype], (
                f"dtype={dtype}, shape={(rows, cols)}, mean_diff={diff.mean().item():.6f}"
            )


def benchmark_gelu_tanh_and_mul_vs_gemma4_act_fn() -> None:
    if not torch.xpu.is_available():
        raise RuntimeError("xpu is not available")

    act_fn = _build_gemma4_act_fn()
    torch.manual_seed(42)

    print(
        f"\n{'Shape':<16} {'DType':>8} | {'Mode':>8} {'Config':>11} {'us':>10} {'GB/s':>10} {'BW%':>8} {'Speedup':>9}"
    )
    print("-" * 94)

    for rows, cols in SHAPES:
        for dtype in DTYPES:
            input_pool = [
                torch.randn(rows, cols, dtype=dtype, device=DEVICE) * 0.1
                for _ in range(64)
            ]
            esimd_output = torch.empty(rows, cols // 2, dtype=dtype, device=DEVICE)
            total_bytes = _activation_bytes(input_pool[0])
            run_state = {"index": 0}

            def run_reference() -> None:
                current_input = input_pool[run_state["index"] % len(input_pool)]
                act_fn(current_input)
                run_state["index"] += 1

            run_state["index"] = 0

            reference_avg_latency_us, reference_bandwidth_gbps, reference_utilization_pct = (
                _benchmark_xpu_callable(run_reference, total_bytes)
            )
            best_cfg = None
            best_esimd_avg_latency_us = None
            best_esimd_bandwidth_gbps = None
            best_esimd_utilization_pct = None
            selected_cfg = select_gelu_tanh_and_mul_vl_ks(input_pool[0])
            selected_latency_us = None

            for vl, ks in _valid_vl_ks(cols):
                run_state["index"] = 0

                def run_esimd(vl=vl, ks=ks) -> None:
                    current_input = input_pool[run_state["index"] % len(input_pool)]
                    esimd_gelu_tanh_and_mul(current_input, esimd_output, vl, ks)
                    run_state["index"] += 1

                esimd_avg_latency_us, esimd_bandwidth_gbps, esimd_utilization_pct = (
                    _benchmark_xpu_callable(run_esimd, total_bytes)
                )
                # print(
                #     f"Tested esimd config for shape={rows}x{cols} dtype={dtype}: VL={vl} KS={ks} -> {esimd_avg_latency_us:.3f} us"
                # )
                if best_esimd_avg_latency_us is None or esimd_avg_latency_us < best_esimd_avg_latency_us:
                    best_cfg = (vl, ks)
                    best_esimd_avg_latency_us = esimd_avg_latency_us
                    best_esimd_bandwidth_gbps = esimd_bandwidth_gbps
                    best_esimd_utilization_pct = esimd_utilization_pct
                if (vl, ks) == selected_cfg:
                    selected_latency_us = esimd_avg_latency_us

            assert best_cfg is not None
            assert selected_latency_us is not None

            shape_str = f"{rows}x{cols}"
            dtype_str = str(dtype).replace("torch.", "")
            print(
                f"{shape_str:<16} {dtype_str:>8} | {'gemma4':>8} {'-':>11} {reference_avg_latency_us:>10.3f} {reference_bandwidth_gbps:>10.2f} {reference_utilization_pct:>7.1f}% {1.0:>9.3f}"
            )
            print(
                f"{shape_str:<16} {dtype_str:>8} | {'esimd':>8} {best_cfg[0]:>3}:{best_cfg[1]:<7} {best_esimd_avg_latency_us:>10.3f} {best_esimd_bandwidth_gbps:>10.2f} {best_esimd_utilization_pct:>7.1f}% {reference_avg_latency_us / best_esimd_avg_latency_us:>9.3f}"
            )
            print(
                f"Selected config: {selected_cfg[0]}:{selected_cfg[1]} | latency delta vs best = {selected_latency_us - best_esimd_avg_latency_us:.3f} us"
            )


if __name__ == "__main__":
    test_gelu_tanh_and_mul_matches_gemma4_act_fn()
    benchmark_gelu_tanh_and_mul_vs_gemma4_act_fn()