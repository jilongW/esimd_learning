import sys
import time
from pathlib import Path

import torch

device = torch.device("xpu")
VL_CANDIDATES = (128, 256, 512, 1024)
KS_CANDIDATES = (1, 2, 5, 8, 10)
HIDDEN_SHAPES = (
    (1, 2048),
    # # (1, 2048),
    (1, 2560),
    (1, 5120),
    # (128, 2048),
    (1, 8, 512),
    # (1, 8, 2048),
    (1, 8, 256),
    (1, 2 ,512),
    (1, 2, 256),
)


def select_vl_ks(rows: int, hidden_size: int) -> tuple[int, int]:
    if hidden_size <= 256:
        vl, ks = 256, 1
    elif hidden_size <= 512:
        vl, ks = (256, 1) if rows >= 64 else (128, 1)
    elif hidden_size <= 2048:
        vl, ks = (512, 2) if rows >= 64 else (1024, 1)
    elif hidden_size <= 2560:
        vl, ks = (256, 5) if rows >= 64 else (256, 8)
    else:
        vl, ks = (256, 5) if rows >= 64 else (256, 8)

    while hidden_size % vl != 0 and vl > 128:
        if vl == 1024:
            vl = 512
        else:
            vl //= 2

    while vl * ks >= hidden_size and not (hidden_size == 256 and vl == 256 and ks == 1):
        if ks == 10:
            ks = 8
        elif ks == 8:
            ks = 5
        elif ks == 5:
            ks = 2
        elif ks == 2:
            ks = 1
        elif vl > 128:
            vl //= 2
            ks = 1
        else:
            raise ValueError(f"No valid vl/ks for hidden_size={hidden_size}")

    return vl, ks


def _shape_rows_hidden(shape: tuple[int, ...]) -> tuple[int, int]:
    hidden_size = shape[-1]
    rows = 1
    for dim in shape[:-1]:
        rows *= dim
    return rows, hidden_size


def ref_rms_norm(hidden_states, weight, eps):
    hidden = hidden_states.cpu().float()
    weight_f = weight.cpu().float()
    variance = hidden.pow(2).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(variance + eps)
    return hidden * inv_rms * weight_f


def test_rms_norm_correctness():
    from custom_esimd_kernels_vllm import esimd_rms_norm

    torch.manual_seed(42)
    eps = 1e-6

    for dtype in (torch.float16, torch.bfloat16):
        for shape in HIDDEN_SHAPES:
            rows, hidden_size = _shape_rows_hidden(shape)
            hidden = torch.randn(*shape, dtype=dtype, device=device)
            weight = torch.randn(hidden_size, dtype=dtype, device=device) * 0.1
            out = torch.empty_like(hidden)
            vl, ks = select_vl_ks(rows, hidden_size)

            esimd_rms_norm(hidden, weight, eps, out, vl, ks)
            torch.xpu.synchronize()

            ref = ref_rms_norm(hidden, weight, eps)
            diff = (out.cpu().float() - ref).abs()
            assert diff.max().item() < 0.1, (
                f"dtype={dtype}, shape={tuple(hidden.shape)}, diff={diff.max().item():.4f}"
            )


def _effective_rms_norm_bytes(hidden_states: torch.Tensor, weight: torch.Tensor) -> int:
    # Compare on the same algorithmic workload: two reads of x, one read of weight, one write of output.
    return (
        2 * hidden_states.numel() * hidden_states.element_size()
        + weight.numel() * weight.element_size()
        + hidden_states.numel() * hidden_states.element_size()
    )


def _benchmark_one(run_fn, output_fn, iters: int) -> tuple[float, list[float]]:
    for _ in range(20):
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
    return (time.perf_counter() - t0) * 1e6 / iters, outputs


def _assert_output_lists_close(
    esimd_outputs: list[float],
    torch_outputs: list[float],
    atol: float = 0.1,
) -> None:
    assert len(esimd_outputs) == len(torch_outputs), (
        f"output list length mismatch: esimd={len(esimd_outputs)} torch={len(torch_outputs)}"
    )

    for index, (esimd_out, torch_out) in enumerate(zip(esimd_outputs, torch_outputs)):
        max_diff = abs(esimd_out - torch_out)
        assert max_diff < atol, (
            f"benchmark output mismatch at iter={index}, diff={max_diff:.4f}"
        )


def _benchmark_iters(hidden_shape: tuple[int, int]) -> int:
    hidden_size = hidden_shape[-1]
    if hidden_size <= 512:
        return 4000
    if hidden_size <= 2560:
        return 2000
    return 1000


def _rms_norm_flops(hidden_states: torch.Tensor) -> int:
    return 4 * hidden_states.numel()


def _make_input_pools(
    shape: tuple[int, ...],
    hidden_size: int,
    dtype: torch.dtype,
    pool_size: int = 64,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    hidden_pool = [torch.randn(*shape, dtype=dtype, device=device) for _ in range(pool_size)]
    weight_pool = [
        torch.randn(hidden_size, dtype=dtype, device=device) * 0.1 for _ in range(pool_size)
    ]
    return hidden_pool, weight_pool


def _valid_vl_ks(hidden_size: int) -> list[tuple[int, int]]:
    valid = []
    for vl in VL_CANDIDATES:
        if hidden_size % vl != 0:
            continue
        for ks in KS_CANDIDATES:
            if vl * ks >= hidden_size and not (hidden_size == 256 and vl == 256 and ks == 1):
                continue
            valid.append((vl, ks))
    return valid


def benchmark_rms_norm():
    from custom_esimd_kernels_vllm import esimd_rms_norm

    if not hasattr(torch.ops, "_C") or not hasattr(torch.ops._C, "rms_norm"):
        vllm_repo = Path(__file__).resolve().parents[2] / "applications.ai.gpu.vllm-xpu"
        if vllm_repo.exists():
            sys.path.insert(0, str(vllm_repo))
            from vllm.platforms import current_platform

            current_platform.import_kernels()

    if not hasattr(torch.ops, "_C") or not hasattr(torch.ops._C, "rms_norm"):
        raise RuntimeError("torch.ops._C.rms_norm is not available in this environment")

    eps = 1e-6
    print("\n--- RMSNorm Benchmark ---")
    print(
        f"\n{'Case':<30} {'Config':>20} | {'Indiv us':>10} {'vllm us':>10} {'Indiv TF':>10} {'vllm TF':>10} {'Indiv GB/s':>12} {'vllm GB/s':>11} {'Speedup':>8}"
    )
    print("-" * 136)

    for shape in HIDDEN_SHAPES:
        rows, hidden_size = _shape_rows_hidden(shape)
        iters = _benchmark_iters(shape)
        for dtype in (torch.float16, torch.bfloat16):
            hidden_pool, weight_pool = _make_input_pools(shape, hidden_size, dtype)
            hidden = hidden_pool[0]
            weight = weight_pool[0]
            out_torch = torch.empty_like(hidden)
            heuristic_vl, heuristic_ks = select_vl_ks(rows, hidden_size)
            total_bytes = _effective_rms_norm_bytes(hidden, weight)
            total_flops = _rms_norm_flops(hidden)
            out_esimd = torch.empty_like(hidden)

            best_cfg = None
            best_us = None
            best_outputs = None
            torch_us, torch_outputs = _benchmark_one(
                lambda index: torch.ops._C.rms_norm(
                    out_torch,
                    hidden_pool[index % len(hidden_pool)],
                    weight_pool[index % len(weight_pool)],
                    eps,
                ),
                lambda: out_torch,
                iters,
            )
            torch_bw = total_bytes / (torch_us * 1e-6) / 1e9
            torch_tflops = total_flops / (torch_us * 1e6)


            for vl, ks in _valid_vl_ks(hidden_size):
                candidate_us, candidate_outputs = _benchmark_one(
                    lambda index, vl=vl, ks=ks: esimd_rms_norm(
                        hidden_pool[index % len(hidden_pool)],
                        weight_pool[index % len(weight_pool)],
                        eps,
                        out_esimd,
                        vl,
                        ks,
                    ),
                    lambda: out_esimd,
                    iters,
                )
                if best_us is None or candidate_us < best_us:
                    best_cfg = (vl, ks)
                    best_us = candidate_us
                    best_outputs = candidate_outputs

            
            assert best_cfg is not None, f"No valid vl/ks for hidden_size={hidden_size}"

            esimd_bw = total_bytes / (best_us * 1e-6) / 1e9
            esimd_tflops = total_flops / (best_us * 1e6)
            _assert_output_lists_close(best_outputs, torch_outputs)

            case_name = f"rms_norm {str(dtype).split('.')[-1]}"
            cfg_str = (
                f"shape={tuple(shape)} rule={heuristic_vl}:{heuristic_ks} "
                f"best={best_cfg[0]}:{best_cfg[1]}"
            )
            speedup = torch_us / best_us if best_us > 0 else 0.0
            print(
                f"{case_name:<30} {cfg_str:>20} | {best_us:>9.2f} {torch_us:>9.2f} "
                f"{esimd_tflops:>9.4f} {torch_tflops:>9.4f} {esimd_bw:>11.2f} {torch_bw:>11.2f} {speedup:>7.2f}x"
            )


if __name__ == "__main__":
    test_rms_norm_correctness()
    benchmark_rms_norm()