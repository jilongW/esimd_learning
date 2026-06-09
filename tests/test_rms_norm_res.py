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
    _ = rows
    V = hidden_size
    vl = 512
    ks = 1

    if V <= 512:
        vl = 256
        ks = 1
    elif V <= 2560:
        vl = 128
        ks = 10
    elif V <= 5120:
        vl = 512
        ks = 8
    else:
        vl = 1024
        ks = 1

    kpt = V // ks
    while vl > kpt or kpt % vl != 0:
        if vl > 128:
            vl //= 2
        elif ks > 1:
            ks //= 2
            kpt = V // ks
        else:
            break

    return vl, ks


def _shape_rows_hidden(shape: tuple[int, ...]) -> tuple[int, int]:
    hidden_size = shape[-1]
    rows = 1
    for dim in shape[:-1]:
        rows *= dim
    return rows, hidden_size


def ref_rms_norm_res(hidden_states, res, weight, eps):
    hidden = hidden_states.cpu().float()
    res = res.cpu().float()
    weight_f = weight.cpu().float()
    variance = hidden.pow(2).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(variance + eps)
    return hidden * inv_rms * weight_f + res


def ref_rms_norm_res_scale(hidden_states, res, weight, scale, eps):
    return ref_rms_norm_res(hidden_states, res, weight, eps) * scale.cpu().float()


def test_rms_norm_res_correctness():
    from custom_esimd_kernels_vllm import esimd_rms_norm_res

    torch.manual_seed(42)
    eps = 1e-6

    for dtype in (torch.float16, torch.bfloat16):
        for shape in HIDDEN_SHAPES:
            rows, hidden_size = _shape_rows_hidden(shape)
            hidden = torch.randn(*shape, dtype=dtype, device=device)
            res = torch.randn(*shape, dtype=dtype, device=device)
            weight = torch.randn(hidden_size, dtype=dtype, device=device) * 0.1
            out = torch.empty_like(hidden)

            esimd_rms_norm_res(hidden, res, weight, eps, out)
            torch.xpu.synchronize()

            ref = ref_rms_norm_res(hidden, res, weight, eps)
            diff = (out.cpu().float() - ref).abs()
            assert diff.max().item() < 0.1, (
                f"dtype={dtype}, shape={tuple(hidden.shape)}, diff={diff.max().item():.4f}"
            )


def test_rms_norm_res_scale_correctness():
    from custom_esimd_kernels_vllm import esimd_rms_norm_res

    torch.manual_seed(42)
    eps = 1e-6

    for dtype in (torch.float16, torch.bfloat16):
        for shape in HIDDEN_SHAPES:
            _, hidden_size = _shape_rows_hidden(shape)
            hidden = torch.randn(*shape, dtype=dtype, device=device)
            res = torch.randn(*shape, dtype=dtype, device=device)
            weight = torch.randn(hidden_size, dtype=dtype, device=device) * 0.1
            scale = torch.tensor([0.73], dtype=torch.float32, device=device)
            out = torch.empty_like(hidden)

            esimd_rms_norm_res(hidden, res, weight, eps, out, scale=scale)
            torch.xpu.synchronize()

            ref = ref_rms_norm_res_scale(hidden, res, weight, scale, eps)
            diff = (out.cpu().float() - ref).abs()
            assert diff.max().item() < 0.1, (
                f"scale dtype={dtype}, shape={tuple(hidden.shape)}, diff={diff.max().item():.4f}"
            )


def _effective_rms_norm_bytes(hidden_states: torch.Tensor, res: torch.Tensor, weight: torch.Tensor) -> int:
    # Compare on the same algorithmic workload: two reads of x, one read of weight, one write of output.
    return (
        2 * hidden_states.numel() * hidden_states.element_size()
        + weight.numel() * weight.element_size()
        + hidden_states.numel() * hidden_states.element_size() + res.numel() * res.element_size()
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
    res = [torch.randn(*shape, dtype=dtype, device=device) for _ in range(pool_size)]
    weight_pool = [
        torch.randn(hidden_size, dtype=dtype, device=device) * 0.1 for _ in range(pool_size)
    ]
    return hidden_pool, res, weight_pool


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


def benchmark_rms_norm_res():
    from custom_esimd_kernels_vllm import esimd_rms_norm_res

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
            hidden_pool, res_pool, weight_pool = _make_input_pools(shape, hidden_size, dtype)
            hidden = hidden_pool[0]
            res = res_pool[0]
            weight = weight_pool[0]
            out_torch = torch.empty_like(hidden)
            heuristic_vl, heuristic_ks = select_vl_ks(rows, hidden_size)
            total_bytes = _effective_rms_norm_bytes(hidden, res, weight)
            total_flops = _rms_norm_flops(hidden)
            out_esimd = torch.empty_like(hidden)

            best_cfg = None
            best_us = None
            best_outputs = None
            torch_us, torch_outputs = _benchmark_one(
                lambda index: (
                    torch.ops._C.rms_norm(
                        out_torch,
                        hidden_pool[index % len(hidden_pool)],
                        weight_pool[index % len(weight_pool)],
                        eps,
                    ),
                    out_torch.add_(res_pool[index % len(res_pool)]),
                ),
                lambda: out_torch,
                iters,
            )
            torch_bw = total_bytes / (torch_us * 1e-6) / 1e9
            torch_tflops = total_flops / (torch_us * 1e6)


            for vl, ks in _valid_vl_ks(hidden_size):
                candidate_us, candidate_outputs = _benchmark_one(
                    lambda index, vl=vl, ks=ks: esimd_rms_norm_res(
                        hidden_pool[index % len(hidden_pool)],
                        res_pool[index % len(res_pool)],
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


def benchmark_rms_norm_res_scale():
    from custom_esimd_kernels_vllm import esimd_rms_norm_res

    eps = 1e-6
    print("\n--- RMSNorm+Res+Scale Benchmark (#sym:rms_norm_res_scale) ---")
    print(
        f"\n{'Case':<30} {'Config':>20} | {'ESIMD us':>10} {'Torch us':>10} {'ESIMD TF':>10} {'Torch TF':>10} {'ESIMD GB/s':>12} {'Torch GB/s':>11} {'Speedup':>8}"
    )
    print("-" * 144)

    for shape in HIDDEN_SHAPES:
        rows, hidden_size = _shape_rows_hidden(shape)
        iters = _benchmark_iters(shape)

        for dtype in (torch.float16, torch.bfloat16):
            hidden_pool, res_pool, weight_pool = _make_input_pools(shape, hidden_size, dtype)
            out_torch = torch.empty(shape, dtype=dtype, device=device)
            out_esimd = torch.empty(shape, dtype=dtype, device=device)
            scale = torch.tensor([0.73], dtype=torch.float32, device=device)

            heuristic_vl, heuristic_ks = select_vl_ks(rows, hidden_size)

            hidden = hidden_pool[0]
            res = res_pool[0]
            weight = weight_pool[0]
            total_bytes = _effective_rms_norm_bytes(hidden, res, weight) + scale.element_size()
            total_flops = _rms_norm_flops(hidden) + hidden.numel()

            torch_us, torch_outputs = _benchmark_one(
                lambda index: (
                    torch.ops._C.rms_norm(
                        out_torch,
                        hidden_pool[index % len(hidden_pool)],
                        weight_pool[index % len(weight_pool)],
                        eps,
                    ),
                    out_torch.add_(res_pool[index % len(res_pool)]),
                    out_torch.mul_(scale),
                ),
                lambda: out_torch,
                iters,
            )
            torch_bw = total_bytes / (torch_us * 1e-6) / 1e9
            torch_tflops = total_flops / (torch_us * 1e6)

            best_cfg = None
            best_us = None
            best_outputs = None
            for vl, ks in _valid_vl_ks(hidden_size):
                candidate_us, candidate_outputs = _benchmark_one(
                    lambda index, vl=vl, ks=ks: esimd_rms_norm_res(
                        hidden_pool[index % len(hidden_pool)],
                        res_pool[index % len(res_pool)],
                        weight_pool[index % len(weight_pool)],
                        eps,
                        out_esimd,
                        scale=scale,
                        vl=vl,
                        ks=ks,
                    ),
                    lambda: out_esimd,
                    iters,
                )
                if best_us is None or candidate_us < best_us:
                    best_cfg = (vl, ks)
                    best_us = candidate_us
                    best_outputs = candidate_outputs

            assert best_cfg is not None, f"No valid vl/ks for hidden_size={hidden_size}"

            _assert_output_lists_close(best_outputs, torch_outputs)

            esimd_bw = total_bytes / (best_us * 1e-6) / 1e9
            esimd_tflops = total_flops / (best_us * 1e6)
            speedup = torch_us / best_us if best_us > 0 else 0.0
            case_name = f"rms_norm_res_scale {str(dtype).split('.')[-1]}"
            cfg_str = (
                f"shape={tuple(shape)} rule={heuristic_vl}:{heuristic_ks} "
                f"best={best_cfg[0]}:{best_cfg[1]}"
            )
            print(
                f"{case_name:<30} {cfg_str:>20} | {best_us:>9.2f} {torch_us:>9.2f} "
                f"{esimd_tflops:>9.4f} {torch_tflops:>9.4f} {esimd_bw:>11.2f} {torch_bw:>11.2f} {speedup:>7.2f}x"
            )


if __name__ == "__main__":
    test_rms_norm_res_correctness()
    test_rms_norm_res_scale_correctness()
    benchmark_rms_norm_res()
    benchmark_rms_norm_res_scale()