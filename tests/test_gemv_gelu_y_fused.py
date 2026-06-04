import time

import torch

from custom_esimd_kernels_vllm import (
    esimd_gemv_gelu_tanh_mul_y_fp8_pert,
)

DEVICE = torch.device("xpu")
EPS = 1e-6
WARMUP_ITERS = 10
BENCHMARK_ITERS = 500
TARGET_BW = 112.0
SHAPES = [
    (256, 2560),
]
NORM_GEMV2_VL_CANDIDATES = (128, 256, 512, 1024)
NORM_GEMV2_KS_CANDIDATES = (1, 2, 4, 8, 10)


def _valid_vl_ks(k_size: int) -> list[tuple[int, int]]:
    valid = []
    for vl in NORM_GEMV2_VL_CANDIDATES:
        if k_size % vl != 0:
            continue
        for ks in NORM_GEMV2_KS_CANDIDATES:
            if vl * ks >= k_size and not (k_size == 256 and vl == 256 and ks == 1):
                continue
            valid.append((vl, ks))
    return valid


def _norm_gemv2_fused_bytes(n0: int, k_size: int, dtype: torch.dtype = torch.float16) -> int:
    elem = torch.tensor([], dtype=dtype).element_size()
    # x read + fp8 weight read + scale read + y read + output write
    return k_size * elem + n0 * k_size + 4 + n0 * elem + n0 * elem


def _norm_gemv2_split_bytes(n0: int, k_size: int, dtype: torch.dtype = torch.float16) -> int:
    elem = torch.tensor([], dtype=dtype).element_size()
    # fp8_gemm: x read + fp8 weight read + scale read + gate write
    # gelu*mul: gate read + y read + output write
    return k_size * elem + n0 * k_size + 4 + n0 * elem + n0 * elem + n0 * elem + n0 * elem


def ref_gemv_gelu_fused(hidden_states, weight, scale, res, eps):
    hidden = hidden_states.cpu().float()
    res = res.cpu().float()
    weight_f = weight.cpu().float()
    ref = (hidden.float() @ weight_f.T) * float(scale[0].cpu())
    return torch.nn.functional.gelu(ref, approximate="tanh") * res

def _benchmark_xpu_callable(fn, warmup_iters=WARMUP_ITERS, benchmark_iters=BENCHMARK_ITERS):
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



def test_gemv_gelu_fused_correctness() -> None:
    torch.manual_seed(42)

    for n0, k_size in SHAPES:
        hidden = torch.randn(1, k_size, dtype=torch.float16, device=DEVICE) * 0.1
        res = torch.randn(n0, dtype=torch.float16, device=DEVICE) * 0.1
        weight_fp16 = torch.randn(n0, k_size, dtype=torch.float16, device=DEVICE) * 0.1
        weight_fp8 = weight_fp16.to(torch.float8_e4m3fn)
        scale = torch.tensor([0.0008], dtype=torch.float32, device=DEVICE)

        fused_output = torch.empty(1, n0, dtype=torch.float16, device=DEVICE)
        
        esimd_gemv_gelu_tanh_mul_y_fp8_pert(
            hidden,
            weight_fp8,
            scale,
            res,
            fused_output,
        )
        torch.xpu.synchronize()

        # Reference implementation 
        ref_output = ref_gemv_gelu_fused(hidden, weight_fp8, scale, res, EPS)

        fused_vs_ref = (fused_output.cpu().float() - ref_output.float()).abs()

        assert fused_vs_ref.max().item() <= 5e-2
        assert fused_vs_ref.mean().item() <= 5e-3

def benchmark_gemv_gelu_fused() -> None:
    torch.manual_seed(42)
    best_cfgs = {}
    print(
        f"\n{'N0':>6} {'N1':>6} {'K':>6} | {'Method':>8} {'Config':>11} {'GB/s':>8} {'BW%':>7} {'us':>9} {'vs_best':>8}"
    )
    print("-" * 78)

    for n0, k_size in SHAPES:
        hidden = torch.randn(1, k_size, dtype=torch.float16, device=DEVICE) * 0.1
        scale = torch.tensor([0.0008], dtype=torch.float32, device=DEVICE)
        res = torch.randn(1, n0, dtype=torch.float16, device=DEVICE) * 0.1
        weight_bytes = n0 * k_size
        target_mem = 32 * 1024 * 1024
        num_copies = max(16, target_mem // max(weight_bytes, 1))
        num_copies = min(num_copies, 512)

        weight_pool = []
        for _ in range(num_copies):
            weight_fp16 = torch.randn(n0, k_size, dtype=torch.float16, device=DEVICE) * 0.1
            weight_fp8 = weight_fp16.to(torch.float8_e4m3fn)
            weight_pool.append(weight_fp8)

        fused_output = torch.empty(1, n0, dtype=torch.float16, device=DEVICE)

        split_output = torch.empty(1, n0, dtype=torch.float16, device=DEVICE)
        run_state = {"index": 0}

        best_cfg = None
        best_fused_latency_us = None
        for vl, ks in _valid_vl_ks(k_size):
            run_state["index"] = 0

            def run_fused(vl=vl, ks=ks):
                current_weight = weight_pool[run_state["index"] % num_copies]
                esimd_gemv_gelu_tanh_mul_y_fp8_pert(
                    hidden,
                    current_weight,
                    scale,
                    res,
                    fused_output,
                    vl,
                    ks,
                )
                run_state["index"] += 1

            candidate_latency_us = _benchmark_xpu_callable(run_fused)
            print(f"Tested fused config N={n0} K={k_size}: VL={vl} KS={ks} -> {candidate_latency_us:.2f} us")
            if best_fused_latency_us is None or candidate_latency_us < best_fused_latency_us:
                best_cfg = (vl, ks)
                best_fused_latency_us = candidate_latency_us

        assert best_cfg is not None
        best_cfgs[(n0, k_size)] = best_cfg

        run_state["index"] = 0
        from vllm.platforms import current_platform
        def run_split() -> None:
            current_weight = weight_pool[run_state["index"] % num_copies]
            gate = torch.ops._xpu_C.fp8_gemm_w8a16(
                hidden, current_weight.t(), scale, None
            )
            gate = torch.nn.functional.gelu(gate, approximate="tanh")
            split_output.copy_(gate * res)
            run_state["index"] += 1

        split_latency_us = _benchmark_xpu_callable(run_split)

        fused_vl, fused_ks = best_cfg
        best_latency_us = min(best_fused_latency_us, split_latency_us)
        fused_bw = (_norm_gemv2_fused_bytes(n0, k_size) / 1e9) / (best_fused_latency_us / 1e6)
        split_bw = (_norm_gemv2_split_bytes(n0, k_size) / 1e9) / (split_latency_us / 1e6)

        print(
            f"{n0:>6} {n0:>6} {k_size:>6} | {'fused':>8} {fused_vl:>3}:{fused_ks:<7} {fused_bw:>7.1f} {fused_bw / TARGET_BW * 100:>6.1f}% {best_fused_latency_us:>8.2f} {best_fused_latency_us / best_latency_us:>8.3f}"
        )
        print(
            f"{n0:>6} {n0:>6} {k_size:>6} | {'split':>8} {'-':>11} {split_bw:>7.1f} {split_bw / TARGET_BW * 100:>6.1f}% {split_latency_us:>8.2f} {split_latency_us / best_latency_us:>8.3f}"
        )


if __name__ == "__main__":
    test_gemv_gelu_fused_correctness()
    benchmark_gemv_gelu_fused()