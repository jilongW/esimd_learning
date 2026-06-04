import time

import torch

from custom_esimd_kernels_vllm import (
    esimd_gemv_fp8,
    esimd_norm_gemv_fp8_pert,
    esimd_rms_norm,
    esimd_rms_norm_res
)

DEVICE = torch.device("xpu")
EPS = 1e-6
WARMUP_ITERS = 10
BENCHMARK_ITERS = 1000
TARGET_BW = 112.0  # GB/s PTL
VL_CANDIDATES = (128, 256, 512)
KS_CANDIDATES = (1, 2, 4, 8, 10)
SHAPES = [
    # ("qkv_proj", 3072, 2560),
    # ("qkv_proj", 6144, 2560),
    # ("attn_o_proj", 2560, 2048),
    # ("attn_o_proj", 2560, 4096),
    # ("gate_up_proj", 20480, 2560),
    # ("down_proj", 2560, 10240),
    # ("per_layer_input_gate", 256, 2560),
    ("per_layer_input_gate_out", 2560, 256),
]


def _normalize_vl_ks(k_size: int, vl: int, ks: int) -> tuple[int, int]:
    k_per_thread = k_size // ks
    while vl > k_per_thread or k_per_thread % vl != 0:
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
        k_per_thread = k_size // ks
    return vl, ks


def _select_norm_gemv_vl_ks(n_size: int, k_size: int) -> tuple[int, int]:
    if k_size < 256:
        vl, ks = 128, 1
    elif k_size == 256:
        vl, ks = 256, 1
    elif k_size >= 10240:
        vl, ks = 256, 8
    elif k_size >= 4096:
        vl, ks = 256, 4
    elif k_size >= 2560 and n_size >= 10240:
        vl, ks = 128, 4
    elif k_size >= 2560:
        vl, ks = 128, 10
    elif k_size >= 2048:
        vl, ks = 256, 8
    else:
        vl, ks = 256, 1

    vl, ks = _normalize_vl_ks(k_size, vl, ks)
    if (vl, ks) not in _valid_norm_gemv_vl_ks(k_size):
        raise ValueError(f"No valid norm_gemv vl/ks for N={n_size}, K={k_size}")
    return vl, ks


def _build_select_norm_gemv_vl_ks(best_cfgs: dict[tuple[int, int], tuple[int, int]]):
    def _select(n_size: int, k_size: int) -> tuple[int, int]:
        if (n_size, k_size) in best_cfgs:
            return best_cfgs[(n_size, k_size)]
        return _select_norm_gemv_vl_ks(n_size, k_size)

    return _select


def _print_suggested_select_vl_ks(best_cfgs: dict[tuple[int, int], tuple[int, int]]) -> None:
    print("\nSuggested norm_gemv select_vl_ks table:")
    print("BEST_CONFIGS = {")
    for n_size, k_size in sorted(best_cfgs):
        vl, ks = best_cfgs[(n_size, k_size)]
        print(f"    ({n_size}, {k_size}): ({vl}, {ks}),")
    print("}")


def _valid_norm_gemv_vl_ks(k_size: int) -> list[tuple[int, int]]:
    valid = []
    for vl in VL_CANDIDATES:
        if k_size % vl != 0:
            continue
        for ks in KS_CANDIDATES:
            if k_size % ks != 0:
                continue
            if (k_size // ks) % vl != 0:
                continue
            valid.append((vl, ks))
    return valid


def _select_rms_norm_vl_ks(rows: int, hidden_size: int) -> tuple[int, int]:
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


def _gemv_norm_res_fused_bytes(n_size: int, k_size: int, scale_count: int = 1) -> int:
    hidden_bytes = k_size * 2
    norm_weight_bytes = k_size * 2
    gemv_weight_bytes = n_size * k_size
    scale_bytes = 4 * scale_count
    res_bytes = n_size * 2
    output_bytes = n_size * 2
    return hidden_bytes * 2 + norm_weight_bytes + gemv_weight_bytes + scale_bytes + res_bytes + output_bytes


def _gemv_norm_res_split_bytes(n_size: int, k_size: int, scale_count: int = 1) -> int:
    rms_norm_bytes = k_size * 2 * 2 + k_size * 2 + k_size * 2
    gemv_bytes = k_size * 2 + n_size * k_size + 4 * scale_count + n_size * 2
    res_bytes = n_size * 2
    return rms_norm_bytes + gemv_bytes + res_bytes


def _ref_gemv_norm_res_fp8(hidden, norm_weight, weight_fp8, scale, res, eps):
    hidden_f = hidden.cpu().float()
    norm_weight_f = norm_weight.cpu().float()
    res = res.cpu().float()
    scale_f = scale.cpu().float()
    dequant_weight = weight_fp8.cpu().float() * scale_f
    tmp = hidden_f @ dequant_weight.T
    variance = tmp.pow(2).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(variance + eps)
    normed = tmp * inv_rms * norm_weight_f
    output = normed + res
    return output


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



def test_gemv_norm_res_fp8_pert_correctness():
    torch.manual_seed(42)

    for name, n_size, k_size in SHAPES:
        hidden = torch.randn(1, k_size, dtype=torch.float16, device=DEVICE) * 0.1
        norm_weight = torch.randn(n_size, dtype=torch.float16, device=DEVICE) * 0.1
        weight_fp16 = torch.randn(n_size, k_size, dtype=torch.float16, device=DEVICE) * 0.1
        weight_fp8 = weight_fp16.to(torch.float8_e4m3fn)
        res = torch.randn(1, n_size, dtype=torch.float16, device=DEVICE) * 0.1
        scale = torch.tensor([0.0008], dtype=torch.float32, device=DEVICE)
        output = torch.empty(1, n_size, dtype=torch.float16, device=DEVICE)
        esimd_gemv_fp8(
            hidden,
            weight_fp8,
            scale,
            output,
        )
        esimd_rms_norm_res(
            output,
            res,
            norm_weight,
            EPS,
            output,
        )
        torch.xpu.synchronize()

        ref_output = _ref_gemv_norm_res_fp8(
            hidden,
            norm_weight,
            weight_fp8,
            scale,
            res,
            EPS,
        )
        out_diff = (output.cpu().float() - ref_output).abs()
        rel_err = out_diff.mean().item() / (ref_output.abs().mean().item() + 1e-6)

        assert rel_err < 0.3, f"{name} relative error too large: {rel_err:.4f}"


def benchmark_gemv_norm_res_fp8_pert_vs_split():
    torch.manual_seed(42)
    best_cfgs = {}
    scale_name = "one_scale"
    scale = torch.tensor([0.0008], dtype=torch.float32, device=DEVICE)

    print(
        f"\n{'Shape':<20} {'N':>6} {'K':>6} | {'Scale':>9} {'Mode':>6} {'Config':>11} {'GB/s':>8} {'BW%':>7} {'us':>8} {'F/S':>8} {'S/F':>8}"
    )
    print("-" * 117)

    for name, n_size, k_size in SHAPES:
        hidden = torch.randn(1, k_size, dtype=torch.float16, device=DEVICE) * 0.1
        norm_weight = torch.randn(n_size, dtype=torch.float16, device=DEVICE) * 0.1
        weight_fp16 = torch.randn(n_size, k_size, dtype=torch.float16, device=DEVICE) * 0.1
        res = torch.randn(1, n_size, dtype=torch.float16, device=DEVICE) * 0.1
        weight_fp8 = weight_fp16.to(torch.float8_e4m3fn)
        scale = torch.tensor([0.0008], dtype=torch.float32, device=DEVICE)
        weight_bytes = n_size * k_size
        target_mem = 32 * 1024 * 1024
        num_copies = max(16, target_mem // max(weight_bytes, 1))
        num_copies = min(num_copies, 512)

        weights = [weight_fp8]
        for _ in range(1, num_copies):
            extra_weight = torch.randn(n_size, k_size, dtype=torch.float16, device=DEVICE) * 0.1
            weights.append(extra_weight.to(torch.float8_e4m3fn))

        rows = hidden.numel() // hidden.shape[-1]
        fused_output = torch.empty(1, n_size, dtype=torch.float16, device=DEVICE)
        split_normed = torch.empty_like(hidden)
        split_output = torch.empty(1, n_size, dtype=torch.float16, device=DEVICE)
        run_state = {"index": 0}
        from vllm.platforms import current_platform
        def run_split():
            current_weight = weights[run_state["index"] % num_copies]
            output = torch.ops._xpu_C.fp8_gemm_w8a16(hidden, current_weight.t(), scale, None)
            torch.ops._C.rms_norm(output, output, norm_weight, EPS)
            split_output = output + res
            run_state["index"] += 1

        best_cfg = None
        best_fused_latency_us = None
        for vl, ks in _valid_norm_gemv_vl_ks(k_size):
            run_state["index"] = 0

            def run_fused(vl=vl, ks=ks):
                current_weight = weights[run_state["index"] % num_copies]
                esimd_gemv_fp8(
                    hidden,
                    current_weight,
                    scale,
                    fused_output,
                )
                esimd_rms_norm_res(
                    fused_output,
                    res,
                    norm_weight,
                    EPS,
                    fused_output,
                )
                run_state["index"] += 1

            candidate_latency_us = _benchmark_xpu_callable(run_fused)
            print(f"Tested fused config for {name} N={n_size} K={k_size}: VL={vl} KS={ks} -> {candidate_latency_us:.2f} us")
            if best_fused_latency_us is None or candidate_latency_us < best_fused_latency_us:
                best_cfg = (vl, ks)
                best_fused_latency_us = candidate_latency_us

        assert best_cfg is not None
        best_cfgs[(n_size, k_size)] = best_cfg

        run_state["index"] = 0
        split_latency_us = _benchmark_xpu_callable(run_split)

        fused_bw = (_gemv_norm_res_fused_bytes(n_size, k_size, scale.numel()) / 1e9) / (best_fused_latency_us / 1e6)
        split_bw = (_gemv_norm_res_split_bytes(n_size, k_size, scale.numel()) / 1e9) / (split_latency_us / 1e6)
        fused_over_split = best_fused_latency_us / split_latency_us
        split_over_fused = split_latency_us / best_fused_latency_us

        print(f"{name:<20} {n_size:>6} {k_size:>6} | {scale_name:>9} {'fused':>6} {best_cfg[0]:>3}:{best_cfg[1]:<7} {fused_bw:>7.1f} {fused_bw / TARGET_BW * 100:>6.1f}% {best_fused_latency_us:>7.2f} {fused_over_split:>8.3f} {split_over_fused:>8.3f}")
        print(f"{name:<20} {n_size:>6} {k_size:>6} | {scale_name:>9} {'split':>6} {split_bw:>7.1f} {split_bw / TARGET_BW * 100:>6.1f}% {split_latency_us:>7.2f} {1.0:>8.3f} {1.0:>8.3f}")

    _print_suggested_select_vl_ks(best_cfgs)
    searched_select = _build_select_norm_gemv_vl_ks(best_cfgs)
    print("\nSelector replay:")
    for name, n_size, k_size in SHAPES:
        vl, ks = searched_select(n_size, k_size)
        print(f"{name:<20} {n_size:>6} {k_size:>6} -> {vl}:{ks}")


if __name__ == "__main__":
    test_gemv_norm_res_fp8_pert_correctness()
    benchmark_gemv_norm_res_fp8_pert_vs_split()