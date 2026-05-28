"""
Test unified esimd_gemv_fp8 kernel — FP8 GEMV with scale-shape dispatch, FP32 accumulation.

Correctness: compare against FP16 dequant reference (torch matmul).
Performance: benchmark Qwen3-Next-80B-A3B TP4 projection shapes.
"""
import gc
import torch
import time
from vllm.platforms import current_platform

device = torch.device("xpu")
DUMP_PATH = "/home/edgeai/applications.ai.gpu.vllm-xpu/xpu_fp8_assert_dump_1778481474998.pt"
VL_CANDIDATES = (128, 256, 512)
KS_CANDIDATES = (1, 2, 4, 8, 10)
SUPPORTED_GEMV_CONFIGS = {
    (512, 1),
    (512, 2),
    (256, 1),
    (256, 2),
    (256, 4),
    (256, 8),
    (128, 1),
    (128, 2),
    (128, 4),
    (128, 8),
    (128, 10),
}


def _normalize_vl_ks(K: int, vl: int, ks: int) -> tuple[int, int]:
    kpt = K // ks
    while vl > kpt or kpt % vl != 0:
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
        kpt = K // ks
    return vl, ks


def _select_vl_ks_impl(N: int, K: int, *, k256_vl: int, k256_ks: int) -> tuple[int, int]:
    if K < 256:
        vl, ks = 128, 1
    elif K == 256:
        vl, ks = k256_vl, k256_ks
    elif K >= 10240:
        vl, ks = 512, 2
    elif K >= 4096:
        vl, ks = 512, 2
    elif K >= 2560 and N >= 10240:
        vl, ks = 512, 1
    elif K >= 2560:
        vl, ks = 128, 10
    elif K >= 2048:
        vl, ks = 256, 8
    else:
        vl, ks = 512, 1

    vl, ks = _normalize_vl_ks(K, vl, ks)
    return vl, ks


def select_vl_ks_pern(N: int, K: int) -> tuple[int, int]:
    vl, ks = _select_vl_ks_impl(N, K, k256_vl=128, k256_ks=1)
    if (vl, ks) not in _valid_vl_ks(K):
        raise ValueError(f"No valid pern vl/ks for N={N}, K={K}")
    return vl, ks


def select_vl_ks_pert(N: int, K: int) -> tuple[int, int]:
    vl, ks = _select_vl_ks_impl(N, K, k256_vl=256, k256_ks=1)
    if (vl, ks) not in _valid_vl_ks(K):
        raise ValueError(f"No valid pert vl/ks for N={N}, K={K}")
    return vl, ks


def select_vl_ks(N: int, K: int) -> tuple[int, int]:
    return select_vl_ks_pern(N, K)


def _valid_vl_ks(K: int) -> list[tuple[int, int]]:
    valid = []
    for vl in VL_CANDIDATES:
        if K % vl != 0:
            continue
        for ks in KS_CANDIDATES:
            if (vl, ks) not in SUPPORTED_GEMV_CONFIGS:
                continue
            if K % ks != 0:
                continue
            if (K // ks) % vl != 0:
                continue
            valid.append((vl, ks))
    return valid


def _run_gemv_auto(input_t, weight_fp8, scale, output):
    from custom_esimd_kernels_vllm import esimd_gemv_fp8

    esimd_gemv_fp8(input_t, weight_fp8, scale, output)
    return output


def _benchmark_one(run_fn, output_fn, iters: int) -> tuple[float, list[float]]:
    for _ in range(10):
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
    elapsed_us = (time.perf_counter() - t0) * 1e6 / iters
    gc.collect()
    if hasattr(torch.xpu, "empty_cache"):
        torch.xpu.empty_cache()
    return elapsed_us, outputs


def _assert_output_lists_close(esimd_outputs: list[float], ref_outputs: list[float]) -> None:
    assert len(esimd_outputs) == len(ref_outputs), (
        f"output list length mismatch: esimd={len(esimd_outputs)} ref={len(ref_outputs)}"
    )
    for index, (esimd_out, ref_out) in enumerate(zip(esimd_outputs, ref_outputs)):
        max_diff = abs(esimd_out - ref_out)
        scale = max(abs(esimd_out), abs(ref_out), 1e-6)
        assert max_diff < 1.0 or (max_diff / scale) < 0.05, (
            f"benchmark output mismatch at iter={index}, diff={max_diff:.4f}"
        )


def test_correctness_basic():
    """Basic correctness: scale=1, compare dequant(fp8)*input vs kernel output."""
    print("\n--- Correctness (scale=1) ---")
    for N, K in [(3072, 2560), (2560, 2048), (20480, 2560), (2560, 10240),
                 (256, 2560), (2048, 128), (3072, 2048), (16, 2048)]:
        weight_ref = torch.randn(N, K, dtype=torch.float16, device=device) * 0.1
        weight_fp8 = weight_ref.to(torch.float8_e4m3fn)
        scale = torch.ones(N, dtype=torch.float16, device=device)

        input_t = torch.randn(1, K, dtype=torch.float16, device=device) * 0.1
        output = torch.zeros(1, N, dtype=torch.float16, device=device)

        _run_gemv_auto(input_t, weight_fp8, scale, output)

        weight_dequant = weight_fp8.to(torch.float16)
        ref = input_t.float() @ weight_dequant.float().T

        max_diff = (output.float() - ref.float()).abs().max().item()
        ref_max = ref.float().abs().max().item()
        rel_err = (max_diff / ref_max) if ref_max > 1e-6 else 0
        ok = max_diff < 1.0 or rel_err < 0.02
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] N={N:5d} K={K:5d}  max_diff={max_diff:.4f}  rel={rel_err:.4f}")
        assert ok, f"Correctness failed for N={N}, K={K}"


def test_correctness_with_scale():
    """Correctness with non-trivial per-N scale."""
    print("\n--- Correctness (with scale) ---")
    for N, K in [(3072, 2560), (2560, 2048), (20480, 2560), (2560, 10240), (128, 2048), (2560,10752)]:
        weight_ref = torch.randn(N, K, dtype=torch.float16, device=device) * 0.1
        weight_fp8 = weight_ref.to(torch.float8_e4m3fn)
        scale = torch.randn(N, dtype=torch.float16, device=device) * 0.1
        #print(scale.shape, scale.dtype)
        input_t = torch.rand(1, K, dtype=torch.float16, device=device) * 0.1
        output = torch.zeros(1, N, dtype=torch.float16, device=device)

        _run_gemv_auto(input_t, weight_fp8, scale, output)

        weight_dequant = weight_fp8.to(torch.float16)
        ref = (input_t.float() @ weight_dequant.float().T) * scale.float().unsqueeze(0)
        output = output.to(torch.bfloat16)
        max_diff = (output.float() - ref.float()).abs().max().item()
        ref_max = ref.float().abs().max().item()
        rel_err = (max_diff / ref_max) if ref_max > 1e-6 else 0
        ok = max_diff < 0.5 or rel_err < 0.05
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] N={N:5d} K={K:5d} (with scale)  max_diff={max_diff:.4f}  rel={rel_err:.4f}")
        assert ok, f"Correctness failed for N={N}, K={K} with scale"



def test_pern_replay_from_assert_dump():
    """Replay a captured failure case for unified esimd_gemv_fp8."""

    dump = torch.load(DUMP_PATH, map_location="cpu")
    input_t = dump["x"].to(device=device).to(torch.float16)
    weight_fp8 = dump["weight"].to(device=device)
    scale = dump["weight_scale"].reshape(-1).to(device=device).to(torch.float16)
    ref = dump["output"].to(device=device)

    N, K = weight_fp8.shape
    if scale.numel() == 1:
        scale = scale.repeat(N)

    output = torch.zeros_like(ref, device=device).to(torch.float16)
    print("input_t shape:", input_t.shape, "dtype:", input_t.dtype)
    print("input_t min/max:", input_t.float().min().item(), input_t.float().max().item())
    print("weight_fp8 shape:", weight_fp8.shape, "dtype:", weight_fp8.dtype)
    print(
        "weight_fp8 min/max:",
        weight_fp8.float().min().item(),
        weight_fp8.float().max().item(),
    )
    print("scale shape:", scale.shape, "dtype:", scale.dtype)
    print("output shape:", output.shape, "dtype:", output.dtype)
    print("ref shape:", ref.shape, "dtype:", ref.dtype)
    _run_gemv_auto(input_t, weight_fp8, scale, output)

    max_diff = (output.float() - ref.float()).abs().max().item()
    ref_max = ref.float().abs().max().item()
    rel_err = (max_diff / ref_max) if ref_max > 1e-6 else 0
    ok = max_diff < 0.5 or rel_err < 0.05
    status = "PASS" if ok else "FAIL"

    print(
        f"  [{status}] replay dump N={N:5d} K={K:5d} "
        f"scale_numel={scale.numel()} max_diff={max_diff:.4f} rel={rel_err:.4f}"
    )
    assert ok, (
        "Replay from assert dump failed for esimd_gemv_fp8: "
        f"max_diff={max_diff:.4f}, rel_err={rel_err:.4f}"
    )


def benchmark_shapes():
    """Benchmark gemma-4-E4B-it shape."""
    from custom_esimd_kernels_vllm import esimd_gemv_fp8_pern

    shapes = [
        ("qkv_proj",     3072, 2560),
        ("qkv_proj",     6144, 2560),
        ("Attn o_proj",  2560, 2048),
        ("Attn o_proj",  2560, 4096),
        ("gate_up_proj", 20480, 2560),
        ("down_proj",    2560, 10240),
        ("per_layer_input_gate",  256, 2560),
        ("per_layer_input_gate_out",     2560, 256),
    ]

    TARGET_BW = 112.0  # GB/s PTL

    print(f"\n{'Shape':<30} {'N':>6} {'K':>6} {'Config':>17} | {'GB/s':>8} {'BW%':>7} {'us':>8}")
    print("-" * 70)

    for name, N, K in shapes:
        weight_ref = torch.randn(N, K, dtype=torch.float16, device=device) * 0.1
        weight_fp8 = weight_ref.to(torch.float8_e4m3fn)
        scale = torch.randn(N, dtype=torch.float16, device=device) * 0.1
        input_t = torch.randn(1, K, dtype=torch.float16, device=device) * 0.1
        output = torch.zeros(1, N, dtype=torch.float16, device=device)

        # Total bytes: input(K*2) + weight(N*K) + scale(N*2) + output(N*2)
        total_bytes = K * 2 + N * K + N * 2 + N * 2

        # Cache-bust: create multiple weight copies
        wb = N * K
        target_mem = 32 * 1024 * 1024
        nc = max(16, target_mem // max(wb, 1))
        nc = min(nc, 512)

        weights = [weight_fp8]
        for i in range(1, nc):
            w = torch.randn(N, K, dtype=torch.float16, device=device) * 0.1
            weights.append(w.to(torch.float8_e4m3fn))

        ni = 4000 if total_bytes < 512 * 1024 else (1000 if total_bytes < 2 * 1024 * 1024 else 300)

        heuristic_vl, heuristic_ks = select_vl_ks_pern(N, K)
        best_cfg = None
        best_us = None
        rule_us = None

        for vl, ks in _valid_vl_ks(K):
            candidate_us, _ = _benchmark_one(
                lambda index, vl=vl, ks=ks: esimd_gemv_fp8_pern(
                    input_t,
                    weights[index % nc],
                    scale,
                    output,
                    N,
                    K,
                    vl,
                    ks,
                ),
                lambda: output,
                ni,
            )
            if (vl, ks) == (heuristic_vl, heuristic_ks):
                rule_us = candidate_us
            if best_us is None or candidate_us < best_us:
                best_cfg = (vl, ks)
                best_us = candidate_us

        assert best_cfg is not None and rule_us is not None
        ms = best_us / 1000
        bw = (total_bytes / 1e9) / (ms / 1e3)
        us = ms * 1000
        bw_pct = bw / TARGET_BW * 100

        print(
            f"{name:<30} {N:>6} {K:>6} "
            f"rule={heuristic_vl}:{heuristic_ks}({rule_us:>7.2f}us) best={best_cfg[0]}:{best_cfg[1]}({best_us:>7.2f}us) | "
            f"{bw:>7.1f} {bw_pct:>6.1f}% {us:>7.2f}"
        )


def _search_best_pern_config(input_t, weights, scale_t, output_t, N: int, K: int, iters: int):
    from custom_esimd_kernels_vllm import esimd_gemv_fp8_pern

    heuristic_vl, heuristic_ks = select_vl_ks_pern(N, K)
    latency_tolerance_us = 0.2
    best_cfg = None
    best_outputs = None
    best_us = None
    rule_us = None
    candidate_records = []

    for vl, ks in _valid_vl_ks(K):
        candidate_us, candidate_outputs = _benchmark_one(
            lambda index, vl=vl, ks=ks: esimd_gemv_fp8_pern(
                input_t,
                weights[index % len(weights)],
                scale_t,
                output_t,
                N,
                K,
                vl,
                ks,
            ),
            lambda: output_t,
            iters,
        )
        candidate_records.append(((vl, ks), candidate_us))
        if (vl, ks) == (heuristic_vl, heuristic_ks):
            rule_us = candidate_us
        if best_us is None or candidate_us < (best_us - latency_tolerance_us):
            best_cfg = (vl, ks)
            best_outputs = candidate_outputs
            best_us = candidate_us

    assert best_cfg is not None and best_outputs is not None and best_us is not None and rule_us is not None
    return best_cfg, best_us, best_outputs, (heuristic_vl, heuristic_ks), rule_us, candidate_records


def _search_best_pert_config(input_t, weights, scale_t, output_t, N: int, K: int, iters: int):
    from custom_esimd_kernels_vllm import esimd_gemv_fp8_pert

    heuristic_vl, heuristic_ks = select_vl_ks_pert(N, K)
    latency_tolerance_us = 0.2
    best_cfg = None
    best_outputs = None
    best_us = None
    rule_us = None
    candidate_records = []

    for vl, ks in _valid_vl_ks(K):
        candidate_us, candidate_outputs = _benchmark_one(
            lambda index, vl=vl, ks=ks: esimd_gemv_fp8_pert(
                input_t,
                weights[index % len(weights)],
                scale_t,
                output_t,
                N,
                K,
                vl,
                ks,
            ),
            lambda: output_t,
            iters,
        )
        candidate_records.append(((vl, ks), candidate_us))
        if (vl, ks) == (heuristic_vl, heuristic_ks):
            rule_us = candidate_us
        if best_us is None or candidate_us < (best_us - latency_tolerance_us):
            best_cfg = (vl, ks)
            best_outputs = candidate_outputs
            best_us = candidate_us

    assert best_cfg is not None and best_outputs is not None and best_us is not None and rule_us is not None
    return best_cfg, best_us, best_outputs, (heuristic_vl, heuristic_ks), rule_us, candidate_records


def _candidate_contains_cfg(candidate_records, cfg: tuple[int, int]) -> bool:
    return any(candidate_cfg == cfg for candidate_cfg, _ in candidate_records)

def test_esimd_vs_vllm():
    TARGET_BW = 112.0  # GB/s PTL

    print(
        f"\n{'Case':<30} {'Config':>20} {'Mode':>8} | {'ESIMD us':>10} {'vllm us':>10} {'ESIMD TF':>10} {'vllm TF':>10} {'ESIMD GB/s':>12} {'vllm GB/s':>11} {'Speedup':>8}"
    )
    print("-" * 138)

    shapes = [
        ("qkv_proj",     3072, 2560),                               # 128/10
        ("qkv_proj",     6144, 2560),                               # 128/10
        ("Attn o_proj",  2560, 2048),                               # 256/8
        ("Attn o_proj",  2560, 4096),                               # 256/4
        ("gate_up_proj", 20480, 2560),                              # 512/1
        ("down_proj",    2560, 10240),                              # 128/10
        ("per_layer_input_gate",  256, 2560),                       # 128/10
        ("per_layer_input_gate_out",     2560, 256),                # 128/2
    ]

    dtype_cases = [
        ("fp16", torch.float16),
        ("bf16", torch.bfloat16),
    ]


    for name, N, K in shapes:
        weight_ref = torch.randn(N, K, dtype=torch.float16, device=device) * 0.1
        weight_fp8 = weight_ref.to(torch.float8_e4m3fn)
        scale = torch.randn(N, dtype=torch.float16, device=device) * 0.1

        wb = N * K
        target_mem = 32 * 1024 * 1024
        nc = max(16, target_mem // max(wb, 1))
        nc = min(nc, 512)

        weights = [weight_fp8]
        for i in range(1, nc):
            w = torch.randn(N, K, dtype=torch.float16, device=device) * 0.1
            weights.append(w.to(torch.float8_e4m3fn))

        for dtype_name, io_dtype in dtype_cases:
            input_t = torch.randn(1, K, dtype=io_dtype, device=device) * 0.1
            pern_output = torch.zeros(1, N, dtype=io_dtype, device=device)
            pert_output = torch.zeros(1, N, dtype=io_dtype, device=device)

            # Total bytes: input + weight + scale + output
            element_bytes = torch.tensor([], dtype=io_dtype).element_size()
            pern_bytes = K * element_bytes + N * K + N * 2 + N * element_bytes
            pert_bytes = K * element_bytes + N * K + 4 + N * element_bytes

            config = f"N={N} K={K} {dtype_name}"
            ni = 1000
            pern_vllm_output = [torch.zeros(1, N, dtype=io_dtype, device=device)]
            pern_vllm_us, pern_vllm_outputs = _benchmark_one(
                lambda index: pern_vllm_output.__setitem__(
                    0,
                    torch.ops._xpu_C.fp8_gemm_w8a16(
                        input_t, weights[index % nc].t(), scale, None
                    ),
                ),
                lambda: pern_vllm_output[0],
                ni,
            )

            best_cfg, pern_us, pern_outputs, rule_cfg, rule_us, pern_candidates = _search_best_pern_config(
                input_t,
                weights,
                scale,
                pern_output,
                N,
                K,
                ni,
            )
            pern_rule_in_search = _candidate_contains_cfg(pern_candidates, rule_cfg)
            _assert_output_lists_close(pern_outputs, pern_vllm_outputs)

            pert_scale_value = 0.05 + torch.rand(1).item() * 0.1
            pert_scale = torch.tensor(pert_scale_value, dtype=torch.float32, device=device)
            pert_scale_vllm = torch.full((N,), pert_scale_value, dtype=torch.float16, device=device)

            pert_vllm_output = [torch.zeros(1, N, dtype=io_dtype, device=device)]
            pert_vllm_us, pert_vllm_outputs = _benchmark_one(
                lambda index: pert_vllm_output.__setitem__(
                    0,
                    torch.ops._xpu_C.fp8_gemm_w8a16(
                        input_t, weights[index % nc].t(), pert_scale_vllm, None
                    ),
                ),
                lambda: pert_vllm_output[0],
                ni,
            )
            pert_best_cfg, pert_us, pert_outputs, pert_rule_cfg, pert_rule_us, pert_candidates = _search_best_pert_config(
                input_t,
                weights,
                pert_scale,
                pert_output,
                N,
                K,
                ni,
            )
            pert_rule_in_search = _candidate_contains_cfg(pert_candidates, pert_rule_cfg)
            _assert_output_lists_close(pert_outputs, pert_vllm_outputs)

            flops = 2 * N * K
            pern_tflops = flops / (pern_us * 1e6) if pern_us > 0 else 0
            pern_vllm_tflops = flops / (pern_vllm_us * 1e6) if pern_vllm_us > 0 else 0
            pern_bw = (pern_bytes / 1e9) / (pern_us / 1e6) if pern_us > 0 else 0
            pern_vllm_bw = (pern_bytes / 1e9) / (pern_vllm_us / 1e6) if pern_vllm_us > 0 else 0

            print(
                f"{name:<30} {config:>20} {'pern':>8} | {pern_us:>9.2f} {pern_vllm_us:>9.2f} "
                f"{pern_tflops:>9.4f} {pern_vllm_tflops:>9.4f} {pern_bw:>11.2f} {pern_vllm_bw:>11.2f} {(pern_vllm_us / pern_us) if pern_us > 0 else 0:>7.2f}x"
            )
            print(
                f"{'':<30} {'':>20} {'cfg':>8} | rule={rule_cfg[0]}:{rule_cfg[1]}({rule_us:>7.2f}us) best={best_cfg[0]}:{best_cfg[1]}({pern_us:>7.2f}us) in_search={pern_rule_in_search}"
            )
            print(
                f"{'':<30} {'':>20} {'all':>8} | "
                f"{' '.join(f'{cfg[0]}:{cfg[1]}={latency:0.2f}us' for cfg, latency in pern_candidates)}"
            )
            assert pern_rule_in_search, f"pern auto cfg {rule_cfg} missing from candidate list"

            pert_tflops = flops / (pert_us * 1e6) if pert_us > 0 else 0
            pert_vllm_tflops = flops / (pert_vllm_us * 1e6) if pert_vllm_us > 0 else 0
            pert_bw = (pert_bytes / 1e9) / (pert_us / 1e6) if pert_us > 0 else 0
            pert_vllm_bw = (pern_bytes / 1e9) / (pert_vllm_us / 1e6) if pert_vllm_us > 0 else 0
            print(
                f"{name:<30} {config:>20} {'pert':>8} | {pert_us:>9.2f} {pert_vllm_us:>9.2f} "
                f"{pert_tflops:>9.4f} {pert_vllm_tflops:>9.4f} {pert_bw:>11.2f} {pert_vllm_bw:>11.2f} {(pert_vllm_us / pert_us) if pert_us > 0 else 0:>7.2f}x"
            )
            print(
                f"{'':<30} {'':>20} {'cfg':>8} | rule={pert_rule_cfg[0]}:{pert_rule_cfg[1]}({pert_rule_us:>7.2f}us) best={pert_best_cfg[0]}:{pert_best_cfg[1]}({pert_us:>7.2f}us) in_search={pert_rule_in_search}"
            )
            print(
                f"{'':<30} {'':>20} {'all':>8} | "
                f"{' '.join(f'{cfg[0]}:{cfg[1]}={latency:0.2f}us' for cfg, latency in pert_candidates)}"
            )
            assert pert_rule_in_search, f"pert auto cfg {pert_rule_cfg} missing from candidate list"


def benchmark_fused():
    """Benchmark fused vs sum-of-individual latencies for target Qwen3 shapes."""
    from custom_esimd_kernels_vllm import (
        esimd_gemv_fp8_pern, 
    )

    TARGET_BW = 112.0  # GB/s PTL

    print(f"\n{'Case':<20} {'Config':>20} | {'Indiv us':>10} {'Fused us':>10} {'Speedup':>8}")
    print("-" * 78)

    def make_tensors(N, K):
        w = (torch.randn(N, K, dtype=torch.float16, device=device) * 0.1).to(torch.float8_e4m3fn)
        s = torch.randn(N, dtype=torch.float16, device=device) * 0.1
        o = torch.zeros(1, N, dtype=torch.float16, device=device)
        return w, s, o

    cases_fused2 = [
        ("DN qkvz+ba",   [(3072, 2048), (16, 2048)]),
        ("Exp gate+up",  [(512, 2048), (512, 2048)]),
        ("Sh gate+up",   [(128, 2048), (128, 2048)]),
    ]

    ni = 1000

    for name, shapes in cases_fused2:
        K = shapes[0][1]
        input_t = torch.randn(1, K, dtype=torch.float16, device=device) * 0.1
        w0, s0, o0 = make_tensors(shapes[0][0], K)
        w1, s1, o1 = make_tensors(shapes[1][0], K)
        config = f"N=[{shapes[0][0]},{shapes[1][0]}] K={K}"
        vl0, ks0 = select_vl_ks(shapes[0][0], K)
        vl1, ks1 = select_vl_ks(shapes[1][0], K)

        # Warmup + bench individual
        for _ in range(10):
            esimd_gemv_fp8_pern(input_t, w0, s0, o0, shapes[0][0], K, vl0, ks0)
            esimd_gemv_fp8_pern(input_t, w1, s1, o1, shapes[1][0], K, vl1, ks1)
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        for _ in range(ni):
            esimd_gemv_fp8_pern(input_t, w0, s0, o0, shapes[0][0], K, vl0, ks0)
            esimd_gemv_fp8_pern(input_t, w1, s1, o1, shapes[1][0], K, vl1, ks1)
        torch.xpu.synchronize()
        indiv_us = (time.perf_counter() - t0) / ni * 1e6

        fused_us = (time.perf_counter() - t0) / ni * 1e6

        speedup = indiv_us / fused_us if fused_us > 0 else 0
        print(f"{name:<20} {config:>20} | {indiv_us:>9.2f} {speedup:>7.2f}x")


def test_pert_correctness():
    """Per-tensor scale: compare against FP16 dequant reference."""
    print("\n--- Per-tensor scale Correctness ---")
    for N, K in [(1024, 1024), (2560, 2048), (512, 2048), (3072, 2048),
                 (128, 2048), (16, 2048), (2048, 512)]:
        weight_ref = torch.randn(N, K, dtype=torch.float16, device=device) * 0.1
        weight_fp8 = weight_ref.to(torch.float8_e4m3fn)
        scale_val = 0.05 + torch.rand(1).item() * 0.1  # random fp32 scalar
        scale_t = torch.tensor(scale_val, dtype=torch.float32, device=device)

        input_t = torch.randn(1, K, dtype=torch.float16, device=device) * 0.1
        output = torch.zeros(1, N, dtype=torch.float16, device=device)

        _run_gemv_auto(input_t, weight_fp8, scale_t, output)

        weight_dequant = weight_fp8.to(torch.float16)
        ref = (input_t.float() @ weight_dequant.float().T) * scale_val

        max_diff = (output.float() - ref.float()).abs().max().item()
        ref_max = ref.float().abs().max().item()
        rel_err = (max_diff / ref_max) if ref_max > 1e-6 else 0
        ok = max_diff < 0.5 or rel_err < 0.05
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] N={N:5d} K={K:5d}  scale={scale_val:.4f}  max_diff={max_diff:.4f}  rel={rel_err:.4f}")
        assert ok, f"Per-tensor correctness failed for N={N}, K={K}"


def test_pert_vs_pern():
    """Per-tensor scale should match per-N when all per-N scales are the same."""
    print("\n--- Per-tensor vs Per-N (uniform scale) ---")
    for N, K in [(2560, 2048), (512, 2048), (128, 2048)]:
        weight_ref = torch.randn(N, K, dtype=torch.float16, device=device) * 0.1
        weight_fp8 = weight_ref.to(torch.float8_e4m3fn)
        scale_val = 0.073
        scale_pern = torch.full((N,), scale_val, dtype=torch.float16, device=device)
        scale_pert = torch.tensor(scale_val, dtype=torch.float32, device=device)

        input_t = torch.randn(1, K, dtype=torch.float16, device=device) * 0.1

        out_pern = torch.zeros(1, N, dtype=torch.float16, device=device)
        out_pert = torch.zeros(1, N, dtype=torch.float16, device=device)

        _run_gemv_auto(input_t, weight_fp8, scale_pern, out_pern)
        _run_gemv_auto(input_t, weight_fp8, scale_pert, out_pert)

        # Not bit-identical due to fp16 vs fp32 scale precision, but should be very close
        max_diff = (out_pert.float() - out_pern.float()).abs().max().item()
        ref_max = out_pern.float().abs().max().item()
        rel_err = (max_diff / ref_max) if ref_max > 1e-6 else 0
        ok = rel_err < 0.01
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] N={N:5d} K={K:5d}  max_diff={max_diff:.6f}  rel={rel_err:.6f}")
        assert ok, f"pert vs pern mismatch for N={N}, K={K}"


def test_e5m2_correctness_pern():
    """E5M2 per-N scale correctness: compare against FP16 dequant reference."""
    print("\n--- E5M2 Per-N Correctness ---")
    for N, K in [(1024, 1024), (2560, 2048), (512, 2048), (128, 2048), (16, 2048)]:
        weight_ref = torch.randn(N, K, dtype=torch.float16, device=device) * 0.1
        weight_fp8 = weight_ref.to(torch.float8_e5m2)
        scale = torch.randn(N, dtype=torch.float16, device=device) * 0.1

        input_t = torch.randn(1, K, dtype=torch.float16, device=device) * 0.1
        output = torch.zeros(1, N, dtype=torch.float16, device=device)

        _run_gemv_auto(input_t, weight_fp8, scale, output)

        weight_dequant = weight_fp8.to(torch.float16)
        ref = (input_t.float() @ weight_dequant.float().T) * scale.float().unsqueeze(0)

        max_diff = (output.float() - ref.float()).abs().max().item()
        ref_max = ref.float().abs().max().item()
        rel_err = (max_diff / ref_max) if ref_max > 1e-6 else 0
        ok = max_diff < 0.5 or rel_err < 0.05
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] N={N:5d} K={K:5d}  max_diff={max_diff:.4f}  rel={rel_err:.4f}")
        assert ok, f"E5M2 pern correctness failed for N={N}, K={K}"


def test_e5m2_correctness_pert():
    """E5M2 per-tensor scale correctness."""
    print("\n--- E5M2 Per-tensor Correctness ---")
    for N, K in [(1024, 1024), (2560, 2048), (512, 2048), (128, 2048)]:
        weight_ref = torch.randn(N, K, dtype=torch.float16, device=device) * 0.1
        weight_fp8 = weight_ref.to(torch.float8_e5m2)
        scale_val = 0.05 + torch.rand(1).item() * 0.1
        scale_t = torch.tensor(scale_val, dtype=torch.float32, device=device)

        input_t = torch.randn(1, K, dtype=torch.float16, device=device) * 0.1
        output = torch.zeros(1, N, dtype=torch.float16, device=device)

        _run_gemv_auto(input_t, weight_fp8, scale_t, output)

        weight_dequant = weight_fp8.to(torch.float16)
        ref = (input_t.float() @ weight_dequant.float().T) * scale_val

        max_diff = (output.float() - ref.float()).abs().max().item()
        ref_max = ref.float().abs().max().item()
        rel_err = (max_diff / ref_max) if ref_max > 1e-6 else 0
        ok = max_diff < 0.5 or rel_err < 0.05
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] N={N:5d} K={K:5d}  scale={scale_val:.4f}  max_diff={max_diff:.4f}  rel={rel_err:.4f}")
        assert ok, f"E5M2 pert correctness failed for N={N}, K={K}"


def test_e5m2_fused():
    """E5M2 fused correctness: fused2 pern + fused2 pert."""
    print("\n--- E5M2 Fused Correctness ---")
    N0, N1, K = 512, 512, 2048
    input_t = torch.randn(1, K, dtype=torch.float16, device=device) * 0.1

    # Fused2 per-N with E5M2
    w0 = (torch.randn(N0, K, dtype=torch.float16, device=device) * 0.1).to(torch.float8_e5m2)
    s0 = torch.randn(N0, dtype=torch.float16, device=device) * 0.1
    w1 = (torch.randn(N1, K, dtype=torch.float16, device=device) * 0.1).to(torch.float8_e5m2)
    s1 = torch.randn(N1, dtype=torch.float16, device=device) * 0.1

    ref_o0 = torch.zeros(1, N0, dtype=torch.float16, device=device)
    ref_o1 = torch.zeros(1, N1, dtype=torch.float16, device=device)
    _run_gemv_auto(input_t, w0, s0, ref_o0)
    _run_gemv_auto(input_t, w1, s1, ref_o1)

   

    # Fused2 per-tensor with E5M2
    st0 = torch.tensor(0.08, dtype=torch.float32, device=device)
    st1 = torch.tensor(0.12, dtype=torch.float32, device=device)

    ref_o0 = torch.zeros(1, N0, dtype=torch.float16, device=device)
    ref_o1 = torch.zeros(1, N1, dtype=torch.float16, device=device)
    _run_gemv_auto(input_t, w0, st0, ref_o0)
    _run_gemv_auto(input_t, w1, st1, ref_o1)


def benchmark_e5m2():
    """Benchmark E5M2 vs E4M3 on key shapes with cache-busting buffer rotation."""
    from custom_esimd_kernels_vllm import esimd_gemv_fp8_pern

    shapes = [
        ("qkv_proj",     3072, 2560),
        ("Attn o_proj",  2560, 2048),
        ("gate_up_proj", 20480, 2560),
        ("down_proj",    2560, 10240),
        ("per_layer_input_gate",  256, 2560),
        ("per_layer_input_gate_out",     2560, 256),
    ]

    TARGET_BW = 450.0

    print(f"\n{'Shape':<30} {'N':>6} {'K':>6} {'Config':>17} | {'E4M3 us':>9} {'E5M2 us':>9} {'E4M3 GB/s':>10} {'E5M2 GB/s':>10}")
    print("-" * 80)

    for name, N, K in shapes:
        total_bytes = K * 2 + N * K + N * 2 + N * 2
        input_t = torch.randn(1, K, dtype=torch.float16, device=device) * 0.1
        scale = torch.randn(N, dtype=torch.float16, device=device) * 0.1
        output = torch.zeros(1, N, dtype=torch.float16, device=device)

        # Cache-bust: rotate through enough buffers to exceed L3
        wb = N * K
        target_mem = 32 * 1024 * 1024
        nc = max(16, target_mem // max(wb, 1))
        nc = min(nc, 512)

        ni = 4000 if total_bytes < 512 * 1024 else (1000 if total_bytes < 2 * 1024 * 1024 else 300)

        results = {}
        heuristic_vl, heuristic_ks = select_vl_ks(N, K)
        for dtype_name, dtype in [("E4M3", torch.float8_e4m3fn), ("E5M2", torch.float8_e5m2)]:
            weights = []
            for i in range(nc):
                w = (torch.randn(N, K, dtype=torch.float16, device=device) * 0.1).to(dtype)
                weights.append(w)

            best_us = None
            best_cfg = None
            rule_us = None
            for vl, ks in _valid_vl_ks(K):
                candidate_us, _ = _benchmark_one(
                    lambda index, vl=vl, ks=ks: esimd_gemv_fp8_pern(
                        input_t,
                        weights[index % nc],
                        scale,
                        output,
                        N,
                        K,
                        vl,
                        ks,
                    ),
                    lambda: output,
                    ni,
                )
                if (vl, ks) == (heuristic_vl, heuristic_ks):
                    rule_us = candidate_us
                if best_us is None or candidate_us < best_us:
                    best_us = candidate_us
                    best_cfg = (vl, ks)

            assert best_us is not None and best_cfg is not None and rule_us is not None
            us = best_us
            bw = (total_bytes / 1e9) / (us / 1e6)
            results[dtype_name] = (us, bw, best_cfg, rule_us)

        e4_us, e4_bw, e4_cfg, e4_rule_us = results["E4M3"]
        e5_us, e5_bw, e5_cfg, e5_rule_us = results["E5M2"]
        print(
            f"{name:<30} {N:>6} {K:>6} rule={heuristic_vl}:{heuristic_ks}(E4/E5={e4_rule_us:>7.2f}/{e5_rule_us:>7.2f}us) best(E4/E5)={e4_cfg[0]}:{e4_cfg[1]}({e4_us:>7.2f}us)/{e5_cfg[0]}:{e5_cfg[1]}({e5_us:>7.2f}us) | "
            f"{e4_us:>8.2f} {e5_us:>8.2f} {e4_bw:>9.1f} {e5_bw:>9.1f}"
        )


if __name__ == "__main__":
    print("=" * 60)
    print("custom-esimd-kernels-vllm: GEMV FP8 Tests")
    print("=" * 60)

    # E4M3 per-N scale tests
    # test_correctness_basic()
    # test_correctness_with_scale()
    # test_pern_replay_from_assert_dump()
    test_esimd_vs_vllm()
    # E4M3 per-tensor scale tests
    # test_pert_correctness()
    # test_pert_vs_pern()

    # # E5M2 tests
    # test_e5m2_correctness_pern()
    # test_e5m2_correctness_pert()

    # # Performance
    # print("\n--- Performance Benchmark (unfused per-N, E4M3) ---")
    # benchmark_shapes()


    # print("\n--- Performance Benchmark (E4M3 vs E5M2) ---")
    # benchmark_e5m2()

    # print("\n" + "=" * 60)
    # print("ALL TESTS PASSED")
    # print("=" * 60)