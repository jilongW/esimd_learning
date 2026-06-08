#!/usr/bin/env python3
from pathlib import Path
import time

import torch

EPS = 1e-6
from vllm.platforms import current_platform

def _reshape_heads(x: torch.Tensor, n_heads: int, head_dim: int) -> torch.Tensor:
    return x.view(x.shape[0], n_heads, head_dim)


def _rms_with_weight(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    out = torch.empty_like(x)
    torch.ops._C.rms_norm(out, x, weight, eps)
    return out


def _rms_no_weight(x: torch.Tensor, eps: float) -> torch.Tensor:
    out = torch.empty_like(x)
    unit_weight = torch.ones((x.shape[-1],), dtype=x.dtype, device=x.device)
    torch.ops._C.rms_norm(out, x, unit_weight, eps)
    return out


def _identity_rope_cache(max_pos: int, rotary_dim: int, device: torch.device) -> torch.Tensor:
    half = rotary_dim // 2
    cos = torch.ones(max_pos, half, dtype=torch.float16, device=device)
    sin = torch.zeros(max_pos, half, dtype=torch.float16, device=device)
    return torch.cat([cos, sin], dim=-1)


def _max_mean_abs_diff(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    diff = (a.float() - b.float()).abs()
    return diff.max(), diff.mean()


def _estimate_bw_gbps(total_bytes: int, avg_ms: float) -> float:
    if avg_ms <= 0.0:
        return 0.0
    return total_bytes / (avg_ms * 1e-3) / 1e9


def _make_random_case(
    kind: str,
    n_tokens: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    is_kv_shared_layer: bool | None,
) -> dict[str, torch.Tensor | int | float | bool]:
    if is_kv_shared_layer is None:
        is_kv_shared_layer = kind == "sliding"

    q_size = q_heads * head_dim
    kv_size = kv_heads * head_dim
    qkv = torch.randn((n_tokens, q_size + 2 * kv_size), dtype=torch.float16, device="xpu").contiguous()
    q_norm_weight = (1.0 + 0.01 * torch.randn((head_dim,), dtype=torch.float16, device="xpu")).contiguous()
    k_norm_weight = (1.0 + 0.01 * torch.randn((head_dim,), dtype=torch.float16, device="xpu")).contiguous()

    return {
        "is_kv_shared_layer": is_kv_shared_layer,
        "q_size": q_size,
        "kv_size": kv_size,
        "qkv": qkv,
        "q_norm_weight": q_norm_weight,
        "k_norm_weight": k_norm_weight,
        "q_norm_eps": EPS,
        "k_norm_eps": EPS,
        "v_norm_eps": EPS,
        "positions": torch.zeros((n_tokens,), dtype=torch.int32, device="xpu"),
        "rope_cache": _identity_rope_cache(max_pos=1, rotary_dim=head_dim, device=torch.device("xpu")),
        "head_dim": head_dim,
        "q_heads": q_heads,
        "kv_heads": kv_heads,
        "n_tokens": n_tokens,
    }


def _run_torch_reference(case: dict[str, torch.Tensor | int | float | bool]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    qkv = case["qkv"]
    q_heads = int(case["q_heads"])
    kv_heads = int(case["kv_heads"])
    head_dim = int(case["head_dim"])
    q_size = int(case["q_size"])
    kv_size = int(case["kv_size"])
    is_kv_shared_layer = bool(case["is_kv_shared_layer"])
    q_norm_weight = case["q_norm_weight"]
    k_norm_weight = case["k_norm_weight"]
    q_norm_eps = float(case["q_norm_eps"])
    k_norm_eps = float(case["k_norm_eps"])
    v_norm_eps = float(case["v_norm_eps"])

    q_raw, k_raw, v_raw = qkv.split([q_size, kv_size, kv_size], dim=-1)

    q_ref = _rms_with_weight(_reshape_heads(q_raw, q_heads, head_dim), q_norm_weight, q_norm_eps).flatten(-2, -1)

    if is_kv_shared_layer:
        k_ref = k_raw
        v_ref = v_raw
    else:
        k_ref = _rms_with_weight(_reshape_heads(k_raw, kv_heads, head_dim), k_norm_weight, k_norm_eps).flatten(-2, -1)
        v_ref = _rms_no_weight(_reshape_heads(v_raw, kv_heads, head_dim), v_norm_eps).flatten(-2, -1)

    return q_ref, k_ref, v_ref


def benchmark_two_paths_perf_and_bandwidth(
    kind: str = "full",
    warmup: int = 20,
    iters: int = 100,
    n_tokens: int = 4096,
    q_heads: int = 8,
    kv_heads: int = 4,
    head_dim: int = 256,
    is_kv_shared_layer: bool | None = None,
) -> None:
    from custom_esimd_kernels_vllm import esimd_qkv_split_norm_rope_gemma

    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        raise RuntimeError("XPU is not available for benchmark")

    case = _make_random_case(kind, n_tokens, q_heads, kv_heads, head_dim, is_kv_shared_layer)
    qkv = case["qkv"]
    q_size = int(case["q_size"])
    kv_size = int(case["kv_size"])
    q_heads = int(case["q_heads"])
    kv_heads = int(case["kv_heads"])
    head_dim = int(case["head_dim"])
    is_kv_shared_layer = bool(case["is_kv_shared_layer"])
    q_norm_weight = case["q_norm_weight"]
    k_norm_weight = case["k_norm_weight"]

    norm_wq_xpu = q_norm_weight.contiguous()
    norm_wk_xpu = k_norm_weight.contiguous()

    q_out = torch.empty((n_tokens, q_size), dtype=torch.float16, device="xpu")
    k_out = torch.empty((n_tokens, kv_size), dtype=torch.float16, device="xpu")
    v_out = torch.empty((n_tokens, kv_size), dtype=torch.float16, device="xpu")

    def _run_esimd_once() -> None:
        esimd_qkv_split_norm_rope_gemma(
            qkv,
            q_out,
            k_out,
            v_out,
            norm_wq_xpu,
            norm_wk_xpu,
            case["positions"],
            q_heads,
            kv_heads,
            head_dim,
            is_kv_shared_layer,
            case["rope_cache"],
        )

    def _run_torch_once() -> None:
        q_ref, k_ref, v_ref = _run_torch_reference(case)
        q_out.copy_(q_ref)
        k_out.copy_(k_ref)
        v_out.copy_(v_ref)

    for _ in range(warmup):
        _run_esimd_once()
    torch.xpu.synchronize()

    for _ in range(warmup):
        _run_torch_once()
    torch.xpu.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        _run_esimd_once()
    torch.xpu.synchronize()
    esimd_ms = (time.perf_counter() - t0) * 1000.0 / iters

    t0 = time.perf_counter()
    for _ in range(iters):
        _run_torch_once()
    torch.xpu.synchronize()
    torch_ms = (time.perf_counter() - t0) * 1000.0 / iters

    bytes_per_elem = 2
    qkv_elems = n_tokens * (q_size + 2 * kv_size)
    out_elems = n_tokens * (q_size + 2 * kv_size)
    norm_elems = head_dim * 2
    rope_elems = head_dim
    pos_elems = n_tokens
    total_bytes = bytes_per_elem * (qkv_elems + out_elems + norm_elems + rope_elems + pos_elems)

    esimd_bw = _estimate_bw_gbps(total_bytes, esimd_ms)
    torch_bw = _estimate_bw_gbps(total_bytes, torch_ms)

    print(f"[bench:{kind}, n_tokens={n_tokens}] ESIMD: {esimd_ms:.4f} ms/iter, approx BW={esimd_bw:.2f} GB/s")
    print(f"[bench:{kind}, n_tokens={n_tokens}] TORCH: {torch_ms:.4f} ms/iter, approx BW={torch_bw:.2f} GB/s")
    if torch_ms > 0:
        print(f"[bench:{kind}, n_tokens={n_tokens}] speedup (torch/esimd): {torch_ms / esimd_ms:.3f}x")


def test_esimd_qkv_split_norm_rope_gemma_vs_random() -> None:
    from custom_esimd_kernels_vllm import esimd_qkv_split_norm_rope_gemma

    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        raise RuntimeError("XPU is not available for this test")

    for kind in ("full", "sliding"):
        for n_tokens in (4096, 1):
            case = _make_random_case(kind, n_tokens, q_heads=8, kv_heads=4, head_dim=256, is_kv_shared_layer=None)
            qkv = case["qkv"]
            q_size = int(case["q_size"])
            kv_size = int(case["kv_size"])
            q_heads = int(case["q_heads"])
            kv_heads = int(case["kv_heads"])
            head_dim = int(case["head_dim"])
            is_kv_shared_layer = bool(case["is_kv_shared_layer"])
            q_norm_weight = case["q_norm_weight"]
            k_norm_weight = case["k_norm_weight"]

            norm_wq_xpu = q_norm_weight.contiguous()
            norm_wk_xpu = k_norm_weight.contiguous()

            q_out = torch.empty((n_tokens, q_size), dtype=torch.float16, device="xpu")
            k_out = torch.empty((n_tokens, kv_size), dtype=torch.float16, device="xpu")
            v_out = torch.empty((n_tokens, kv_size), dtype=torch.float16, device="xpu")

            esimd_qkv_split_norm_rope_gemma(
                qkv,
                q_out,
                k_out,
                v_out,
                norm_wq_xpu,
                norm_wk_xpu,
                case["positions"],
                q_heads,
                kv_heads,
                head_dim,
                is_kv_shared_layer,
                case["rope_cache"],
            )
            torch.xpu.synchronize()

            q_ref, k_ref, v_ref = _run_torch_reference(case)
            q_max, q_mean = _max_mean_abs_diff(q_out, q_ref)
            k_max, k_mean = _max_mean_abs_diff(k_out, k_ref)
            v_max, v_mean = _max_mean_abs_diff(v_out, v_ref)

            print(f"[acc:{kind}, n_tokens={n_tokens}] Q diff: max={q_max.item():.6f}, mean={q_mean.item():.6f}")
            print(f"[acc:{kind}, n_tokens={n_tokens}] K diff: max={k_max.item():.6f}, mean={k_mean.item():.6f}")
            print(f"[acc:{kind}, n_tokens={n_tokens}] V diff: max={v_max.item():.6f}, mean={v_mean.item():.6f}")

            assert bool((q_max < 0.2).item()), f"{kind} n_tokens={n_tokens} Q max diff too large: {q_max.item()}"
            assert bool((k_max < 0.2).item()), f"{kind} n_tokens={n_tokens} K max diff too large: {k_max.item()}"
            assert bool((v_max < 0.2).item()), f"{kind} n_tokens={n_tokens} V max diff too large: {v_max.item()}"


def main() -> None:
    test_esimd_qkv_split_norm_rope_gemma_vs_random()
    for n_tokens in (4096, 1):
        benchmark_two_paths_perf_and_bandwidth(kind="full", n_tokens=n_tokens)
        benchmark_two_paths_perf_and_bandwidth(kind="sliding", n_tokens=n_tokens)


if __name__ == "__main__":
    main()
