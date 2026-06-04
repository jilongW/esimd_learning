#!/usr/bin/env python3
from pathlib import Path
import time

import torch


EPS = 1e-6
DUMP_DIR = Path.home() / "esimd_learning"/ "dumps" 


def describe_tensor(name: str, tensor: torch.Tensor | None) -> None:
    if tensor is None:
        print(f"  {name}: None")
        return
    print(
        f"  {name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}"
    )


def load_and_show(path: Path):
    if not path.exists():
        print(f"[MISSING] {path}")
        return None
    obj = torch.load(path, map_location="cpu")
    print(f"[OK] {path}")
    return obj


def _reshape_heads(x: torch.Tensor, n_heads: int, head_dim: int) -> torch.Tensor:
    return x.view(x.shape[0], n_heads, head_dim)


def _rms_with_weight(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x_f = x.float()
    inv = torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x_f * inv * weight.float()).to(x.dtype)


def _rms_no_weight(x: torch.Tensor, eps: float) -> torch.Tensor:
    x_f = x.float()
    inv = torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x_f * inv).to(x.dtype)


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


def benchmark_two_paths_perf_and_bandwidth(
    kind: str = "full",
    warmup: int = 20,
    iters: int = 100,
) -> None:
    from custom_esimd_kernels_vllm import esimd_qkv_split_norm_rope_gemma

    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        raise RuntimeError("XPU is not available for benchmark")

    base = DUMP_DIR
    qkv = torch.load(base / f"gemma4_{kind}_qkv.pt", map_location="cpu")
    qkv_norm = torch.load(base / f"gemma4_{kind}_qkv_norm.pt", map_location="cpu")
    rope = torch.load(base / f"gemma4_{kind}_rotary_emb.pt", map_location="cpu")

    if not isinstance(qkv, torch.Tensor) or not isinstance(qkv_norm, dict) or not isinstance(rope, dict):
        raise RuntimeError(f"Invalid benchmark dump inputs for kind={kind}")

    is_kv_shared_layer = bool(qkv_norm.get("is_kv_shared_layer"))
    head_dim = int(qkv_norm.get("head_dim") or rope.get("head_size") or rope.get("rotary_dim") or 256)
    rope_rotary_dim = int(rope.get("rotary_dim") or head_dim)
    if rope_rotary_dim != head_dim:
        raise RuntimeError(
            f"Kernel now requires full rotary: rotary_dim({rope_rotary_dim}) != head_dim({head_dim}) for {kind}"
        )

    q_heads = int(qkv_norm.get("num_heads") or 0)
    kv_heads = int(qkv_norm.get("num_kv_heads") or 0)
    if q_heads <= 0 or kv_heads <= 0:
        raise RuntimeError(f"Invalid num_heads/num_kv_heads in qkv_norm dump for {kind}")

    q_norm_weight = qkv_norm.get("q_norm_weight")
    k_norm_weight = qkv_norm.get("k_norm_weight")
    q_norm_eps = float(qkv_norm.get("q_norm_eps") or EPS)
    k_norm_eps = float(qkv_norm.get("k_norm_eps") or EPS)
    v_norm_eps = float(qkv_norm.get("v_norm_eps") or EPS)
    if not isinstance(q_norm_weight, torch.Tensor) or not isinstance(k_norm_weight, torch.Tensor):
        raise RuntimeError(f"Missing q_norm_weight/k_norm_weight for benchmark kind={kind}")

    q_size = q_heads * head_dim
    kv_size = kv_heads * head_dim
    n_tokens = qkv.shape[0]

    qkv_xpu = qkv.to("xpu", dtype=torch.float16).contiguous()
    q_norm_weight_xpu = q_norm_weight.to("xpu", dtype=torch.float16).contiguous()
    k_norm_weight_xpu = k_norm_weight.to("xpu", dtype=torch.float16).contiguous()

    norm_wq_xpu = (q_norm_weight_xpu.float() - 1.0).to(torch.float16).contiguous()
    norm_wk_xpu = (k_norm_weight_xpu.float() - 1.0).to(torch.float16).contiguous()

    positions = torch.zeros((n_tokens,), dtype=torch.int32, device="xpu")
    rope_cache = _identity_rope_cache(max_pos=1, rotary_dim=head_dim, device=torch.device("xpu"))

    q_out = torch.empty((n_tokens, q_size), dtype=torch.float16, device="xpu")
    k_out = torch.empty((n_tokens, kv_size), dtype=torch.float16, device="xpu")
    v_out = torch.empty((n_tokens, kv_size), dtype=torch.float16, device="xpu")

    def _run_esimd_once() -> None:
        esimd_qkv_split_norm_rope_gemma(
            qkv_xpu,
            q_out,
            k_out,
            v_out,
            norm_wq_xpu,
            norm_wk_xpu,
            positions,
            q_heads,
            kv_heads,
            head_dim,
            is_kv_shared_layer,
            rope_cache,
        )

    def _run_torch_once() -> None:
        q_raw, k_raw, v_raw = qkv_xpu.split([q_size, kv_size, kv_size], dim=-1)

        q_ref_h = _rms_with_weight(_reshape_heads(q_raw, q_heads, head_dim), q_norm_weight_xpu, q_norm_eps)
        q_ref = q_ref_h.flatten(-2, -1)

        if is_kv_shared_layer:
            k_ref = k_raw
            v_ref = v_raw
        else:
            k_ref_h = _rms_with_weight(_reshape_heads(k_raw, kv_heads, head_dim), k_norm_weight_xpu, k_norm_eps)
            v_ref_h = _rms_no_weight(_reshape_heads(v_raw, kv_heads, head_dim), v_norm_eps)
            k_ref = k_ref_h.flatten(-2, -1)
            v_ref = v_ref_h.flatten(-2, -1)

        q_out.copy_(q_ref)
        k_out.copy_(k_ref)
        v_out.copy_(v_ref)

    # Warmup
    for _ in range(warmup):
        _run_esimd_once()
    torch.xpu.synchronize()

    for _ in range(warmup):
        _run_torch_once()
    torch.xpu.synchronize()

    # Time ESIMD
    t0 = time.perf_counter()
    for _ in range(iters):
        _run_esimd_once()
    torch.xpu.synchronize()
    esimd_ms = (time.perf_counter() - t0) * 1000.0 / iters

    # Time torch baseline
    t0 = time.perf_counter()
    for _ in range(iters):
        _run_torch_once()
    torch.xpu.synchronize()
    torch_ms = (time.perf_counter() - t0) * 1000.0 / iters

    # Rough traffic model (bytes per call):
    # read qkv + write q/k/v + read norm weights + read rope cache + read positions
    bytes_per_elem = 2  # fp16/int16 scale for dominant tensors
    qkv_elems = n_tokens * (q_size + 2 * kv_size)
    out_elems = n_tokens * (q_size + 2 * kv_size)
    norm_elems = head_dim * 2
    rope_elems = head_dim
    pos_elems = n_tokens
    total_bytes = bytes_per_elem * (qkv_elems + out_elems + norm_elems + rope_elems + pos_elems)

    esimd_bw = _estimate_bw_gbps(total_bytes, esimd_ms)
    torch_bw = _estimate_bw_gbps(total_bytes, torch_ms)

    print(
        f"[bench:{kind}] ESIMD: {esimd_ms:.4f} ms/iter, approx BW={esimd_bw:.2f} GB/s"
    )
    print(
        f"[bench:{kind}] TORCH: {torch_ms:.4f} ms/iter, approx BW={torch_bw:.2f} GB/s"
    )
    if torch_ms > 0:
        print(f"[bench:{kind}] speedup (torch/esimd): {torch_ms / esimd_ms:.3f}x")


def test_esimd_qkv_split_norm_rope_gemma_vs_dump() -> None:
    from custom_esimd_kernels_vllm import esimd_qkv_split_norm_rope_gemma

    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        raise RuntimeError("XPU is not available for this test")

    base = DUMP_DIR
    kinds = ["full", "sliding"]

    for kind in kinds:
        qkv = torch.load(base / f"gemma4_{kind}_qkv.pt", map_location="cpu")
        qkv_norm = torch.load(base / f"gemma4_{kind}_qkv_norm.pt", map_location="cpu")
        rope = torch.load(base / f"gemma4_{kind}_rotary_emb.pt", map_location="cpu")

        if not isinstance(qkv, torch.Tensor):
            raise RuntimeError(f"Invalid qkv dump for {kind}")
        if not isinstance(qkv_norm, dict):
            raise RuntimeError(f"Invalid qkv_norm dump for {kind}")
        if not isinstance(rope, dict):
            raise RuntimeError(f"Invalid rotary dump for {kind}")

        is_kv_shared_layer = bool(qkv_norm.get("is_kv_shared_layer"))

        head_dim = int(qkv_norm.get("head_dim") or rope.get("head_size") or rope.get("rotary_dim") or 256)
        rope_rotary_dim = int(rope.get("rotary_dim") or head_dim)
        if head_dim <= 0:
            raise RuntimeError(f"Invalid head_dim from rotary dump for {kind}")
        if rope_rotary_dim != head_dim:
            raise RuntimeError(
                f"Kernel now requires full rotary: rotary_dim({rope_rotary_dim}) != head_dim({head_dim}) for {kind}"
            )

        q_heads = int(qkv_norm.get("num_heads") or 0)
        kv_heads = int(qkv_norm.get("num_kv_heads") or 0)
        if q_heads <= 0 or kv_heads <= 0:
            raise RuntimeError(f"Invalid num_heads/num_kv_heads in qkv_norm dump for {kind}")

        q_norm_weight = qkv_norm.get("q_norm_weight")
        k_norm_weight = qkv_norm.get("k_norm_weight")
        v_norm_weight = qkv_norm.get("v_norm_weight")
        q_norm_eps = float(qkv_norm.get("q_norm_eps") or EPS)
        k_norm_eps = float(qkv_norm.get("k_norm_eps") or EPS)
        v_norm_eps = float(qkv_norm.get("v_norm_eps") or EPS)

        if not isinstance(q_norm_weight, torch.Tensor):
            raise RuntimeError(f"Missing q_norm_weight dump for {kind}")
        if not isinstance(k_norm_weight, torch.Tensor):
            raise RuntimeError(f"Missing k_norm_weight dump for {kind}")

        if isinstance(v_norm_weight, torch.Tensor):
            raise RuntimeError(f"Expected v_norm has no weight, but found v_norm_weight for {kind}")

        qkv_xpu = qkv.to("xpu", dtype=torch.float16).contiguous()
        q_norm_weight_xpu = q_norm_weight.to("xpu", dtype=torch.float16).contiguous()
        k_norm_weight_xpu = k_norm_weight.to("xpu", dtype=torch.float16).contiguous()

        q_size = q_heads * head_dim
        kv_size = kv_heads * head_dim
        q_raw, k_raw, v_raw = qkv_xpu.split([q_size, kv_size, kv_size], dim=-1)

        q_ref_h = _rms_with_weight(_reshape_heads(q_raw, q_heads, head_dim), q_norm_weight_xpu, q_norm_eps)
        q_ref = q_ref_h.flatten(-2, -1)

        if is_kv_shared_layer:
            k_ref = k_raw
            v_ref = v_raw
        else:
            k_ref_h = _rms_with_weight(_reshape_heads(k_raw, kv_heads, head_dim), k_norm_weight_xpu, k_norm_eps)
            v_ref_h = _rms_no_weight(_reshape_heads(v_raw, kv_heads, head_dim), v_norm_eps)
            k_ref = k_ref_h.flatten(-2, -1)
            v_ref = v_ref_h.flatten(-2, -1)

        # Kernel expects Gemma4 RMSNorm parameter convention: (weight - 1.0)
        norm_wq_xpu = (q_norm_weight_xpu.float() - 1.0).to(torch.float16).contiguous()
        norm_wk_xpu = (k_norm_weight_xpu.float() - 1.0).to(torch.float16).contiguous()

        q_out = torch.empty((qkv.shape[0], q_size), dtype=torch.float16, device="xpu")
        k_out = torch.empty((qkv.shape[0], kv_size), dtype=torch.float16, device="xpu")
        v_out = torch.empty((qkv.shape[0], kv_size), dtype=torch.float16, device="xpu")

        positions = torch.zeros((qkv.shape[0],), dtype=torch.int32, device="xpu")
        rope_cache = _identity_rope_cache(max_pos=1, rotary_dim=head_dim, device=torch.device("xpu"))

        esimd_qkv_split_norm_rope_gemma(
            qkv_xpu,
            q_out,
            k_out,
            v_out,
            norm_wq_xpu,
            norm_wk_xpu,
            positions,
            q_heads,
            kv_heads,
            head_dim,
            is_kv_shared_layer,
            rope_cache,
        )
        torch.xpu.synchronize()

        q_max, q_mean = _max_mean_abs_diff(q_out, q_ref)
        k_max, k_mean = _max_mean_abs_diff(k_out, k_ref)
        v_max, v_mean = _max_mean_abs_diff(v_out, v_ref)

        print(f"[{kind}] Q diff: max={q_max.item():.6f}, mean={q_mean.item():.6f}")
        print(f"[{kind}] K diff: max={k_max.item():.6f}, mean={k_mean.item():.6f}")
        print(f"[{kind}] V diff: max={v_max.item():.6f}, mean={v_mean.item():.6f}")

        assert bool((q_max < 0.2).item()), f"{kind} Q max diff too large: {q_max.item()}"
        assert bool((k_max < 0.2).item()), f"{kind} K max diff too large: {k_max.item()}"
        assert bool((v_max < 0.2).item()), f"{kind} V max diff too large: {v_max.item()}"


def build() -> None:
    base = DUMP_DIR
    kinds = ["full", "sliding"]

    for kind in kinds:
        print(f"\n=== {kind.upper()} ===")

        qkv = load_and_show(base / f"gemma4_{kind}_qkv.pt")
        if isinstance(qkv, torch.Tensor):
            describe_tensor("qkv", qkv)

        qkv_norm = load_and_show(base / f"gemma4_{kind}_qkv_norm.pt")
        if isinstance(qkv_norm, dict):
            print(f"  is_kv_shared_layer: {qkv_norm.get('is_kv_shared_layer')}")
            print(f"  num_heads: {qkv_norm.get('num_heads')}, num_kv_heads: {qkv_norm.get('num_kv_heads')}, head_dim: {qkv_norm.get('head_dim')}")
            describe_tensor("q_norm_weight", qkv_norm.get("q_norm_weight"))
            describe_tensor("k_norm_weight", qkv_norm.get("k_norm_weight"))
            describe_tensor("v_norm_weight", qkv_norm.get("v_norm_weight"))

        rope = load_and_show(base / f"gemma4_{kind}_rotary_emb.pt")
        if isinstance(rope, dict):
            print(
                "  rotary: "
                f"class={rope.get('class_name')}, "
                f"head_size={rope.get('head_size')}, "
                f"rotary_dim={rope.get('rotary_dim')}, "
                f"max_pos={rope.get('max_position_embeddings')}, "
                f"base={rope.get('base')}, "
                f"is_neox_style={rope.get('is_neox_style')}, "
                f"dtype={rope.get('dtype')}"
            )
            describe_tensor("cos_sin_cache", rope.get("cos_sin_cache"))


def main() -> None:
    build()
    test_esimd_qkv_split_norm_rope_gemma_vs_dump()
    benchmark_two_paths_perf_and_bandwidth(kind="full")
    benchmark_two_paths_perf_and_bandwidth(kind="sliding")


if __name__ == "__main__":
    main()
