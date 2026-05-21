"""Benchmark the standalone FP16 OpenCL GEMM from cm.gemm.examples.kernels.

This test is opt-in because it depends on an external OpenCL runtime plus a
CM kernel binary. Enable it with CM_GEMM_RUN=1.

Environment variables:
  CM_GEMM_RUN=1
      Enable the benchmark test.
    CM_GEMM_LIB=/abs/path/to/libcm_fp16_gemm.so
      Use an existing shared library.
  CM_GEMM_KERNEL_BIN=/abs/path/to/kernel.cm.bin
      Use an existing kernel binary.
  CM_GEMM_CASES=5120x2560x5120x100x512x256;2048x2048x2048x200
      Semicolon-separated list of MxNxKxiters[xTileMxTileN] cases.
"""

from __future__ import annotations

import ctypes
import argparse
import os
import re
import time
from pathlib import Path
from vllm.platforms import current_platform
import numpy as np


ESIMD_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ESIMD_ROOT.parent
CM_GEMM_DIR = WORKSPACE_ROOT / "cm.gemm.examples.kernels" / "standalone" / "fp16.gemm"
BUILD_DIR = CM_GEMM_DIR / "build_pytest"
LIB_NAME = "libcm_fp16_gemm.so"


def _env_flag(name: str) -> bool:
    value = os.getenv(name)
    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}


def _tool_path(env_name: str) -> Path | None:
    tool = os.getenv(env_name)
    if not tool:
        return None
    path = Path(tool).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{env_name} points to a missing file: {path}")
    return path


def _default_library() -> Path:
    return BUILD_DIR / LIB_NAME


def _default_kernel_bin() -> Path:
    return BUILD_DIR / "kernel.cm.bin"


def _resolve_library_path() -> Path:
    explicit = _tool_path("CM_GEMM_LIB")
    if explicit is not None:
        return explicit

    target = _default_library()
    if target.is_file():
        return target

    raise FileNotFoundError("standalone FP16 GEMM library not found; set CM_GEMM_LIB")


def _load_library() -> ctypes.CDLL:
    library_path = _resolve_library_path()
    library = ctypes.CDLL(str(library_path))
    library.cm_fp16_gemm_run.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
    ]
    library.cm_fp16_gemm_run.restype = ctypes.c_int
    return library


def _resolve_kernel_bin_path() -> Path:
    explicit = _tool_path("CM_GEMM_KERNEL_BIN")
    if explicit is not None:
        return explicit

    target = _default_kernel_bin()
    if target.is_file():
        return target

    raise FileNotFoundError("kernel.cm.bin not found; set CM_GEMM_KERNEL_BIN")


def _parse_case(token: str) -> tuple[int, int, int, int, int, int, int, int]:
    parts = [piece for piece in re.split(r"[x,:]", token.strip()) if piece]
    if len(parts) not in {4, 6, 8}:
        raise ValueError(
            "CM_GEMM_CASES entries must be MxNxKxiters, MxNxKxitersxTileMxTileN, "
            "or MxNxKxitersxTileMxTileNxSubTileMxSubTileN"
        )
    values = [int(piece) for piece in parts]
    if len(values) == 4:
        values.extend([512, 256, 32, 64])
    elif len(values) == 6:
        values.extend([32, 64])
    return tuple(values)  # type: ignore[return-value]


def _benchmark_cases() -> list[tuple[int, int, int, int, int, int, int, int]]:
    spec = os.getenv("CM_GEMM_CASES")
    if not spec:
        return [(5120, 2560, 5120, 100, 512, 256, 32, 64)]
    return [_parse_case(token) for token in spec.split(";") if token.strip()]


def _prepare_artifacts() -> tuple[ctypes.CDLL, Path]:
    kernel_bin = _resolve_kernel_bin_path()
    return _load_library(), kernel_bin


def _candidate_sub_tiles() -> tuple[int, ...]:
    return (8, 16, 32, 64)


def _candidate_tiles() -> tuple[int, ...]:
    return (8, 16, 32, 64, 128, 256, 512)


def _iter_valid_cm_configs(m: int, n: int) -> list[tuple[int, int, int, int]]:
    configs: list[tuple[int, int, int, int]] = []
    for sub_tile_m in _candidate_sub_tiles():
        for sub_tile_n in _candidate_sub_tiles():
            if sub_tile_m == 64 and sub_tile_n == 64:
                continue
            for tile_m in _candidate_tiles():
                if tile_m < sub_tile_m or tile_m % sub_tile_m != 0:
                    continue
                for tile_n in _candidate_tiles():
                    if tile_n < sub_tile_n or tile_n % sub_tile_n != 0:
                        continue
                    configs.append((tile_m, tile_n, sub_tile_m, sub_tile_n))
    configs.sort(key=lambda item: (item[0] * item[1], item[2] * item[3], item[0], item[1]))
    return configs


def _find_best_cm_gemm_config(
    library: ctypes.CDLL,
    kernel_bin: Path,
    host_a: np.ndarray,
    host_b: np.ndarray,
    host_c: np.ndarray,
    m: int,
    n: int,
    k: int,
) -> tuple[int, int, int, int]:
    best_config: tuple[int, int, int, int] | None = None
    best_ms: float | None = None
    ref = _reference_output(host_a, host_b, m, n, k)

    for tile_m, tile_n, sub_tile_m, sub_tile_n in _iter_valid_cm_configs(m, n):
        host_c.fill(0)
        try:
            _run_cm_fp16_gemm_once(
                library,
                kernel_bin,
                host_a,
                host_b,
                host_c,
                m,
                n,
                k,
                tile_m,
                tile_n,
                sub_tile_m,
                sub_tile_n,
            )
            _verify_output_against_ref(host_c, ref, m, n, k)
        except RuntimeError:
            continue
        except AssertionError:
            continue

        start = time.perf_counter()
        host_c.fill(0)
        _run_cm_fp16_gemm_once(
            library,
            kernel_bin,
            host_a,
            host_b,
            host_c,
            m,
            n,
            k,
            tile_m,
            tile_n,
            sub_tile_m,
            sub_tile_n,
        )
        elapsed_ms = (time.perf_counter() - start) * 1e3
        if best_ms is None or elapsed_ms < best_ms:
            best_ms = elapsed_ms
            best_config = (tile_m, tile_n, sub_tile_m, sub_tile_n)

    if best_config is None:
        raise RuntimeError(
            f"no valid FP16 CM GEMM tile/sub-tile config found for shape m={m} n={n} k={k}"
        )
    return best_config


def _splitmix64_next(state: int) -> tuple[int, int]:
    state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return state, (z ^ (z >> 31)) & 0xFFFFFFFFFFFFFFFF


def _fp32_to_fp16_scalar(value: float) -> int:
    return np.float16(value).view(np.uint16).item()


def _fp16_to_fp32_array(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.uint16).view(np.float16).astype(np.float32)


def _fp32_to_fp16_array(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float16).view(np.uint16)


def _torch_to_fp16_host_array(tensor) -> np.ndarray:
    import torch

    values = tensor.detach().to(device="cpu", dtype=torch.float16).contiguous().numpy()
    return values.view(np.uint16).reshape(-1)


def _as_fp16_host_array(values) -> np.ndarray:
    if isinstance(values, np.ndarray):
        if values.dtype == np.uint16 and values.flags.c_contiguous:
            return values.reshape(-1)
        return np.ascontiguousarray(values, dtype=np.float16).view(np.uint16).reshape(-1)

    try:
        import torch

        if isinstance(values, torch.Tensor):
            return _torch_to_fp16_host_array(values)
    except Exception:
        pass

    raise TypeError(f"expected numpy.ndarray or torch.Tensor, got {type(values)!r}")


def _prepare_fp16_output_buffer(values):
    if isinstance(values, np.ndarray):
        if values.dtype == np.uint16 and values.flags.c_contiguous:
            return values.reshape(-1), None
        host_array = np.ascontiguousarray(values, dtype=np.float16).view(np.uint16).reshape(-1)
        return host_array, None

    try:
        import torch

        if isinstance(values, torch.Tensor):
            if values.dtype not in {torch.float16, torch.uint16}:
                raise TypeError(
                    f"expected output tensor dtype torch.float16 or torch.uint16, got {values.dtype}"
                )

            host_array = np.zeros(values.numel(), dtype=np.uint16)
            output_shape = tuple(values.shape)
            output_dtype = values.dtype

            def sync_back() -> None:
                if output_dtype == torch.float16:
                    cpu_tensor = torch.from_numpy(host_array.view(np.float16).reshape(output_shape))
                else:
                    cpu_tensor = torch.from_numpy(host_array.reshape(output_shape))
                values.copy_(cpu_tensor.to(device=values.device))

            return host_array, sync_back
    except Exception:
        pass

    raise TypeError(f"expected numpy.ndarray or torch.Tensor, got {type(values)!r}")


def _initialize_matrix(rows: int, cols: int, base_seed: int) -> np.ndarray:
    out = np.empty(rows * cols, dtype=np.uint16)
    for row in range(rows):
        state = (base_seed + row) & 0xFFFFFFFFFFFFFFFF
        row_offset = row * cols
        for col in range(cols):
            state, rnd = _splitmix64_next(state)
            u = float((rnd >> 40) & 0xFFFFFF) * (1.0 / 16777216.0)
            out[row_offset + col] = _fp32_to_fp16_scalar(u * 2.0 - 1.0)
    return out


def _initialize_inputs(m: int, n: int, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    host_a = _initialize_matrix(m, k, 0x9E3779B97F4A7C15)
    host_b = _initialize_matrix(n, k, 0xBF58476D1CE4E5B9)
    host_c = np.zeros(m * n, dtype=np.uint16)
    return host_a, host_b, host_c


def _pad_cm_input_rows(host_a: np.ndarray, m: int, n: int, k: int) -> tuple[np.ndarray, np.ndarray, int]:
    min_m = min(_candidate_sub_tiles())
    padded_m = max(m, min_m)
    if padded_m == m:
        return host_a, np.zeros(m * n, dtype=np.uint16), m

    padded_a = np.zeros(padded_m * k, dtype=np.uint16)
    padded_a[: m * k] = host_a
    padded_c = np.zeros(padded_m * n, dtype=np.uint16)
    return padded_a, padded_c, padded_m


def _reference_output(host_a: np.ndarray, host_b: np.ndarray, m: int, n: int, k: int) -> np.ndarray:
    matrix_a = _fp16_to_fp32_array(host_a).reshape(m, k)
    matrix_b = _fp16_to_fp32_array(host_b).reshape(n, k)
    return matrix_a @ matrix_b.T


def _verify_output_against_ref(host_c: np.ndarray, ref: np.ndarray, m: int, n: int, k: int) -> None:
    rtol = max(2e-2, (1.0 / 512.0) * (k ** 0.5) * 4.0)
    atol = max(2e-2, (1.0 / 512.0) * (k ** 0.5) * 4.0)
    gpu = _fp16_to_fp32_array(host_c).reshape(n, m).T
    if not np.allclose(gpu, ref, rtol=rtol, atol=atol):
        abs_err = np.abs(gpu - ref)
        max_flat = int(abs_err.argmax())
        max_i, max_j = np.unravel_index(max_flat, abs_err.shape)
        raise AssertionError(
            "Correctness check failed: "
            f"idx=({max_i},{max_j}) gpu={float(gpu[max_i, max_j]):.6f} "
            f"ref={float(ref[max_i, max_j]):.6f} "
            f"abs_err={float(abs_err[max_i, max_j]):.6e} "
            f"limit={float(atol + rtol * abs(ref[max_i, max_j])):.6e}"
        )


def _verify_output(host_a: np.ndarray, host_b: np.ndarray, host_c: np.ndarray, m: int, n: int, k: int) -> None:
    ref = _reference_output(host_a, host_b, m, n, k)
    _verify_output_against_ref(host_c, ref, m, n, k)


def _run_cm_fp16_gemm_once(
    library: ctypes.CDLL,
    kernel_bin: Path,
    host_a,
    host_b,
    host_c,
    m: int,
    n: int,
    k: int,
    tile_m: int,
    tile_n: int,
    sub_tile_m: int,
    sub_tile_n: int,
) -> None:
    host_a_array = _as_fp16_host_array(host_a)
    host_b_array = _as_fp16_host_array(host_b)
    host_c_array, sync_output = _prepare_fp16_output_buffer(host_c)
    ret = library.cm_fp16_gemm_run(
        str(kernel_bin).encode(),
        host_a_array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        host_b_array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        host_c_array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        m,
        n,
        k,
        tile_m,
        tile_n,
        sub_tile_m,
        sub_tile_n,
    )
    if ret != 0:
        raise RuntimeError(
            "standalone FP16 GEMM function call failed\n"
            f"kernel_bin: {kernel_bin}\n"
            f"return_code: {ret}"
        )
    if sync_output is not None:
        sync_output()


def run_cm_fp16_gemm_benchmark(
    m: int,
    n: int,
    k: int,
    num_iters: int,
    tile_m: int = 512,
    tile_n: int = 256,
    sub_tile_m: int = 32,
    sub_tile_n: int = 64,
) -> tuple[float, float]:
    library, kernel_bin = _prepare_artifacts()
    host_a, host_b, host_c = _initialize_inputs(m, n, k)
    _run_cm_fp16_gemm_once(
        library, kernel_bin, host_a, host_b, host_c, m, n, k, tile_m, tile_n, sub_tile_m, sub_tile_n
    )
    _verify_output(host_a, host_b, host_c, m, n, k)

    warmup_iters = min(3, max(num_iters, 1))
    for _ in range(warmup_iters):
        host_c.fill(0)
        _run_cm_fp16_gemm_once(
            library, kernel_bin, host_a, host_b, host_c, m, n, k, tile_m, tile_n, sub_tile_m, sub_tile_n
        )

    total_iters = max(num_iters, 1)
    start = time.perf_counter()
    for _ in range(total_iters):
        host_c.fill(0)
        _run_cm_fp16_gemm_once(
            library, kernel_bin, host_a, host_b, host_c, m, n, k, tile_m, tile_n, sub_tile_m, sub_tile_n
        )
    total_ms = (time.perf_counter() - start) * 1e3
    avg_ms = total_ms / total_iters
    return avg_ms, total_ms


def run_cm_fp16_gemm_benchmarks() -> None:
    if not CM_GEMM_DIR.is_dir():
        raise FileNotFoundError(f"cm.gemm.examples.kernels not found: {CM_GEMM_DIR}")

    print(
        f"\n{'M':>8} {'N':>8} {'K':>8} {'iters':>8} {'tile_m':>8} {'tile_n':>8} {'sub_m':>8} {'sub_n':>8} | {'avg ms':>10} {'total ms':>10}"
    )
    print("-" * 104)
    for m, n, k, num_iters, tile_m, tile_n, sub_tile_m, sub_tile_n in _benchmark_cases():
        avg_ms, total_ms = run_cm_fp16_gemm_benchmark(m, n, k, num_iters, tile_m, tile_n, sub_tile_m, sub_tile_n)
        print(
            f"{m:>8} {n:>8} {k:>8} {num_iters:>8} {tile_m:>8} {tile_n:>8} {sub_tile_m:>8} {sub_tile_n:>8} | "
            f"{avg_ms:>10.3f} {total_ms:>10.3f}"
        )


def run_cm_vs_esimd_gemv_vs_vllm_vs_esimd_gemm() -> None:
    if not CM_GEMM_DIR.is_dir():
        raise FileNotFoundError(f"cm.gemm.examples.kernels not found: {CM_GEMM_DIR}")
    if not _env_flag("CM_GEMM_RUN"):
        raise RuntimeError("set CM_GEMM_RUN=1 to enable the standalone FP16 GEMM comparison")

    import torch

    try:
        from custom_esimd_kernels_vllm import esimd_gemm_fp8_pert, esimd_gemv_fp8_pern
    except Exception as exc:
        raise RuntimeError(f"custom_esimd_kernels_vllm is unavailable: {exc}") from exc

    if not torch.xpu.is_available():
        raise RuntimeError("XPU device is not available")
    if not hasattr(torch.ops, "_xpu_C") or not hasattr(torch.ops._xpu_C, "fp8_gemm_w8a16"):
        raise RuntimeError("torch.ops._xpu_C.fp8_gemm_w8a16 is unavailable")

    device = torch.device("xpu")
    library, kernel_bin = _prepare_artifacts()

    shapes = [
        ("qkv_proj", 3072, 2560),
        ("qkv_proj", 6144, 2560),
        ("attn_o_proj", 2560, 2048),
        ("attn_o_proj", 2560, 4096),
        ("gate_up_proj", 20480, 2560),
        ("down_proj", 2560, 10240),
        ("per_layer_input_gate", 256, 2560),
        ("per_layer_input_gate_out", 2560, 256),
    ]

    # print(
    #     f"\n{'Case':<22} {'Config':>18} | {'CM us':>9} {'ESIMD GEMV':>11} {'vLLM':>9} {'ESIMD GEMM':>11} {'CM/vLLM':>9} {'GEMV/vLLM':>11} {'GEMM/vLLM':>11}"
    # )
    # print("-" * 128)

    cm_best_configs: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for _, n, k in shapes:
        input_t = torch.randn(1, k, dtype=torch.float16, device=device) * 0.1
        weight_ref = torch.randn(n, k, dtype=torch.float16, device=device) * 0.1
        host_a = _torch_to_fp16_host_array(input_t)
        host_b = _torch_to_fp16_host_array(weight_ref)
        cm_host_a, cm_host_c, cm_m = _pad_cm_input_rows(host_a, 1, n, k)
        cm_best_configs[(n, k)] = _find_best_cm_gemm_config(
            library,
            kernel_bin,
            cm_host_a,
            host_b,
            cm_host_c,
            cm_m,
            n,
            k,
        )
    print(
        f"\n{'Case':<22} {'Config':>18} | {'CM us':>9} {'ESIMD GEMV':>11} {'vLLM':>9} {'ESIMD GEMM':>11} {'CM/vLLM':>9} {'GEMV/vLLM':>11} {'GEMM/vLLM':>11}"
    )
    print("-" * 128)
    for name, n, k in shapes:
        input_t = torch.randn(1, k, dtype=torch.float16, device=device) * 0.1
        scale_value = 0.073
        scale_pern = torch.full((n,), scale_value, dtype=torch.float16, device=device)
        scale_pert = torch.tensor(scale_value, dtype=torch.float32, device=device)
        out_esimd_gemv = torch.zeros(1, n, dtype=torch.float16, device=device)
        out_esimd_gemm = torch.zeros(1, n, dtype=torch.float16, device=device)

        wb = n * k
        target_mem = 32 * 1024 * 1024
        num_copies = max(16, target_mem // max(wb, 1))
        num_copies = min(num_copies, 256)

        weight_fp8_copies = []
        weight_fp16_copies = []
        for _ in range(num_copies):
            weight_ref = torch.randn(n, k, dtype=torch.float16, device=device) * 0.1
            weight_fp8_copies.append(weight_ref.to(torch.float8_e4m3fn))
            weight_fp16_copies.append(_torch_to_fp16_host_array(weight_ref))

        host_a = _torch_to_fp16_host_array(input_t)
        cm_host_a, cm_host_c, cm_m = _pad_cm_input_rows(host_a, 1, n, k)
        cm_tile_m, cm_tile_n, cm_sub_tile_m, cm_sub_tile_n = cm_best_configs[(n, k)]

        _run_cm_fp16_gemm_once(
            library,
            kernel_bin,
            cm_host_a,
            weight_fp16_copies[0],
            cm_host_c,
            cm_m,
            n,
            k,
            cm_tile_m,
            cm_tile_n,
            cm_sub_tile_m,
            cm_sub_tile_n,
        )
        # _verify_output(cm_host_a, weight_fp16_copies[0], cm_host_c, cm_m, n, k)

        esimd_gemv_fp8_pern(input_t, weight_fp8_copies[0], scale_pern, out_esimd_gemv, n, k)
        out_vllm = torch.ops._xpu_C.fp8_gemm_w8a16(input_t, weight_fp8_copies[0].t(), scale_pern, None)
        esimd_gemm_fp8_pert(input_t, weight_fp8_copies[0], scale_pert, out_esimd_gemm)

        ref_max = max(out_esimd_gemv.float().abs().max().item(), 1e-6)
        vllm_diff = (out_vllm.float() - out_esimd_gemv.float()).abs().max().item()
        gemm_diff = (out_esimd_gemm.float() - out_esimd_gemv.float()).abs().max().item()
        assert vllm_diff < 1.0 or (vllm_diff / ref_max) < 0.05, (
            f"vLLM vs ESIMD GEMV mismatch for N={n}, K={k}: max_diff={vllm_diff:.4f}"
        )
        assert gemm_diff < 1.0 or (gemm_diff / ref_max) < 0.05, (
            f"ESIMD GEMM vs ESIMD GEMV mismatch for N={n}, K={k}: max_diff={gemm_diff:.4f}"
        )

        num_iters = 1000
        cm_host_a_bench, cm_host_c_bench, cm_m_bench = _pad_cm_input_rows(host_a, 1, n, k)
        cm_weight_schedule = [weight_fp16_copies[i % num_copies] for i in range(num_iters)]

        for _ in range(3):
            cm_host_c_bench.fill(0)
            weight_fp16_host = weight_fp16_copies[_ % num_copies]
            _run_cm_fp16_gemm_once(
                library,
                kernel_bin,
                cm_host_a_bench,
                weight_fp16_host,
                cm_host_c_bench,
                cm_m_bench,
                n,
                k,
                cm_tile_m,
                cm_tile_n,
                cm_sub_tile_m,
                cm_sub_tile_n,
            )
        t0 = time.perf_counter()
        for weight_fp16_host in cm_weight_schedule:
            cm_host_c_bench.fill(0)
            _run_cm_fp16_gemm_once(
                library,
                kernel_bin,
                cm_host_a_bench,
                weight_fp16_host,
                cm_host_c_bench,
                cm_m_bench,
                n,
                k,
                cm_tile_m,
                cm_tile_n,
                cm_sub_tile_m,
                cm_sub_tile_n,
            )
        cm_us = (time.perf_counter() - t0) / num_iters * 1e6
        time.sleep(2)
        for i in range(10):
            esimd_gemv_fp8_pern(input_t, weight_fp8_copies[i % num_copies], scale_pern, out_esimd_gemv, n, k)
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        for i in range(num_iters):
            esimd_gemv_fp8_pern(input_t, weight_fp8_copies[i % num_copies], scale_pern, out_esimd_gemv, n, k)
        torch.xpu.synchronize()
        esimd_gemv_us = (time.perf_counter() - t0) / num_iters * 1e6
        time.sleep(2)
        for i in range(10):
            out_vllm = torch.ops._xpu_C.fp8_gemm_w8a16(input_t, weight_fp8_copies[i % num_copies].t(), scale_pern, None)
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        for i in range(num_iters):
            out_vllm = torch.ops._xpu_C.fp8_gemm_w8a16(input_t, weight_fp8_copies[i % num_copies].t(), scale_pern, None)
        torch.xpu.synchronize()
        vllm_us = (time.perf_counter() - t0) / num_iters * 1e6
        time.sleep(2)
        for i in range(10):
            esimd_gemm_fp8_pert(input_t, weight_fp8_copies[i % num_copies], scale_pert, out_esimd_gemm)
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        for i in range(num_iters):
            esimd_gemm_fp8_pert(input_t, weight_fp8_copies[i % num_copies], scale_pert, out_esimd_gemm)
        torch.xpu.synchronize()
        esimd_gemm_us = (time.perf_counter() - t0) / num_iters * 1e6

        print(
            f"{name:<22} {f'N={n} K={k}':>18} | "
            f"{cm_us:>8.2f} {esimd_gemv_us:>10.2f} {vllm_us:>8.2f} {esimd_gemm_us:>10.2f} "
            f"{(cm_us / vllm_us) if vllm_us > 0 else 0.0:>8.2f}x "
            f"{(esimd_gemv_us / vllm_us) if vllm_us > 0 else 0.0:>10.2f}x "
            f"{(esimd_gemm_us / vllm_us) if vllm_us > 0 else 0.0:>10.2f}x"
        )
        print(
            f"{'':<22} {'CM cfg':>18} | tile={cm_tile_m}x{cm_tile_n} sub={cm_sub_tile_m}x{cm_sub_tile_n}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run standalone CM GEMM benchmarks or comparisons")
    parser.add_argument(
        "--mode",
        choices=("benchmark", "compare"),
        default="benchmark",
        help="benchmark: run CM GEMM cases; compare: run CM/ESIMD/vLLM comparison",
    )
    args = parser.parse_args()

    os.environ.setdefault("CM_GEMM_RUN", "1")

    if args.mode == "compare":
        run_cm_vs_esimd_gemv_vs_vllm_vs_esimd_gemm()
    else:
        run_cm_fp16_gemm_benchmarks()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())