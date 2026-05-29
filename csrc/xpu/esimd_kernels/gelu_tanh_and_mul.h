#pragma once

#include "utils.h"

// GeGLU activation: GELU(tanh approximation) on the first half, then multiply
// by the second half. Input is [M, 2D], output is [M, D].

// Step down through the small set of ks values that this kernel benchmarks well
// on, mirroring the fixed-choice selector style used by the GEMV kernels.
inline int gelu_tanh_and_mul_step_down_ks(int value) {
    if (value == 64) return 40;
    if (value == 40) return 32;
    if (value == 32) return 20;
    if (value == 20) return 16;
    if (value == 16) return 10;
    if (value == 10) return 8;
    if (value == 8) return 5;
    if (value == 5) return 4;
    if (value == 4) return 2;
    if (value == 2) return 1;
    return value;
}

// Normalize a tentative vl/ks pair so it matches the row width and launch
// limits. vl must divide D, and ks must not exceed either chunk count or the
// device work-group limit used by this kernel.
inline void normalize_gelu_tanh_and_mul_vl_ks(uint32_t half_cols, int& vl, int& ks) {
    while (vl > 128 && (half_cols % vl != 0 || static_cast<uint32_t>(vl) > half_cols)) {
        vl /= 2;
    }

    if (half_cols % vl != 0 || static_cast<uint32_t>(vl) > half_cols) {
        return;
    }

    int chunks_per_row = static_cast<int>(half_cols / vl);
    int max_ks = chunks_per_row < 64 ? chunks_per_row : 64;
    while (ks > max_ks) {
        int next_ks = gelu_tanh_and_mul_step_down_ks(ks);
        if (next_ks == ks) {
            break;
        }
        ks = next_ks;
    }
}

// Heuristic selector for Gemma4-like GeGLU shapes. Unlike GEMV, ks is not a
// reduction split here; it is the per-row thread count used to stripe chunks of
// the D dimension across a work-group.
inline void select_vl_ks_gelu_tanh_and_mul(uint32_t rows, uint32_t cols, bool is_bf16, int& vl, int& ks) {
    uint32_t half_cols = cols / 2;

    if (rows <= 1) {
        vl = 256;
        ks = 20;
    } else if (rows <= 2) {
        vl = is_bf16 ? 256 : 128;
        ks = is_bf16 ? 40 : 32;
    } else if (rows <= 8) {
        vl = 128;
        ks = 40;
    } else if (rows <= 16) {
        vl = is_bf16 ? 256 : 128;
        ks = is_bf16 ? 20 : 40;
    } else if (rows <= 24) {
        vl = is_bf16 ? 128 : 256;
        ks = is_bf16 ? 40 : 20;
    } else {
        vl = 256;
        ks = is_bf16 ? 16 : 20;
    }

    normalize_gelu_tanh_and_mul_vl_ks(half_cols, vl, ks);
}

template<typename scalar_t, int VL>
struct GeluTanhAndMulKernel {
    const scalar_t* input_ptr;
    scalar_t* output_ptr;
    int rows;
    int cols;
    int half_cols;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        int chunks_per_row = half_cols / VL;
        int row = item.get_group(0);
        int lid = item.get_local_id(0);
        if (row >= rows) return;

        int input_row_base = row * cols;
        int output_row_base = row * half_cols;

        for (int chunk = lid; chunk < chunks_per_row; chunk += item.get_local_range(0)) {
            int offset = chunk * VL;
            simd<float, VL> gate = block_load<scalar_t, VL>(input_ptr + input_row_base + offset);
            simd<float, VL> up = block_load<scalar_t, VL>(input_ptr + input_row_base + half_cols + offset);

            constexpr float kAlpha = 0.7978845608028654f;
            constexpr float kBeta = 0.044715f;
            simd<float, VL> gate_sq = gate * gate;
            simd<float, VL> gate_cube = gate_sq * gate;
            simd<float, VL> tanh_arg = (gate + gate_cube * kBeta) * kAlpha;
            simd<float, VL> clamped_tanh_arg = tanh_arg;
            clamped_tanh_arg.merge(10.0f, tanh_arg > 10.0f);
            clamped_tanh_arg.merge(-10.0f, tanh_arg < -10.0f);
            simd<float, VL> exp_2x = exp(clamped_tanh_arg * 2.0f);
            simd<float, VL> tanh_val = (exp_2x - 1.0f) / (exp_2x + 1.0f);
            simd<float, VL> gelu = 0.5f * gate * (1.0f + tanh_val);
            simd<float, VL> out = gelu * up;

            block_store<scalar_t, VL>(output_ptr + output_row_base + offset, simd<scalar_t, VL>(out));
        }
    }
};

template<typename scalar_t>
inline void gelu_tanh_and_mul_host_impl(
    const scalar_t* input_ptr,
    scalar_t* output_ptr,
    int rows,
    int cols,
    int vl,
    int ks,
    sycl::queue& q)
{
    int half_cols = cols / 2;
    int global = rows * ks;
    int local = ks;

    #define LAUNCH_GELU_TANH_AND_MUL(V) \
        q.submit([&](sycl::handler& cgh) { \
            cgh.parallel_for( \
                sycl::nd_range<1>(global, local), \
                GeluTanhAndMulKernel<scalar_t, V>{input_ptr, output_ptr, rows, cols, half_cols}); \
        });

    if (vl == 512) { LAUNCH_GELU_TANH_AND_MUL(512) }
    else if (vl == 256) { LAUNCH_GELU_TANH_AND_MUL(256) }
    else { LAUNCH_GELU_TANH_AND_MUL(128) }

    #undef LAUNCH_GELU_TANH_AND_MUL
}

// Explicit vl/ks path: use the caller-provided launch shape as-is after any
// validation performed by the C++ op entrypoint.
inline void gelu_tanh_and_mul_host(
    const uint8_t* input_ptr,
    uint8_t* output_ptr,
    int rows,
    int cols,
    int vl,
    int ks,
    bool is_bf16,
    sycl::queue& q)
{
    if (is_bf16) {
        gelu_tanh_and_mul_host_impl<bf16>(
            reinterpret_cast<const bf16*>(input_ptr),
            reinterpret_cast<bf16*>(output_ptr),
            rows,
            cols,
            vl,
            ks,
            q);
    } else {
        gelu_tanh_and_mul_host_impl<fp16>(
            reinterpret_cast<const fp16*>(input_ptr),
            reinterpret_cast<fp16*>(output_ptr),
            rows,
            cols,
            vl,
            ks,
            q);
    }
}

// Auto-select path: derive vl/ks from shape and dtype, then forward to the
// explicit launch helper.
inline void gelu_tanh_and_mul_host(
    const uint8_t* input_ptr,
    uint8_t* output_ptr,
    int rows,
    int cols,
    bool is_bf16,
    sycl::queue& q)
{
    int vl = 128;
    int ks = 1;
    select_vl_ks_gelu_tanh_and_mul(rows, cols, is_bf16, vl, ks);
    gelu_tanh_and_mul_host(input_ptr, output_ptr, rows, cols, vl, ks, is_bf16, q);
}