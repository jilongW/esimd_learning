/* gemv_gelu_tanh_mul_fused.h — FP8 GEMV + GELU(tanh)+MUL fused kernel.
 *
 * This kernel is used in the RMSNorm + GEMV + GELU(tanh) + MUL pipeline.
 * It performs GEMV + GeGLU fusion:
 *   logits0 = x @ dequant(gemv_weight[0:N]^T) * scale0
 *   logits1 = x @ dequant(gemv_weight[N:2N]^T) * scale1
 *   output  = GELU_tanh(logits0) * logits1
 *
 * RMSNorm is applied by upstream kernels.
 *
 * Designed for decode shapes such as:
 *   x:           [1, K] fp16      (e.g. [1, 2560])
 *   gemv_weight: [2N, K] FP8      (e.g. [20480, 2560])
 *   gemv_scale:  [1] or [2] float32
 *   output:      [N] fp16
 *
 * Dispatch is aligned with the tuned GEMV path: VL and K_SPLIT are selected
 * from N/K, and the kernel uses nd_range(global=N*K_SPLIT, local=K_SPLIT).
 */

#pragma once
#include "utils.h"

inline void select_vl_ks_gemv_gelu_tanh_mul_xe3(uint32_t N, uint32_t K, int& vl, int& ks) {
    if (K == 256) {
        vl = 256;
        ks = 1;
    } else if (K <= 2048) {
        vl = 128;
        ks = 8;
    } else if (K <= 2560) {
        vl = 128;
        ks = 10;
    } else if (K <= 4096) {
        vl = 512;
        ks = 2;
    } else {
        vl = 512;
        ks = 2;
    }

    int k_per_thread = (ks > 0) ? (int)(K / ks) : 0;
    while (vl > k_per_thread || (k_per_thread > 0 && k_per_thread % vl != 0)) {
        if (vl > 128) {
            vl /= 2;
        } else if (ks == 10) {
            ks = 8;
        } else if (ks == 8) {
            ks = 4;
        } else if (ks == 4) {
            ks = 2;
        } else if (ks == 2) {
            ks = 1;
        } else {
            break;
        }
        k_per_thread = (ks > 0) ? (int)(K / ks) : 0;
    }
}

inline void select_vl_ks_gemv_gelu_tanh_mul_xe2(uint32_t N, uint32_t K, int& vl, int& ks) {
    select_vl_ks_gemv_gelu_tanh_mul_xe3(N, K, vl, ks);
}

inline void select_vl_ks_gemv_gelu_tanh_mul(
    uint32_t N,
    uint32_t K,
    int& vl,
    int& ks,
    const sycl::device* dev = nullptr) {
    if (dev != nullptr) {
        auto arch = dev->get_info<sycl::ext::oneapi::experimental::info::device::architecture>();
        if (is_ptl_architecture_device(arch)) {
            select_vl_ks_gemv_gelu_tanh_mul_xe3(N, K, vl, ks);
        } else {
            select_vl_ks_gemv_gelu_tanh_mul_xe2(N, K, vl, ks);
        }
        return;
    }
    select_vl_ks_gemv_gelu_tanh_mul_xe2(N, K, vl, ks);
}

template<int VL>
SYCL_ESIMD_FUNCTION inline simd<float, VL> fp8_dequant_gemv_gelu_tanh_mul(
    simd<uint8_t, VL> raw, int fp8_mode) {
    simd<uint16_t, VL> u16 = convert<uint16_t>(raw);
    simd<uint16_t, VL> fp8_sign = (u16 >> 7) & 1;
    simd<uint16_t, VL> fp16_bits;

    if (fp8_mode == 0) {
        simd<uint16_t, VL> fp8_exp  = (u16 >> 3) & 0xF;
        simd<uint16_t, VL> fp8_mant = u16 & 0x7;
        fp16_bits = (fp8_sign << 15) | ((fp8_exp + 8) << 10) | (fp8_mant << 7);
        fp16_bits.merge(fp8_sign << 15, fp8_exp == 0);
    } else {
        simd<uint16_t, VL> fp8_exp  = (u16 >> 2) & 0x1F;
        simd<uint16_t, VL> fp8_mant = u16 & 0x3;
        fp16_bits = (fp8_sign << 15) | (fp8_exp << 10) | (fp8_mant << 8);
        fp16_bits.merge(fp8_sign << 15, fp8_exp == 0);
    }

    simd<fp16, VL> wh = fp16_bits.template bit_cast_view<fp16>().read();
    return simd<float, VL>(wh);
}

SYCL_ESIMD_FUNCTION inline float gelu_tanh_scalar_gemv(float x) {
    constexpr float BETA = M_SQRT2 * M_2_SQRTPI * 0.5f;
    constexpr float KAPPA = 0.044715f;
    float x_cube = x * x * x;
    float inner = BETA * (x + KAPPA * x_cube);
    if (inner > 10.0f) inner = 10.0f;
    if (inner < -10.0f) inner = -10.0f;
    float exp_2x = sycl::exp(inner * 2.0f);
    float tanh_val = (exp_2x - 1.0f) / (exp_2x + 1.0f);
    return 0.5f * x * (1.0f + tanh_val);
}

/* ================================================================
 * Kernel: FP8 GEMV + GeGLU (GELU(tanh)+MUL)
 * - w0: first half rows [0, N)
 * - w1: second half rows [N, 2N)
 * output has N elements.
 * ================================================================ */
constexpr int GEMV_GELU_TANH_MUL_SLM_CACHE_K = 2560;

template<typename scalar_t>
struct GemvGeluTanhMul_fp8_pert_kernel_256_1 {
    const scalar_t* x_ptr;       // [1, K]
    const uint8_t* gemv_weight;  // [N, K] FP8
    const float*   gemv_scale;   // [1] or [2]
    scalar_t*      output;       // [N/2]
    int N;
    int K;
    int scale_count;
    int fp8_mode;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        constexpr int VL = 256;
        int n = item.get_group(0);
        if (n >= N) return;

        int n_chunks = K / VL;
        int n1 = n + N;

        float scale0 = gemv_scale[0];
        float scale1 = (scale_count > 1) ? gemv_scale[1] : gemv_scale[0];

        float acc0 = 0.0f;
        float acc1 = 0.0f;
        for (int c = 0; c < n_chunks; c++) {
            int offset = c * VL;

            simd<float, VL> x = block_load<scalar_t, VL>(x_ptr + offset);
            simd<uint8_t, VL> w0_raw = block_load<uint8_t, VL>(
                gemv_weight + (size_t)n * K + offset);
            simd<float, VL> w0_f = fp8_dequant_gemv_gelu_tanh_mul<VL>(w0_raw, fp8_mode);
            acc0 += reduce<float>(x * w0_f, std::plus<>());

            simd<uint8_t, VL> w1_raw = block_load<uint8_t, VL>(
                gemv_weight + (size_t)n1 * K + offset);
            simd<float, VL> w1_f = fp8_dequant_gemv_gelu_tanh_mul<VL>(w1_raw, fp8_mode);
            acc1 += reduce<float>(x * w1_f, std::plus<>());
        }

        float first = acc0 * scale0;
        float second = acc1 * scale1;
        output[n] = scalar_t(gelu_tanh_scalar_gemv(first) * second);
    }
};

template<typename scalar_t, int VL, int K_SPLIT, bool CACHE_X_NORM>
struct GemvGeluTanhMul_fp8_pert_kernel {
    const scalar_t* x_ptr;       // [1, K]
    const uint8_t* gemv_weight;  // [N, K] FP8
    const float*   gemv_scale;   // [1] or [2] per-tensor scale
    scalar_t*      output;       // [N/2]
    int N;
    int K;
    int scale_count;
    int fp8_mode;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        if constexpr (K_SPLIT > 1) {
            constexpr int slm_floats = K_SPLIT * 2;
            constexpr int cache_bytes = CACHE_X_NORM ? (GEMV_GELU_TANH_MUL_SLM_CACHE_K * sizeof(scalar_t)) : 0;
            slm_init<slm_floats * sizeof(float) + cache_bytes>();
        }

        constexpr int slm_scratch_bytes = K_SPLIT * 2 * sizeof(float);
        constexpr int x_cache_slm_offset = slm_scratch_bytes;

        int n = item.get_group(0);
        int lid = item.get_local_id(0);
        if (n >= N) return;
        int n1 = n + N;

        float scale0 = gemv_scale[0];
        float scale1 = (scale_count > 1) ? gemv_scale[1] : gemv_scale[0];

        int k_per_thread = K / K_SPLIT;
        int k_start = lid * k_per_thread;
        int n_chunks = k_per_thread / VL;

        for (int c = 0; c < n_chunks; c++) {
            int offset = k_start + c * VL;
            simd<scalar_t, VL> x_h = block_load<scalar_t, VL>(x_ptr + offset);
            if constexpr (CACHE_X_NORM) {
                slm_block_store<scalar_t, VL>(x_cache_slm_offset + offset * sizeof(scalar_t), x_h);
            }
        }

        float acc0 = 0.0f;
        float acc1 = 0.0f;

        for (int c = 0; c < n_chunks; c++) {
            int offset = k_start + c * VL;

            simd<float, VL> x_f;
            if constexpr (CACHE_X_NORM) {
                x_f = simd<float, VL>(slm_block_load<scalar_t, VL>(x_cache_slm_offset + offset * sizeof(scalar_t)));
            } else {
                x_f = block_load<scalar_t, VL>(x_ptr + offset);
            }

            simd<uint8_t, VL> w0_raw = block_load<uint8_t, VL>(
                gemv_weight + (size_t)n * K + offset);
            simd<float, VL> w0_f = fp8_dequant_gemv_gelu_tanh_mul<VL>(w0_raw, fp8_mode);
            acc0 += reduce<float>(x_f * w0_f, std::plus<>());

            simd<uint8_t, VL> w1_raw = block_load<uint8_t, VL>(
                gemv_weight + (size_t)n1 * K + offset);
            simd<float, VL> w1_f = fp8_dequant_gemv_gelu_tanh_mul<VL>(w1_raw, fp8_mode);
            acc1 += reduce<float>(x_f * w1_f, std::plus<>());
        }

        float my_sum0 = acc0 * scale0;
        float my_sum1 = acc1 * scale1;

        if constexpr (K_SPLIT == 1) {
            output[n] = scalar_t(gelu_tanh_scalar_gemv(my_sum0) * my_sum1);
        } else {
            constexpr int slm_out0_offset = 0;
            constexpr int slm_out1_offset = K_SPLIT * sizeof(float);
            slm_block_store<float, 1>(slm_out0_offset + lid * sizeof(float), simd<float, 1>(my_sum0));
            slm_block_store<float, 1>(slm_out1_offset + lid * sizeof(float), simd<float, 1>(my_sum1));
            barrier();
            if (lid == 0) {
                simd<float, K_SPLIT> parts0 = slm_block_load<float, K_SPLIT>(slm_out0_offset);
                simd<float, K_SPLIT> parts1 = slm_block_load<float, K_SPLIT>(slm_out1_offset);
                float sum0 = reduce<float>(parts0, std::plus<>());
                float sum1 = reduce<float>(parts1, std::plus<>());
                output[n] = scalar_t(gelu_tanh_scalar_gemv(sum0) * sum1);
            }
        }
    }
};

/* Host dispatcher */
template<typename scalar_t>
inline void gemv_gelu_tanh_mul_fp8_pert_host_impl(
    const scalar_t* x_ptr,
    const uint8_t* gemv_weight,
    const float* gemv_scale,
    scalar_t* output,
    int N, int K,
    int vl, int ks,
    int scale_count,
    int fp8_mode,
    sycl::queue& q)
{
    int global = N * ks;
    int local = ks;

    if (vl == 256 && ks == 1) {
        q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<1>(global, local),
                GemvGeluTanhMul_fp8_pert_kernel_256_1<scalar_t>{
                    x_ptr,
                    gemv_weight,
                    gemv_scale,
                    output,
                    N,
                    K,
                    scale_count,
                    fp8_mode});
        });
        return;
    }

    bool use_slm_cache = (ks > 1) && (K <= GEMV_GELU_TANH_MUL_SLM_CACHE_K);

    #define LAUNCH_GEMV_GELU_TANH_MUL(V, S, CACHE_X_NORM_FLAG) \
        q.submit([&](sycl::handler& cgh) { \
            cgh.parallel_for( \
                sycl::nd_range<1>(global, local), \
                GemvGeluTanhMul_fp8_pert_kernel<scalar_t, V, S, CACHE_X_NORM_FLAG>{ \
                    x_ptr, gemv_weight, gemv_scale, output, \
                    N, K, scale_count, fp8_mode}); \
        });

    if (use_slm_cache) {
        if (vl == 512 && ks == 1) { LAUNCH_GEMV_GELU_TANH_MUL(512, 1, true) }
        else if (vl == 512 && ks == 2) { LAUNCH_GEMV_GELU_TANH_MUL(512, 2, true) }
        else if (vl == 512 && ks == 5) { LAUNCH_GEMV_GELU_TANH_MUL(512, 5, true) }
        else if (vl == 512 && ks == 4) { LAUNCH_GEMV_GELU_TANH_MUL(512, 4, true) }
        else if (vl == 512 && ks == 8) { LAUNCH_GEMV_GELU_TANH_MUL(512, 8, true) }
        else if (vl == 512 && ks == 10) { LAUNCH_GEMV_GELU_TANH_MUL(512, 10, true) }
        if (vl == 256 && ks == 1) { LAUNCH_GEMV_GELU_TANH_MUL(256, 1, true) }
        else if (vl == 256 && ks == 2) { LAUNCH_GEMV_GELU_TANH_MUL(256, 2, true) }
        else if (vl == 256 && ks == 5) { LAUNCH_GEMV_GELU_TANH_MUL(256, 5, true) }
        else if (vl == 256 && ks == 4) { LAUNCH_GEMV_GELU_TANH_MUL(256, 4, true) }
        else if (vl == 256 && ks == 8) { LAUNCH_GEMV_GELU_TANH_MUL(256, 8, true) }
        else if (vl == 128 && ks == 1) { LAUNCH_GEMV_GELU_TANH_MUL(128, 1, true) }
        else if (vl == 128 && ks == 2) { LAUNCH_GEMV_GELU_TANH_MUL(128, 2, true) }
        else if (vl == 128 && ks == 5) { LAUNCH_GEMV_GELU_TANH_MUL(128, 5, true) }
        else if (vl == 128 && ks == 4) { LAUNCH_GEMV_GELU_TANH_MUL(128, 4, true) }
        else if (vl == 128 && ks == 8) { LAUNCH_GEMV_GELU_TANH_MUL(128, 8, true) }
        else if (vl == 128 && ks == 10) { LAUNCH_GEMV_GELU_TANH_MUL(128, 10, true) }
        else { LAUNCH_GEMV_GELU_TANH_MUL(128, 1, true) }
    } else {
        if (vl == 512 && ks == 1) { LAUNCH_GEMV_GELU_TANH_MUL(512, 1, false) }
        else if (vl == 512 && ks == 2) { LAUNCH_GEMV_GELU_TANH_MUL(512, 2, false) }
        else if (vl == 512 && ks == 5) { LAUNCH_GEMV_GELU_TANH_MUL(512, 5, false) }
        else if (vl == 512 && ks == 4) { LAUNCH_GEMV_GELU_TANH_MUL(512, 4, false) }
        else if (vl == 512 && ks == 8) { LAUNCH_GEMV_GELU_TANH_MUL(512, 8, false) }
        else if (vl == 512 && ks == 10) { LAUNCH_GEMV_GELU_TANH_MUL(512, 10, false) }
        if (vl == 256 && ks == 1) { LAUNCH_GEMV_GELU_TANH_MUL(256, 1, false) }
        else if (vl == 256 && ks == 2) { LAUNCH_GEMV_GELU_TANH_MUL(256, 2, false) }
        else if (vl == 256 && ks == 5) { LAUNCH_GEMV_GELU_TANH_MUL(256, 5, false) }
        else if (vl == 256 && ks == 4) { LAUNCH_GEMV_GELU_TANH_MUL(256, 4, false) }
        else if (vl == 256 && ks == 8) { LAUNCH_GEMV_GELU_TANH_MUL(256, 8, false) }
        else if (vl == 128 && ks == 1) { LAUNCH_GEMV_GELU_TANH_MUL(128, 1, false) }
        else if (vl == 128 && ks == 2) { LAUNCH_GEMV_GELU_TANH_MUL(128, 2, false) }
        else if (vl == 128 && ks == 4) { LAUNCH_GEMV_GELU_TANH_MUL(128, 4, false) }
        else if (vl == 128 && ks == 5) { LAUNCH_GEMV_GELU_TANH_MUL(128, 5, false) }
        else if (vl == 128 && ks == 8) { LAUNCH_GEMV_GELU_TANH_MUL(128, 8, false) }
        else if (vl == 128 && ks == 10) { LAUNCH_GEMV_GELU_TANH_MUL(128, 10, false) }
        else { LAUNCH_GEMV_GELU_TANH_MUL(128, 1, false) }
    }

    #undef LAUNCH_GEMV_GELU_TANH_MUL
}

inline void gemv_gelu_tanh_mul_fp8_pert_host(
    const fp16* x_ptr,
    const uint8_t* gemv_weight,
    const float* gemv_scale,
    fp16* output,
    int N, int K,
    int vl, int ks,
    int scale_count,
    int fp8_mode,
    sycl::queue& q)
{
    gemv_gelu_tanh_mul_fp8_pert_host_impl<fp16>(
        x_ptr, gemv_weight, gemv_scale, output, N, K, vl, ks, scale_count, fp8_mode, q);
}

inline void gemv_gelu_tanh_mul_fp8_pert_host(
    const bf16* x_ptr,
    const uint8_t* gemv_weight,
    const float* gemv_scale,
    bf16* output,
    int N, int K,
    int vl, int ks,
    int scale_count,
    int fp8_mode,
    sycl::queue& q)
{
    gemv_gelu_tanh_mul_fp8_pert_host_impl<bf16>(
        x_ptr, gemv_weight, gemv_scale, output, N, K, vl, ks, scale_count, fp8_mode, q);
}

inline void gemv_gelu_tanh_mul_fp8_pert_host(
    const fp16* x_ptr,
    const uint8_t* gemv_weight,
    const float* gemv_scale,
    fp16* output,
    int N,
    int K,
    int scale_count,
    int fp8_mode,
    sycl::queue& q)
{
    int vl;
    int ks;
    auto dev = q.get_device();
    select_vl_ks_gemv_gelu_tanh_mul(N, K, vl, ks, &dev);
    gemv_gelu_tanh_mul_fp8_pert_host(
        x_ptr,
        gemv_weight,
        gemv_scale,
        output,
        N,
        K,
        vl,
        ks,
        scale_count,
        fp8_mode,
        q);
}

inline void gemv_gelu_tanh_mul_fp8_pert_host(
    const bf16* x_ptr,
    const uint8_t* gemv_weight,
    const float* gemv_scale,
    bf16* output,
    int N,
    int K,
    int scale_count,
    int fp8_mode,
    sycl::queue& q)
{
    int vl;
    int ks;
    auto dev = q.get_device();
    select_vl_ks_gemv_gelu_tanh_mul(N, K, vl, ks, &dev);
    gemv_gelu_tanh_mul_fp8_pert_host(
        x_ptr,
        gemv_weight,
        gemv_scale,
        output,
        N,
        K,
        vl,
        ks,
        scale_count,
        fp8_mode,
        q);
}

/* ================================================================
 * Kernel: FP8 GEMV + GELU(tanh) * y
 * output[n] = GELU_tanh((x @ dequant(gemv_weight[n]^T)) * scale) * y[n]
 * ================================================================ */
template<typename scalar_t>
struct GemvGeluTanhMulY_fp8_pert_kernel_256_1 {
    const scalar_t* x_ptr;       // [1, K]
    const uint8_t* gemv_weight;  // [N, K] FP8
    const float*   gemv_scale;   // [1]
    const scalar_t* y_ptr;       // [N]
    scalar_t*      output;       // [N]
    int N;
    int K;
    int fp8_mode;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        constexpr int VL = 256;
        int n = item.get_group(0);
        if (n >= N) return;

        int n_chunks = K / VL;
        float scale = gemv_scale[0];

        float acc = 0.0f;
        for (int c = 0; c < n_chunks; c++) {
            int offset = c * VL;
            simd<float, VL> x = block_load<scalar_t, VL>(x_ptr + offset);
            simd<uint8_t, VL> w_raw = block_load<uint8_t, VL>(
                gemv_weight + (size_t)n * K + offset);
            simd<float, VL> w_f = fp8_dequant_gemv_gelu_tanh_mul<VL>(w_raw, fp8_mode);
            acc += reduce<float>(x * w_f, std::plus<>());
        }

        float x_proj = acc * scale;
        float y = (float)y_ptr[n];
        output[n] = scalar_t(gelu_tanh_scalar_gemv(x_proj) * y);
    }
};

template<typename scalar_t, int VL, int K_SPLIT, bool CACHE_X_NORM>
struct GemvGeluTanhMulY_fp8_pert_kernel {
    const scalar_t* x_ptr;       // [1, K]
    const uint8_t* gemv_weight;  // [N, K] FP8
    const float*   gemv_scale;   // [1]
    const scalar_t* y_ptr;       // [N]
    scalar_t*      output;       // [N]
    int N;
    int K;
    int fp8_mode;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        if constexpr (K_SPLIT > 1) {
            constexpr int slm_floats = K_SPLIT;
            constexpr int cache_bytes = CACHE_X_NORM ? (GEMV_GELU_TANH_MUL_SLM_CACHE_K * sizeof(scalar_t)) : 0;
            slm_init<slm_floats * sizeof(float) + cache_bytes>();
        }

        constexpr int slm_scratch_bytes = K_SPLIT * sizeof(float);
        constexpr int x_cache_slm_offset = slm_scratch_bytes;

        int n = item.get_group(0);
        int lid = item.get_local_id(0);
        if (n >= N) return;

        float scale = gemv_scale[0];
        int k_per_thread = K / K_SPLIT;
        int k_start = lid * k_per_thread;
        int n_chunks = k_per_thread / VL;

        for (int c = 0; c < n_chunks; c++) {
            int offset = k_start + c * VL;
            simd<scalar_t, VL> x_h = block_load<scalar_t, VL>(x_ptr + offset);
            if constexpr (CACHE_X_NORM) {
                slm_block_store<scalar_t, VL>(x_cache_slm_offset + offset * sizeof(scalar_t), x_h);
            }
        }

        float acc = 0.0f;
        for (int c = 0; c < n_chunks; c++) {
            int offset = k_start + c * VL;

            simd<float, VL> x_f;
            if constexpr (CACHE_X_NORM) {
                x_f = simd<float, VL>(slm_block_load<scalar_t, VL>(x_cache_slm_offset + offset * sizeof(scalar_t)));
            } else {
                x_f = block_load<scalar_t, VL>(x_ptr + offset);
            }

            simd<uint8_t, VL> w_raw = block_load<uint8_t, VL>(
                gemv_weight + (size_t)n * K + offset);
            simd<float, VL> w_f = fp8_dequant_gemv_gelu_tanh_mul<VL>(w_raw, fp8_mode);
            acc += reduce<float>(x_f * w_f, std::plus<>());
        }

        float my_sum = acc * scale;
        if constexpr (K_SPLIT == 1) {
            float y = (float)y_ptr[n];
            output[n] = scalar_t(gelu_tanh_scalar_gemv(my_sum) * y);
        } else {
            constexpr int slm_out_offset = 0;
            slm_block_store<float, 1>(slm_out_offset + lid * sizeof(float), simd<float, 1>(my_sum));
            barrier();
            if (lid == 0) {
                simd<float, K_SPLIT> parts = slm_block_load<float, K_SPLIT>(slm_out_offset);
                float sum = reduce<float>(parts, std::plus<>());
                float y = (float)y_ptr[n];
                output[n] = scalar_t(gelu_tanh_scalar_gemv(sum) * y);
            }
        }
    }
};

/* Host dispatcher: FP8 GEMV + GELU(tanh) * y */
template<typename scalar_t>
inline void gemv_gelu_tanh_mul_y_fp8_pert_host_impl(
    const scalar_t* x_ptr,
    const uint8_t* gemv_weight,
    const float* gemv_scale,
    const scalar_t* y_ptr,
    scalar_t* output,
    int N,
    int K,
    int vl,
    int ks,
    int fp8_mode,
    sycl::queue& q)
{
    int global = N * ks;
    int local = ks;

    if (vl == 256 && ks == 1) {
        q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<1>(global, local),
                GemvGeluTanhMulY_fp8_pert_kernel_256_1<scalar_t>{
                    x_ptr,
                    gemv_weight,
                    gemv_scale,
                    y_ptr,
                    output,
                    N,
                    K,
                    fp8_mode});
        });
        return;
    }

    bool use_slm_cache = (ks > 1) && (K <= GEMV_GELU_TANH_MUL_SLM_CACHE_K);

    #define LAUNCH_GEMV_GELU_TANH_MUL_Y(V, S, CACHE_X_NORM_FLAG) \
        q.submit([&](sycl::handler& cgh) { \
            cgh.parallel_for( \
                sycl::nd_range<1>(global, local), \
                GemvGeluTanhMulY_fp8_pert_kernel<scalar_t, V, S, CACHE_X_NORM_FLAG>{ \
                    x_ptr, gemv_weight, gemv_scale, y_ptr, output, \
                    N, K, fp8_mode}); \
        });

    if (use_slm_cache) {
        if (vl == 512 && ks == 1) { LAUNCH_GEMV_GELU_TANH_MUL_Y(512, 1, true) }
        else if (vl == 512 && ks == 2) { LAUNCH_GEMV_GELU_TANH_MUL_Y(512, 2, true) }
        else if (vl == 512 && ks == 5) { LAUNCH_GEMV_GELU_TANH_MUL_Y(512, 5, true) }
        else if (vl == 512 && ks == 4) { LAUNCH_GEMV_GELU_TANH_MUL_Y(512, 4, true) }
        else if (vl == 512 && ks == 8) { LAUNCH_GEMV_GELU_TANH_MUL_Y(512, 8, true) }
        else if (vl == 512 && ks == 10) { LAUNCH_GEMV_GELU_TANH_MUL_Y(512, 10, true) }
        if (vl == 256 && ks == 1) { LAUNCH_GEMV_GELU_TANH_MUL_Y(256, 1, true) }
        else if (vl == 256 && ks == 2) { LAUNCH_GEMV_GELU_TANH_MUL_Y(256, 2, true) }
        else if (vl == 256 && ks == 5) { LAUNCH_GEMV_GELU_TANH_MUL_Y(256, 5, true) }
        else if (vl == 256 && ks == 4) { LAUNCH_GEMV_GELU_TANH_MUL_Y(256, 4, true) }
        else if (vl == 256 && ks == 8) { LAUNCH_GEMV_GELU_TANH_MUL_Y(256, 8, true) }
        else if (vl == 128 && ks == 1) { LAUNCH_GEMV_GELU_TANH_MUL_Y(128, 1, true) }
        else if (vl == 128 && ks == 2) { LAUNCH_GEMV_GELU_TANH_MUL_Y(128, 2, true) }
        else if (vl == 128 && ks == 5) { LAUNCH_GEMV_GELU_TANH_MUL_Y(128, 5, true) }
        else if (vl == 128 && ks == 4) { LAUNCH_GEMV_GELU_TANH_MUL_Y(128, 4, true) }
        else if (vl == 128 && ks == 8) { LAUNCH_GEMV_GELU_TANH_MUL_Y(128, 8, true) }
        else if (vl == 128 && ks == 10) { LAUNCH_GEMV_GELU_TANH_MUL_Y(128, 10, true) }
        else { LAUNCH_GEMV_GELU_TANH_MUL_Y(128, 1, true) }
    } else {
        if (vl == 512 && ks == 1) { LAUNCH_GEMV_GELU_TANH_MUL_Y(512, 1, false) }
        else if (vl == 512 && ks == 2) { LAUNCH_GEMV_GELU_TANH_MUL_Y(512, 2, false) }
        else if (vl == 512 && ks == 5) { LAUNCH_GEMV_GELU_TANH_MUL_Y(512, 5, false) }
        else if (vl == 512 && ks == 4) { LAUNCH_GEMV_GELU_TANH_MUL_Y(512, 4, false) }
        else if (vl == 512 && ks == 8) { LAUNCH_GEMV_GELU_TANH_MUL_Y(512, 8, false) }
        else if (vl == 512 && ks == 10) { LAUNCH_GEMV_GELU_TANH_MUL_Y(512, 10, false) }
        if (vl == 256 && ks == 1) { LAUNCH_GEMV_GELU_TANH_MUL_Y(256, 1, false) }
        else if (vl == 256 && ks == 2) { LAUNCH_GEMV_GELU_TANH_MUL_Y(256, 2, false) }
        else if (vl == 256 && ks == 5) { LAUNCH_GEMV_GELU_TANH_MUL_Y(256, 5, false) }
        else if (vl == 256 && ks == 4) { LAUNCH_GEMV_GELU_TANH_MUL_Y(256, 4, false) }
        else if (vl == 256 && ks == 8) { LAUNCH_GEMV_GELU_TANH_MUL_Y(256, 8, false) }
        else if (vl == 128 && ks == 1) { LAUNCH_GEMV_GELU_TANH_MUL_Y(128, 1, false) }
        else if (vl == 128 && ks == 2) { LAUNCH_GEMV_GELU_TANH_MUL_Y(128, 2, false) }
        else if (vl == 128 && ks == 5) { LAUNCH_GEMV_GELU_TANH_MUL_Y(128, 5, false) }
        else if (vl == 128 && ks == 4) { LAUNCH_GEMV_GELU_TANH_MUL_Y(128, 4, false) }
        else if (vl == 128 && ks == 8) { LAUNCH_GEMV_GELU_TANH_MUL_Y(128, 8, false) }
        else if (vl == 128 && ks == 10) { LAUNCH_GEMV_GELU_TANH_MUL_Y(128, 10, false) }
        else { LAUNCH_GEMV_GELU_TANH_MUL_Y(128, 1, false) }
    }

    #undef LAUNCH_GEMV_GELU_TANH_MUL_Y
}

inline void gemv_gelu_tanh_mul_y_fp8_pert_host(
    const fp16* x_ptr,
    const uint8_t* gemv_weight,
    const float* gemv_scale,
    const fp16* y_ptr,
    fp16* output,
    int N,
    int K,
    int vl,
    int ks,
    int fp8_mode,
    sycl::queue& q)
{
    gemv_gelu_tanh_mul_y_fp8_pert_host_impl<fp16>(
        x_ptr, gemv_weight, gemv_scale, y_ptr, output, N, K, vl, ks, fp8_mode, q);
}

inline void gemv_gelu_tanh_mul_y_fp8_pert_host(
    const bf16* x_ptr,
    const uint8_t* gemv_weight,
    const float* gemv_scale,
    const bf16* y_ptr,
    bf16* output,
    int N,
    int K,
    int vl,
    int ks,
    int fp8_mode,
    sycl::queue& q)
{
    gemv_gelu_tanh_mul_y_fp8_pert_host_impl<bf16>(
        x_ptr, gemv_weight, gemv_scale, y_ptr, output, N, K, vl, ks, fp8_mode, q);
}

inline void gemv_gelu_tanh_mul_y_fp8_pert_host(
    const fp16* x_ptr,
    const uint8_t* gemv_weight,
    const float* gemv_scale,
    const fp16* y_ptr,
    fp16* output,
    int N,
    int K,
    int fp8_mode,
    sycl::queue& q)
{
    int vl;
    int ks;
    auto dev = q.get_device();
    select_vl_ks_gemv_gelu_tanh_mul(N, K, vl, ks, &dev);
    gemv_gelu_tanh_mul_y_fp8_pert_host(
        x_ptr,
        gemv_weight,
        gemv_scale,
        y_ptr,
        output,
        N,
        K,
        vl,
        ks,
        fp8_mode,
        q);
}

inline void gemv_gelu_tanh_mul_y_fp8_pert_host(
    const bf16* x_ptr,
    const uint8_t* gemv_weight,
    const float* gemv_scale,
    const bf16* y_ptr,
    bf16* output,
    int N,
    int K,
    int fp8_mode,
    sycl::queue& q)
{
    int vl;
    int ks;
    auto dev = q.get_device();
    select_vl_ks_gemv_gelu_tanh_mul(N, K, vl, ks, &dev);
    gemv_gelu_tanh_mul_y_fp8_pert_host(
        x_ptr,
        gemv_weight,
        gemv_scale,
        y_ptr,
        output,
        N,
        K,
        vl,
        ks,
        fp8_mode,
        q);
}
