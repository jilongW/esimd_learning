/* norm_gemv_fused.h — Fused RMSNorm + FP8 GEMV for GDN out_proj.
 *
 * Combines two operations into a single kernel:
 *   1. RMSNorm: normed = x / rms(x) * weight
 *   2. GEMV: output = normed @ dequant(gemv_weight^T) * scale
 *
 * Designed for decode shapes such as:
 *   x:           [1, K] fp16      (e.g. [1, 2560])
 *   norm_weight: [K] fp16         (e.g. [2560])
 *   gemv_weight: [N, K] FP8       (e.g. [20480, 2560])
 *   gemv_scale:  [1] float32
 *   output:      [N] fp16
 *
 * Dispatch is aligned with the tuned GEMV path: VL and K_SPLIT are selected
 * from N/K, and the kernel uses nd_range(global=N*K_SPLIT, local=K_SPLIT).
 */

#pragma once
#include "utils.h"

inline void select_vl_ks_norm_gemv(uint32_t N, uint32_t K, int& vl, int& ks) {
    if (K == 256) {
        vl = 256;
        ks = 1;
    } else if (K == 2048) {
        vl = 256;
        ks = 8;
    } else if (K == 4096) {
        vl = 256;
        ks = 4;
    } else if (K == 10240) {
        vl = 256;
        ks = 8;
    } else if (K == 2560) {
        if (N <= 256) {
            vl = 128;
            ks = 10;
        } else if (N >= 10240) {
            vl = 256;
            ks = 1;
        } else {
            vl = 128;
            ks = 4;
        }
    } else if (K >= 4096) {
        vl = 256;
        ks = 4;
    } else if (K >= 2048) {
        vl = 256;
        ks = 8;
    } else {
        vl = 128;
        ks = 1;
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

template<int VL>
SYCL_ESIMD_FUNCTION inline simd<float, VL> fp8_dequant_norm(
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

/* ================================================================
 * Kernel: Fused RMSNorm + FP8 GEMV (per-tensor scale)
 * Dispatch follows the same VL/K_SPLIT strategy as resadd_norm_gemv.
 * ================================================================ */
constexpr int NORM_GEMV_SLM_CACHE_K = 2560;

struct NormGEMV_fp8_pert_kernel_256_1 {
    const fp16*    x_ptr;        // [1, K]
    const fp16*    norm_w_ptr;   // [K]
    const uint8_t* gemv_weight;  // [N, K] FP8
    const float*   gemv_scale;   // [1]
    fp16*          output;       // [N]
    int N;
    int K;
    float eps;
    int fp8_mode;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        constexpr int VL = 256;
        int n = item.get_group(0);
        if (n >= N) return;

        int n_chunks = K / VL;
        float sum_sq = 0.0f;
        for (int c = 0; c < n_chunks; c++) {
            int offset = c * VL;
            simd<float, VL> x = block_load<fp16, VL>(x_ptr + offset);
            sum_sq += reduce<float>(x * x, std::plus<>());
        }

        float inv_rms = sycl::ext::intel::esimd::rsqrt(
            simd<float, 8>(sum_sq / (float)K + eps))[0];

        float acc = 0.0f;
        for (int c = 0; c < n_chunks; c++) {
            int offset = c * VL;

            simd<float, VL> x = block_load<fp16, VL>(x_ptr + offset);
            simd<float, VL> nw = block_load<fp16, VL>(norm_w_ptr + offset);
            simd<float, VL> normed = x * inv_rms * nw;

            simd<uint8_t, VL> w_raw = block_load<uint8_t, VL>(
                gemv_weight + (size_t)n * K + offset);
            simd<float, VL> w_f = fp8_dequant_norm<VL>(w_raw, fp8_mode);
            acc += reduce<float>(normed * w_f, std::plus<>());
        }

        output[n] = fp16(acc * gemv_scale[0]);
    }
};

template<int VL, int K_SPLIT, bool CACHE_X_NORM>
struct NormGEMV_fp8_pert_kernel {
    const fp16*    x_ptr;        // [1, K]
    const fp16*    norm_w_ptr;   // [K]
    const uint8_t* gemv_weight;  // [N, K] FP8
    const float*   gemv_scale;   // [1] per-tensor scale
    fp16*          output;       // [N]
    int N;
    int K;
    float eps;
    int fp8_mode;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        if constexpr (K_SPLIT > 1) {
            constexpr int slm_floats = K_SPLIT + 1;
            constexpr int cache_bytes = CACHE_X_NORM ? (2 * NORM_GEMV_SLM_CACHE_K * sizeof(fp16)) : 0;
            slm_init<slm_floats * sizeof(float) + cache_bytes>();
        }

        constexpr int slm_scratch_bytes = (K_SPLIT + 1) * sizeof(float);
        constexpr int x_cache_slm_offset = slm_scratch_bytes;
        constexpr int norm_w_cache_slm_offset = x_cache_slm_offset + NORM_GEMV_SLM_CACHE_K * sizeof(fp16);

        int n = item.get_group(0);
        int lid = item.get_local_id(0);
        if (n >= N) return;

        int k_per_thread = K / K_SPLIT;
        int k_start = lid * k_per_thread;
        int n_chunks = k_per_thread / VL;

        float sum_sq = 0.0f;

        for (int c = 0; c < n_chunks; c++) {
            int offset = k_start + c * VL;
            simd<fp16, VL> x_h = block_load<fp16, VL>(x_ptr + offset);
            if constexpr (CACHE_X_NORM) {
                simd<fp16, VL> norm_w_h = block_load<fp16, VL>(norm_w_ptr + offset);
                slm_block_store<fp16, VL>(x_cache_slm_offset + offset * sizeof(fp16), x_h);
                slm_block_store<fp16, VL>(norm_w_cache_slm_offset + offset * sizeof(fp16), norm_w_h);
            }
            simd<float, VL> x_f = x_h;
            simd<float, VL> x_sq = x_f * x_f;
            sum_sq += reduce<float>(x_sq, std::plus<>());
        }

        float inv_rms = 0.0f;

        if constexpr (K_SPLIT == 1) {
            inv_rms = sycl::ext::intel::esimd::rsqrt(
                simd<float, 8>(sum_sq / (float)K + eps))[0];
        } else {
            constexpr int inv_rms_slm_offset = K_SPLIT * sizeof(float);
            slm_block_store<float, 1>(lid * sizeof(float), simd<float, 1>(sum_sq));
            barrier();
            if (lid == 0) {
                simd<float, K_SPLIT> parts = slm_block_load<float, K_SPLIT>(0);
                float total_sum_sq = reduce<float>(parts, std::plus<>());
                inv_rms = sycl::ext::intel::esimd::rsqrt(
                    simd<float, 8>(total_sum_sq / (float)K + eps))[0];
                slm_block_store<float, 1>(inv_rms_slm_offset, simd<float, 1>(inv_rms));
            }
            barrier();
            inv_rms = slm_block_load<float, 1>(inv_rms_slm_offset)[0];
        }

        float acc = 0.0f;

        for (int c = 0; c < n_chunks; c++) {
            int offset = k_start + c * VL;

            simd<float, VL> x_f;
            simd<float, VL> norm_w;
            if constexpr (CACHE_X_NORM) {
                x_f = simd<float, VL>(slm_block_load<fp16, VL>(x_cache_slm_offset + offset * sizeof(fp16)));
                norm_w = simd<float, VL>(slm_block_load<fp16, VL>(norm_w_cache_slm_offset + offset * sizeof(fp16)));
            } else {
                x_f = block_load<fp16, VL>(x_ptr + offset);
                norm_w = block_load<fp16, VL>(norm_w_ptr + offset);
            }
            simd<float, VL> normed = x_f * inv_rms * norm_w;

            simd<uint8_t, VL> w_raw = block_load<uint8_t, VL>(
                gemv_weight + (size_t)n * K + offset);
            simd<float, VL> w_f = fp8_dequant_norm<VL>(w_raw, fp8_mode);
            acc += reduce<float>(normed * w_f, std::plus<>());
        }

        float my_sum = acc * gemv_scale[0];

        if constexpr (K_SPLIT == 1) {
            output[n] = fp16(my_sum);
        } else {
            slm_block_store<float, 1>(lid * sizeof(float), simd<float, 1>(my_sum));
            barrier();
            if (lid == 0) {
                simd<float, K_SPLIT> parts = slm_block_load<float, K_SPLIT>(0);
                output[n] = fp16(reduce<float>(parts, std::plus<>()));
            }
        }
    }
};

/* Host dispatcher */
inline void norm_gemv_fp8_pert_host(
    const fp16* x_ptr,
    const fp16* norm_w_ptr,
    const uint8_t* gemv_weight,
    const float* gemv_scale,
    fp16* output,
    int N, int K,
    int vl, int ks,
    float eps,
    int fp8_mode,
    sycl::queue& q)
{
    int global = N * ks;
    int local = ks;

    if (vl == 256 && ks == 1) {
        q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<1>(global, local),
                NormGEMV_fp8_pert_kernel_256_1{
                    x_ptr,
                    norm_w_ptr,
                    gemv_weight,
                    gemv_scale,
                    output,
                    N,
                    K,
                    eps,
                    fp8_mode});
        });
        return;
    }

    bool use_slm_cache = (ks > 1) && (K <= NORM_GEMV_SLM_CACHE_K);

    #define LAUNCH_NORM_GEMV(V, S, CACHE_X_NORM_FLAG) \
        q.submit([&](sycl::handler& cgh) { \
            cgh.parallel_for( \
                sycl::nd_range<1>(global, local), \
                NormGEMV_fp8_pert_kernel<V, S, CACHE_X_NORM_FLAG>{ \
                    x_ptr, norm_w_ptr, gemv_weight, gemv_scale, output, \
                    N, K, eps, fp8_mode}); \
        });

    if (use_slm_cache) {
        if (vl == 256 && ks == 1) { LAUNCH_NORM_GEMV(256, 1, true) }
        else if (vl == 256 && ks == 2) { LAUNCH_NORM_GEMV(256, 2, true) }
        else if (vl == 256 && ks == 4) { LAUNCH_NORM_GEMV(256, 4, true) }
        else if (vl == 256 && ks == 8) { LAUNCH_NORM_GEMV(256, 8, true) }
        else if (vl == 128 && ks == 1) { LAUNCH_NORM_GEMV(128, 1, true) }
        else if (vl == 128 && ks == 2) { LAUNCH_NORM_GEMV(128, 2, true) }
        else if (vl == 128 && ks == 4) { LAUNCH_NORM_GEMV(128, 4, true) }
        else if (vl == 128 && ks == 8) { LAUNCH_NORM_GEMV(128, 8, true) }
        else if (vl == 128 && ks == 10) { LAUNCH_NORM_GEMV(128, 10, true) }
        else { LAUNCH_NORM_GEMV(128, 1, true) }
    } else {
        if (vl == 256 && ks == 1) { LAUNCH_NORM_GEMV(256, 1, false) }
        else if (vl == 256 && ks == 2) { LAUNCH_NORM_GEMV(256, 2, false) }
        else if (vl == 256 && ks == 4) { LAUNCH_NORM_GEMV(256, 4, false) }
        else if (vl == 256 && ks == 8) { LAUNCH_NORM_GEMV(256, 8, false) }
        else if (vl == 128 && ks == 1) { LAUNCH_NORM_GEMV(128, 1, false) }
        else if (vl == 128 && ks == 2) { LAUNCH_NORM_GEMV(128, 2, false) }
        else if (vl == 128 && ks == 4) { LAUNCH_NORM_GEMV(128, 4, false) }
        else if (vl == 128 && ks == 8) { LAUNCH_NORM_GEMV(128, 8, false) }
        else if (vl == 128 && ks == 10) { LAUNCH_NORM_GEMV(128, 10, false) }
        else { LAUNCH_NORM_GEMV(128, 1, false) }
    }

    #undef LAUNCH_NORM_GEMV
}

inline void norm_gemv_fp8_pert_host(
    const fp16* x_ptr,
    const fp16* norm_w_ptr,
    const uint8_t* gemv_weight,
    const float* gemv_scale,
    fp16* output,
    int N,
    int K,
    float eps,
    int fp8_mode,
    sycl::queue& q)
{
    int vl;
    int ks;
    select_vl_ks_norm_gemv(N, K, vl, ks);
    norm_gemv_fp8_pert_host(
        x_ptr,
        norm_w_ptr,
        gemv_weight,
        gemv_scale,
        output,
        N,
        K,
        vl,
        ks,
        eps,
        fp8_mode,
        q);
}
