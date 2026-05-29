/* norm_gemv2_fused.h — Fused RMSNorm + 2-matrix FP8 GEMV.
 *
 * Two-pass over the normalized input: accumulate sum_sq for RMS, then re-read
 * hidden_states, normalize, and run two independent FP8 GEMVs.
 *
 * Key insight: we need sum_sq (from pass 1) before normalizing (pass 2), but
 * storing all chunks needs too many registers for large K.
 * Solution: TWO loops over global memory. Pass 1 reads hidden_states for
 * sum_sq only. Pass 2 re-reads hidden_states (from L3), normalizes, does GEMV.
 *
 * Grid: (N0 + N1) WGs, 1 thread each.
 */

#pragma once
#include "utils.h"

inline void normalize_norm_gemv2_vl_ks(uint32_t K, int& vl, int& ks) {
    auto step_down_ks = [](int value) {
        if (value == 10) return 8;
        if (value == 8) return 4;
        if (value == 4) return 2;
        if (value == 2) return 1;
        return value;
    };

    int kpt = static_cast<int>(K) / ks;
    while (vl > kpt || kpt % vl != 0) {
        if (vl > 128) {
            vl /= 2;
        } else {
            int next_ks = step_down_ks(ks);
            if (next_ks == ks) {
                break;
            }
            ks = next_ks;
            kpt = static_cast<int>(K) / ks;
        }
    }
}

inline void select_vl_ks_norm_gemv2(uint32_t total_N, uint32_t K, int& vl, int& ks) {
    vl = 512;
    ks = 1;

    if (K < 256) {
        vl = 128;
        ks = 1;
    } else if (K == 256) {
        vl = 256;
        ks = 1;
    } else if (K >= 10240) {
        vl = 512;
        ks = 2;
    } else if (K >= 4096) {
        vl = 512;
        ks = 2;
    } else if (K >= 2560 && total_N >= 10240) {
        vl = 512;
        ks = 1;
    } else if (K >= 2560) {
        vl = 128;
        ks = 10;
    } else if (K >= 2048) {
        vl = 256;
        ks = 8;
    }

    normalize_norm_gemv2_vl_ks(K, vl, ks);
}

template<int VL>
SYCL_ESIMD_FUNCTION inline simd<float, VL> fp8_dequant_rng2(
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

template<int VL, int K_SPLIT>
struct NormGEMV2_fp8_pert_kernel {
    const fp16*    hidden_ptr;   // [1, K] — read-only
    const fp16*    norm_w_ptr;   // [K]
    const uint8_t* w0_ptr;       // [N0, K] FP8
    const float*   s0_ptr;       // [1]
    fp16*          o0_ptr;       // [1, N0]
    const uint8_t* w1_ptr;       // [N1, K] FP8
    const float*   s1_ptr;       // [1]
    fp16*          o1_ptr;       // [1, N1]
    int N0, N1, K;
    float eps;
    int fp8_mode;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        if constexpr (K_SPLIT > 1) {
            slm_init<K_SPLIT * sizeof(float)>();
        }

        int gid = item.get_group(0);
        int lid = item.get_local_id(0);
        int total_N = N0 + N1;
        if (gid >= total_N) return;

        int mat_idx = (gid < N0) ? 0 : 1;
        int local_n = (mat_idx == 0) ? gid : gid - N0;

        int k_per_thread = K / K_SPLIT;
        int k_start = lid * k_per_thread;
        int n_chunks = k_per_thread / VL;

        // Pass 1: compute sum_sq for RMS
        float sum_sq = 0.0f;
        for (int c = 0; c < n_chunks; c++) {
            int offset = k_start + c * VL;
            simd<float, VL> h = block_load<fp16, VL>(hidden_ptr + offset);

            simd<float, VL> sq = h * h;
            sum_sq += reduce<float>(sq, std::plus<>());
        }

        float inv_rms = 0.0f;

        if constexpr (K_SPLIT == 1) {
            inv_rms = sycl::ext::intel::esimd::rsqrt(
                simd<float, 8>(sum_sq / (float)K + eps))[0];
        } else {
            slm_block_store<float, 1>(lid * sizeof(float), simd<float, 1>(sum_sq));
            barrier();
            if (lid == 0) {
                simd<float, K_SPLIT> parts = slm_block_load<float, K_SPLIT>(0);
                float total_sum_sq = reduce<float>(parts, std::plus<>());
                inv_rms = sycl::ext::intel::esimd::rsqrt(
                    simd<float, 8>(total_sum_sq / (float)K + eps))[0];
                slm_block_store<float, 1>(0, simd<float, 1>(inv_rms));
            }
            barrier();
            inv_rms = slm_block_load<float, 1>(0)[0];
            barrier();
        }

        // Pass 2: re-load hidden_states (L3 cache hit), normalize, GEMV
        const uint8_t* w_ptr = (mat_idx == 0) ? w0_ptr : w1_ptr;
        const float* s_ptr = (mat_idx == 0) ? s0_ptr : s1_ptr;
        fp16* o_ptr = (mat_idx == 0) ? o0_ptr : o1_ptr;

        simd<float, VL> acc = 0.0f;
        for (int c = 0; c < n_chunks; c++) {
            int offset = k_start + c * VL;

            simd<float, VL> h = block_load<fp16, VL>(hidden_ptr + offset);
            simd<float, VL> nw = block_load<fp16, VL>(norm_w_ptr + offset);
            simd<float, VL> normed = h * inv_rms * nw;

            simd<uint8_t, VL> w_raw = block_load<uint8_t, VL>(
                w_ptr + (size_t)local_n * K + offset);
            simd<float, VL> w_f = fp8_dequant_rng2<VL>(w_raw, fp8_mode);
            acc += normed * w_f;
        }

        float my_sum = reduce<float>(acc, std::plus<>()) * *s_ptr;

        if constexpr (K_SPLIT == 1) {
            o_ptr[local_n] = fp16(my_sum);
        } else {
            slm_block_store<float, 1>(lid * sizeof(float), simd<float, 1>(my_sum));
            barrier();
            if (lid == 0) {
                simd<float, K_SPLIT> parts = slm_block_load<float, K_SPLIT>(0);
                o_ptr[local_n] = fp16(reduce<float>(parts, std::plus<>()));
            }
        }
    }
};

inline void norm_gemv2_fp8_pert_host(
    const fp16* hidden_ptr, const fp16* norm_w_ptr,
    const uint8_t* w0, const float* s0, fp16* o0,
    const uint8_t* w1, const float* s1, fp16* o1,
    int N0, int N1, int K, int vl, int ks, float eps, int fp8_mode,
    sycl::queue& q)
{
    int total_N = N0 + N1;
    int global = total_N * ks;
    int local = ks;

    #define LAUNCH_FUSED(V, S) \
        q.submit([&](sycl::handler& cgh) { \
            cgh.parallel_for( \
                sycl::nd_range<1>(global, local), \
                NormGEMV2_fp8_pert_kernel<V, S>{ \
                    hidden_ptr, norm_w_ptr, \
                    w0, s0, o0, w1, s1, o1, \
                    N0, N1, K, eps, fp8_mode}); \
        });

    if (vl == 512 && ks == 1) { LAUNCH_FUSED(512, 1) }
    else if (vl == 512 && ks == 2) { LAUNCH_FUSED(512, 2) }
    else if (vl == 256 && ks == 1) { LAUNCH_FUSED(256, 1) }
    else if (vl == 256 && ks == 2) { LAUNCH_FUSED(256, 2) }
    else if (vl == 256 && ks == 4) { LAUNCH_FUSED(256, 4) }
    else if (vl == 256 && ks == 8) { LAUNCH_FUSED(256, 8) }
    else if (vl == 128 && ks == 1) { LAUNCH_FUSED(128, 1) }
    else if (vl == 128 && ks == 2) { LAUNCH_FUSED(128, 2) }
    else if (vl == 128 && ks == 4) { LAUNCH_FUSED(128, 4) }
    else if (vl == 128 && ks == 8) { LAUNCH_FUSED(128, 8) }
    else if (vl == 128 && ks == 10) { LAUNCH_FUSED(128, 10) }
    else { LAUNCH_FUSED(128, 1) }

    #undef LAUNCH_FUSED
}

inline void norm_gemv2_fp8_pert_host(
    const fp16* hidden_ptr, const fp16* norm_w_ptr,
    const uint8_t* w0, const float* s0, fp16* o0,
    const uint8_t* w1, const float* s1, fp16* o1,
    int N0, int N1, int K, float eps, int fp8_mode,
    sycl::queue& q)
{
    int total_N = N0 + N1;
    int vl, ks;
    select_vl_ks_norm_gemv2(total_N, K, vl, ks);
    norm_gemv2_fp8_pert_host(
        hidden_ptr,
        norm_w_ptr,
        w0,
        s0,
        o0,
        w1,
        s1,
        o1,
        N0,
        N1,
        K,
        vl,
        ks,
        eps,
        fp8_mode,
        q);
}