/* norm_gemv2_geglu_fused.h — Fused RMSNorm + 2-matrix FP8 GEMV + GeGLU.
 *
 * Two-pass over the normalized input: accumulate sum_sq for RMS, then re-read
 * hidden_states, normalize, and run two independent FP8 GEMVs.
 *
 * Final output applies GeGLU in-kernel:
 *   out = GELU_tanh(gemv0(normed_hidden)) * gemv1(normed_hidden)
 *
 * This path returns a single output tensor [1, N0] and requires N0 == N1.
 *
 * Key insight: we need sum_sq (from pass 1) before normalizing (pass 2), but
 * storing all chunks needs too many registers for large K.
 * Solution: TWO loops over global memory. Pass 1 reads hidden_states for
 * sum_sq only. Pass 2 re-reads hidden_states (from L3), normalizes, does GEMV.
 *
 * Grid: N0 WGs, each WG produces one fused GeGLU output element.
 */

#pragma once
#include "utils.h"

inline void normalize_norm_gemv2_geglu_vl_ks(uint32_t K, int& vl, int& ks) {
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

inline void select_vl_ks_norm_gemv2_geglu(uint32_t total_N, uint32_t K, int& vl, int& ks) {
    vl = 512;
    ks = 1;

    if (K < 256) {
        vl = 128;
        ks = 1;
    } else if (K == 256) {
        vl = 256;
        ks = 1;
    } else if (K == 2560 && total_N < 8192) {
        // Measured crossover: small-N decode favors the narrow 256:1 path.
        vl = 256;
        ks = 1;
    } else if (K >= 10240) {
        vl = 512;
        ks = 2;
    } else if (K >= 4096) {
        vl = 512;
        ks = 2;
    } else if (K >= 2560) {
        vl = 128;
        ks = 10;
    } else if (K >= 2048) {
        vl = 256;
        ks = 8;
    }

    normalize_norm_gemv2_geglu_vl_ks(K, vl, ks);
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

template<int VL, int FP8_MODE>
SYCL_ESIMD_FUNCTION inline simd<float, VL> fp8_dequant_rng2_mode(
    simd<uint8_t, VL> raw) {
    static_assert(FP8_MODE == 0 || FP8_MODE == 1, "FP8_MODE must be 0 or 1");
    simd<uint16_t, VL> u16 = convert<uint16_t>(raw);
    simd<uint16_t, VL> fp8_sign = (u16 >> 7) & 1;
    simd<uint16_t, VL> fp16_bits;
    if constexpr (FP8_MODE == 0) {
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

SYCL_ESIMD_FUNCTION inline float gelu_tanh_scalar(float x) {
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

template<int FP8_MODE>
struct NormGEMV2_fp8_pert_kernel_256_1 {
    const fp16*    hidden_ptr;   // [1, K] — read-only
    const fp16*    norm_w_ptr;   // [K]
    const uint8_t* w0_ptr;       // [N0, K] FP8
    const float*   s0_ptr;       // [1]
    fp16*          out_ptr;      // [1, N0]
    const uint8_t* w1_ptr;       // [N1, K] FP8
    const float*   s1_ptr;       // [1]
    int N0, N1, K;
    float eps;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        constexpr int VL = 256;

        int gid = item.get_group(0);
        if (gid >= N0) return;

        int n_chunks = K / VL;

        float sum_sq = 0.0f;
        for (int c = 0; c < n_chunks; c++) {
            int offset = c * VL;
            simd<float, VL> h = block_load<fp16, VL>(hidden_ptr + offset);
            sum_sq += reduce<float>(h * h, std::plus<>());
        }

        float inv_rms = sycl::ext::intel::esimd::rsqrt(
            simd<float, 8>(sum_sq / (float)K + eps))[0];

        float acc0 = 0.0f;
        float acc1 = 0.0f;
        for (int c = 0; c < n_chunks; c++) {
            int offset = c * VL;

            simd<float, VL> h = block_load<fp16, VL>(hidden_ptr + offset);
            simd<float, VL> nw = block_load<fp16, VL>(norm_w_ptr + offset);
            simd<float, VL> normed = h * inv_rms * nw;

            simd<uint8_t, VL> w0_raw = block_load<uint8_t, VL>(
                w0_ptr + (size_t)gid * K + offset);
            simd<float, VL> w0_f = fp8_dequant_rng2_mode<VL, FP8_MODE>(w0_raw);
            acc0 += reduce<float>(normed * w0_f, std::plus<>());

            simd<uint8_t, VL> w1_raw = block_load<uint8_t, VL>(
                w1_ptr + (size_t)gid * K + offset);
            simd<float, VL> w1_f = fp8_dequant_rng2_mode<VL, FP8_MODE>(w1_raw);
            acc1 += reduce<float>(normed * w1_f, std::plus<>());
        }

        float first = acc0 * *s0_ptr;
        float second = acc1 * *s1_ptr;
        float gated = gelu_tanh_scalar(first) * second;

        out_ptr[gid] = fp16(gated);
    }
};

template<int VL, int K_SPLIT>
struct NormGEMV2_fp8_pert_kernel {
    const fp16*    hidden_ptr;   // [1, K] — read-only
    const fp16*    norm_w_ptr;   // [K]
    const uint8_t* w0_ptr;       // [N0, K] FP8
    const float*   s0_ptr;       // [1]
    fp16*          out_ptr;      // [1, N0]
    const uint8_t* w1_ptr;       // [N1, K] FP8
    const float*   s1_ptr;       // [1]
    int N0, N1, K;
    float eps;
    int fp8_mode;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        if constexpr (K_SPLIT > 1) {
            constexpr int kInvRmsSlmOffset = K_SPLIT * sizeof(float);
            constexpr int kOutput0SlmOffset = kInvRmsSlmOffset + sizeof(float);
            constexpr int kOutput1SlmOffset = kOutput0SlmOffset + K_SPLIT * sizeof(float);
            slm_init<kOutput1SlmOffset + K_SPLIT * sizeof(float)>();
        }

        int gid = item.get_group(0);
        int lid = item.get_local_id(0);
        if (gid >= N0) return;

        int k_per_thread = K / K_SPLIT;
        int k_start = lid * k_per_thread;
        int n_chunks = k_per_thread / VL;
        const uint8_t* w0_row = w0_ptr + (size_t)gid * K;
        const uint8_t* w1_row = w1_ptr + (size_t)gid * K;

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
            constexpr int kSumSqSlmOffset = 0;
            constexpr int kInvRmsSlmOffset = K_SPLIT * sizeof(float);
            constexpr int kOutput0SlmOffset = kInvRmsSlmOffset + sizeof(float);
            constexpr int kOutput1SlmOffset = kOutput0SlmOffset + K_SPLIT * sizeof(float);

            slm_block_store<float, 1>(kSumSqSlmOffset + lid * sizeof(float), simd<float, 1>(sum_sq));
            barrier();
            if (lid == 0) {
                simd<float, K_SPLIT> parts = slm_block_load<float, K_SPLIT>(kSumSqSlmOffset);
                float total_sum_sq = reduce<float>(parts, std::plus<>());
                inv_rms = sycl::ext::intel::esimd::rsqrt(
                    simd<float, 8>(total_sum_sq / (float)K + eps))[0];
                slm_block_store<float, 1>(kInvRmsSlmOffset, simd<float, 1>(inv_rms));
            }
            barrier();
            inv_rms = slm_block_load<float, 1>(kInvRmsSlmOffset)[0];
        }

        // Pass 2: accumulate h*nw*w first, then apply inv_rms once after reduce.
        // This removes one vector multiply from the hot loop.
        simd<float, VL> acc0 = 0.0f;
        simd<float, VL> acc1 = 0.0f;
        if (fp8_mode == 0) {
            for (int c = 0; c < n_chunks; c++) {
                int offset = k_start + c * VL;

                simd<float, VL> h = block_load<fp16, VL>(hidden_ptr + offset);
                simd<float, VL> nw = block_load<fp16, VL>(norm_w_ptr + offset);
                simd<float, VL> hn = h * nw;

                simd<uint8_t, VL> w0_raw = block_load<uint8_t, VL>(w0_row + offset);
                simd<float, VL> w0_f = fp8_dequant_rng2_mode<VL, 0>(w0_raw);
                acc0 += hn * w0_f;

                simd<uint8_t, VL> w1_raw = block_load<uint8_t, VL>(w1_row + offset);
                simd<float, VL> w1_f = fp8_dequant_rng2_mode<VL, 0>(w1_raw);
                acc1 += hn * w1_f;
            }
        } else {
            for (int c = 0; c < n_chunks; c++) {
                int offset = k_start + c * VL;

                simd<float, VL> h = block_load<fp16, VL>(hidden_ptr + offset);
                simd<float, VL> nw = block_load<fp16, VL>(norm_w_ptr + offset);
                simd<float, VL> hn = h * nw;

                simd<uint8_t, VL> w0_raw = block_load<uint8_t, VL>(w0_row + offset);
                simd<float, VL> w0_f = fp8_dequant_rng2_mode<VL, 1>(w0_raw);
                acc0 += hn * w0_f;

                simd<uint8_t, VL> w1_raw = block_load<uint8_t, VL>(w1_row + offset);
                simd<float, VL> w1_f = fp8_dequant_rng2_mode<VL, 1>(w1_raw);
                acc1 += hn * w1_f;
            }
        }

        float my_sum0 = reduce<float>(acc0, std::plus<>());
        float my_sum1 = reduce<float>(acc1, std::plus<>());
        float s0 = *s0_ptr;
        float s1 = *s1_ptr;

        if constexpr (K_SPLIT == 1) {
            float first = my_sum0 * inv_rms * s0;
            float second = my_sum1 * inv_rms * s1;
            float gated = gelu_tanh_scalar(first) * second;
            out_ptr[gid] = fp16(gated);
        } else {
            constexpr int kInvRmsSlmOffset = K_SPLIT * sizeof(float);
            constexpr int kOutput0SlmOffset = kInvRmsSlmOffset + sizeof(float);
            constexpr int kOutput1SlmOffset = kOutput0SlmOffset + K_SPLIT * sizeof(float);

            slm_block_store<float, 1>(kOutput0SlmOffset + lid * sizeof(float), simd<float, 1>(my_sum0));
            slm_block_store<float, 1>(kOutput1SlmOffset + lid * sizeof(float), simd<float, 1>(my_sum1));
            barrier();
            if (lid == 0) {
                simd<float, K_SPLIT> parts0 = slm_block_load<float, K_SPLIT>(kOutput0SlmOffset);
                simd<float, K_SPLIT> parts1 = slm_block_load<float, K_SPLIT>(kOutput1SlmOffset);
                float sum0 = reduce<float>(parts0, std::plus<>());
                float sum1 = reduce<float>(parts1, std::plus<>());
                float inv = slm_block_load<float, 1>(kInvRmsSlmOffset)[0];
                float first = sum0 * inv * s0;
                float second = sum1 * inv * s1;
                float gated = gelu_tanh_scalar(first) * second;
                out_ptr[gid] = fp16(gated);
            }
        }
    }
};

inline void norm_gemv2_geglu_fp8_pert_host(
    const fp16* hidden_ptr, const fp16* norm_w_ptr,
    const uint8_t* w0, const float* s0, fp16* out,
    const uint8_t* w1, const float* s1,
    int N0, int N1, int K, int vl, int ks, float eps, int fp8_mode,
    sycl::queue& q)
{
    TORCH_CHECK(N0 == N1, "norm_gemv2_geglu_fp8_pert_host: only equal-row weights are supported");

    int global = N0 * ks;
    int local = ks;

    if (vl == 256 && ks == 1) {
        if (fp8_mode == 0) {
            q.submit([&](sycl::handler& cgh) {
                cgh.parallel_for(
                    sycl::nd_range<1>(global, local),
                    NormGEMV2_fp8_pert_kernel_256_1<0>{
                        hidden_ptr,
                        norm_w_ptr,
                        w0,
                        s0,
                        out,
                        w1,
                        s1,
                        N0,
                        N1,
                        K,
                        eps});
            });
        } else {
            q.submit([&](sycl::handler& cgh) {
                cgh.parallel_for(
                    sycl::nd_range<1>(global, local),
                    NormGEMV2_fp8_pert_kernel_256_1<1>{
                        hidden_ptr,
                        norm_w_ptr,
                        w0,
                        s0,
                        out,
                        w1,
                        s1,
                        N0,
                        N1,
                        K,
                        eps});
            });
        }
        return;
    }

    #define LAUNCH_FUSED(V, S) \
        q.submit([&](sycl::handler& cgh) { \
            cgh.parallel_for( \
                sycl::nd_range<1>(global, local), \
                NormGEMV2_fp8_pert_kernel<V, S>{ \
                    hidden_ptr, norm_w_ptr, \
                    w0, s0, out, w1, s1, \
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

inline void norm_gemv2_geglu_fp8_pert_host(
    const fp16* hidden_ptr, const fp16* norm_w_ptr,
    const uint8_t* w0, const float* s0, fp16* out,
    const uint8_t* w1, const float* s1,
    int N0, int N1, int K, float eps, int fp8_mode,
    sycl::queue& q)
{
    int total_N = N0 + N1;
    int vl, ks;
    select_vl_ks_norm_gemv2_geglu(total_N, K, vl, ks);
    norm_gemv2_geglu_fp8_pert_host(
        hidden_ptr,
        norm_w_ptr,
        w0,
        s0,
        out,
        w1,
        s1,
        N0,
        N1,
        K,
        vl,
        ks,
        eps,
        fp8_mode,
        q);
}