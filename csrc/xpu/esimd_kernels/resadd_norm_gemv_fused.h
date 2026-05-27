/* resadd_norm_gemv_fused.h — Fused ResidualAdd + RMSNorm + FP8 GEMV.
 *
 * Combines three operations into a single kernel:
 *   1. Residual add: residual = hidden_states + residual  (in-place)
 *   2. RMSNorm (Gemma-style): normed = residual / rms(residual) * weight
 *      where weight is pre-adjusted (w+1.0 already applied by caller)
 *   3. GEMV: output = normed @ dequant(gemv_weight^T) * scale
 *
 * Designed for Qwen3-Next post_attention_layernorm + MoE router:
 *   hidden_states: [1, K] fp16   (K=2048)
 *   residual:      [1, K] fp16   (updated in-place)
 *   norm_weight:   [K] fp16      (Gemma _gemma_w = original_w + 1.0)
 *   gemv_weight:   [N, K] FP8    (N=512 for router)
 *   gemv_scale:    [1] float32
 *   output:        [1, N] fp16
 *
 * Grid: N work-groups, 1 thread each.
 * Each WG redundantly computes residual_add + norm (data in L3 cache).
 * Only WG 0 writes the updated residual back to global memory.
 *
 * For K=2048 with VL=512: 4 loop iterations for norm, then 4 for GEMV.
 * Interleaved approach: compute norm chunk + GEMV chunk per iteration.
 */

#pragma once
#include "utils.h"

namespace xesimd = sycl::ext::intel::experimental::esimd;

template<int VL>
SYCL_ESIMD_FUNCTION inline simd<float, VL> fp8_dequant_rng(
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

struct ResAddNormOnce_fp8_kernel {
    const fp16*    hidden_ptr;   // [1, K] — input
    fp16*          residual_ptr; // [1, K] — updated in-place
    const fp16*    norm_w_ptr;   // [K] — Gemma norm weight (w+1.0)
    fp16*          normed_out;   // [1, K] — normed hidden_states
    int K;
    float eps;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        if (item.get_group(0) > 0) return;

        constexpr int VL = 512;
        const int n_chunks = K / VL;
        float sum_sq = 0.0f;

        for (int c = 0; c < n_chunks; c++) {
            int offset = c * VL;
            simd<float, VL> h = block_load<fp16, VL>(hidden_ptr + offset);
            simd<float, VL> r = block_load<fp16, VL>(residual_ptr + offset);
            simd<float, VL> added = h + r;

            block_store<fp16, VL>(residual_ptr + offset, simd<fp16, VL>(added));

            simd<float, VL> sq = added * added;
            sq.select<256,1>(0) += sq.select<256,1>(256);
            sq.select<128,1>(0) += sq.select<128,1>(128);
            sq.select<64,1>(0) += sq.select<64,1>(64);
            sq.select<32,1>(0) += sq.select<32,1>(32);
            sq.select<16,1>(0) += sq.select<16,1>(16);
            sq.select<8,1>(0) += sq.select<8,1>(8);
            sq.select<4,1>(0) += sq.select<4,1>(4);
            sq.select<2,1>(0) += sq.select<2,1>(2);
            sum_sq += (float)sq[0] + (float)sq[1];
        }

        float inv_rms = sycl::ext::intel::esimd::rsqrt(
            simd<float, 8>(sum_sq / (float)K + eps))[0];

        for (int c = 0; c < n_chunks; c++) {
            int offset = c * VL;
            simd<float, VL> r = block_load<fp16, VL>(residual_ptr + offset);
            simd<float, VL> w = block_load<fp16, VL>(norm_w_ptr + offset);
            simd<float, VL> normed = r * inv_rms * w;
            block_store<fp16, VL>(normed_out + offset, simd<fp16, VL>(normed));
        }
    }
};

struct NormedGEMV_fp8_pert_kernel {
    const fp16*    normed_ptr;   // [1, K] — precomputed normalized hidden
    const uint8_t* gemv_weight;  // [N, K] FP8
    const float*   gemv_scale;   // [1] or [2]
    fp16*          output;       // [1, N] — router logits
    int N, K;
    int gemv_scale_count;
    int fp8_mode;

    SYCL_ESIMD_FUNCTION float gemv_scale_for_n(int n) const {
        if (gemv_scale_count == 1) {
            return gemv_scale[0];
        }

        int split_n = (N + 1) / 2;
        return gemv_scale[n < split_n ? 0 : 1];
    }

    template<int MAX_CHUNKS>
    void run_large_k_impl(int n) const SYCL_ESIMD_FUNCTION {
        constexpr int VL = 512;
        constexpr int PREFETCH_BYTES = 64;
        const int n_chunks = K / VL;

        simd<float, VL> acc = 0.0f;

        if (n_chunks > 0) {
            xesimd::lsc_prefetch<uint8_t, PREFETCH_BYTES, xesimd::lsc_data_size::default_size,
                xesimd::cache_hint::uncached, xesimd::cache_hint::cached>(
                gemv_weight + (size_t)n * K);
        }

        for (int c = 0; c < n_chunks; c++) {
            int offset = c * VL;

            if (c + 1 < n_chunks) {
                xesimd::lsc_prefetch<uint8_t, PREFETCH_BYTES, xesimd::lsc_data_size::default_size,
                    xesimd::cache_hint::uncached, xesimd::cache_hint::cached>(
                    gemv_weight + (size_t)n * K + (c + 1) * VL);
            }

                simd<float, VL> normed = block_load<fp16, VL>(normed_ptr + offset);

            simd<uint8_t, VL> w_raw = block_load<uint8_t, VL>(
                gemv_weight + (size_t)n * K + offset);
            simd<float, VL> w_f = fp8_dequant_rng<VL>(w_raw, fp8_mode);
            acc += normed * w_f;
        }

        acc.select<256,1>(0) += acc.select<256,1>(256);
        acc.select<128,1>(0) += acc.select<128,1>(128);
        acc.select<64,1>(0) += acc.select<64,1>(64);
        acc.select<32,1>(0) += acc.select<32,1>(32);
        acc.select<16,1>(0) += acc.select<16,1>(16);
        acc.select<8,1>(0) += acc.select<8,1>(8);
        acc.select<4,1>(0) += acc.select<4,1>(4);
        acc.select<2,1>(0) += acc.select<2,1>(2);
        float dot = ((float)acc[0] + (float)acc[1]) * gemv_scale_for_n(n);
        output[n] = fp16(dot);
    }

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        int n = item.get_group(0);
        if (n >= N) return;

        if (K <= 4096) {
            run_large_k_impl<8>(n);
        } else {
            run_large_k_impl<16>(n);
        }
    }
};

/* Host dispatcher */
inline void resadd_norm_gemv_fp8_pert_host(
    fp16* hidden_ptr,
    fp16* residual_ptr,
    const fp16* norm_w_ptr,
    const uint8_t* gemv_weight,
    const float* gemv_scale,
    fp16* output,
    fp16* normed_out,
    int N, int K,
    int gemv_scale_count,
    float eps,
    int fp8_mode,
    sycl::queue& q)
{
    auto norm_event = q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(1, 1),
            ResAddNormOnce_fp8_kernel{
                hidden_ptr, residual_ptr, norm_w_ptr,
                normed_out, K, eps});
    });

    q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(norm_event);
        cgh.parallel_for(
            sycl::nd_range<1>(N, 1),
            NormedGEMV_fp8_pert_kernel{
                normed_out, gemv_weight, gemv_scale, output,
                N, K, gemv_scale_count, fp8_mode});
    });
}
