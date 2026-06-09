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

/* ================================================================
 * Kernel: Fused ResidualAdd + RMSNorm + FP8 GEMV (per-tensor scale)
 *
 * Dispatch is aligned with fp8_GEMV_v2.h: VL and K_SPLIT are selected
 * by select_vl_ks(), and GEMV uses nd_range(global=N*K_SPLIT, local=K_SPLIT).
 * ================================================================ */
template<int VL, int K_SPLIT>
struct ResAddNormGEMV_fp8_pert_kernel {
    fp16*          hidden_ptr;   // [1, K] — input (read-only for this kernel)
    fp16*          residual_ptr; // [1, K] — updated in-place
    const fp16*    norm_w_ptr;   // [K] — Gemma norm weight (w+1.0)
    const uint8_t* gemv_weight;  // [N, K] FP8
    const float*   gemv_scale;   // [1] or [2]
    fp16*          output;       // [1, N] — router logits
    fp16*          normed_out;   // [1, K] — normed hidden_states (for MoE experts)
    int N, K;
    int gemv_scale_count;
    float eps;
    int fp8_mode;

    SYCL_ESIMD_FUNCTION float gemv_scale_for_n(int n) const {
        if (gemv_scale_count == 1) {
            return gemv_scale[0];
        }

        int split_n = (N + 1) / 2;
        return gemv_scale[n < split_n ? 0 : 1];
    }

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        if constexpr (K_SPLIT > 1) {
            slm_init<K_SPLIT * sizeof(float)>();
        }

        int n = item.get_group(0);
        int lid = item.get_local_id(0);
        if (n >= N) return;

        int k_per_thread = K / K_SPLIT;
        int k_start = lid * k_per_thread;
        int n_chunks = k_per_thread / VL;

        float sum_sq = 0.0f;

        for (int c = 0; c < n_chunks; c++) {
            int offset = k_start + c * VL;
            simd<float, VL> h = block_load<fp16, VL>(hidden_ptr + offset);
            simd<float, VL> r = block_load<fp16, VL>(residual_ptr + offset);

            simd<float, VL> added = h + r;

            simd<float, VL> sq = added * added;
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

        simd<float, VL> acc = 0.0f;

        for (int c = 0; c < n_chunks; c++) {
            int offset = k_start + c * VL;

            simd<float, VL> h = block_load<fp16, VL>(hidden_ptr + offset);
            simd<float, VL> r = block_load<fp16, VL>(residual_ptr + offset);
            simd<float, VL> added = h + r;
            simd<float, VL> nw = block_load<fp16, VL>(norm_w_ptr + offset);
            simd<float, VL> normed = added * inv_rms * nw;

            if (n == 0) {
                block_store<fp16, VL>(residual_ptr + offset, simd<fp16, VL>(added));
                block_store<fp16, VL>(normed_out + offset, simd<fp16, VL>(normed));
            }

            simd<uint8_t, VL> w_raw = block_load<uint8_t, VL>(
                gemv_weight + (size_t)n * K + offset);
            simd<float, VL> w_f = fp8_dequant_rng<VL>(w_raw, fp8_mode);
            acc += normed * w_f;
        }

        float my_sum = reduce<float>(acc, std::plus<>()) * gemv_scale_for_n(n);

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
    int vl, ks;
    auto dev = q.get_device();
    select_vl_ks(N, K, vl, ks, &dev);

    int global = N * ks;
    int local = ks;

    #define LAUNCH_FUSED(V, S) \
        q.submit([&](sycl::handler& cgh) { \
            cgh.parallel_for( \
                sycl::nd_range<1>(global, local), \
                ResAddNormGEMV_fp8_pert_kernel<V, S>{ \
                    hidden_ptr, residual_ptr, norm_w_ptr, \
                    gemv_weight, gemv_scale, output, normed_out, \
                    N, K, gemv_scale_count, eps, fp8_mode}); \
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
