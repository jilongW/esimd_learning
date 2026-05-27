/* fused_add_rms_norm_batched.h — Batched Fused residual add + RMSNorm.
 *
 * Multi-row version of fused_add_rms_norm.h:
 *   residual[i] += hidden[i]   (in-place)
 *   hidden[i] = rmsnorm(residual[i]) * weight
 *
 * Grid: rows WGs, 1 thread each. K=2048 → 4 iterations with VL=512.
 * Replaces PyTorch dispatch chain (~87us) with single kernel (~5us).
 */

#pragma once
#include "utils.h"

template <typename scalar_t>
struct FusedAddRmsNorm_batched_kernel {
    scalar_t*       hidden_ptr;    // [rows, K] — input and output
    scalar_t*       residual_ptr;  // [rows, K] — updated in-place
    const scalar_t* weight_ptr;    // [K] — Gemma norm weight (w+1.0)
    int rows;
    int K;
    float eps;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        int row = item.get_group(0);
        if (row >= rows) return;

        constexpr int VL = 512;
        int n_chunks = K / VL;
        const int base = row * K;

        // Pass 1: residual += hidden, accumulate sum_sq
        float sum_sq = 0.0f;
        for (int c = 0; c < n_chunks; c++) {
            int offset = base + c * VL;
            simd<float, VL> h = block_load<scalar_t, VL>(hidden_ptr + offset);
            simd<float, VL> r = block_load<scalar_t, VL>(residual_ptr + offset);
            simd<float, VL> added = h + r;

            block_store<scalar_t, VL>(
                residual_ptr + offset,
                simd<scalar_t, VL>(added));

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

        // Pass 2: normalize and write output
        for (int c = 0; c < n_chunks; c++) {
            int offset = base + c * VL;
            simd<float, VL> r = block_load<scalar_t, VL>(residual_ptr + offset);
            simd<float, VL> w = block_load<scalar_t, VL>(weight_ptr + c * VL);
            simd<float, VL> normed = r * inv_rms * w;
            block_store<scalar_t, VL>(hidden_ptr + offset, simd<scalar_t, VL>(normed));
        }
    }
};

template <typename scalar_t>
inline void fused_add_rms_norm_batched_host_impl(
    scalar_t* hidden_ptr, scalar_t* residual_ptr, const scalar_t* weight_ptr,
    int rows, int K, float eps, sycl::queue& q)
{
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>({(size_t)rows}, {1}),
            FusedAddRmsNorm_batched_kernel<scalar_t>{
                hidden_ptr, residual_ptr, weight_ptr, rows, K, eps});
    });
}

inline void fused_add_rms_norm_batched_host(
    uint8_t* hidden_ptr, uint8_t* residual_ptr, const uint8_t* weight_ptr,
    int rows, int K, float eps, bool is_bf16, sycl::queue& q)
{
    if (is_bf16) {
        fused_add_rms_norm_batched_host_impl<bf16>(
            reinterpret_cast<bf16*>(hidden_ptr),
            reinterpret_cast<bf16*>(residual_ptr),
            reinterpret_cast<const bf16*>(weight_ptr),
            rows, K, eps, q);
        return;
    }

    fused_add_rms_norm_batched_host_impl<fp16>(
        reinterpret_cast<fp16*>(hidden_ptr),
        reinterpret_cast<fp16*>(residual_ptr),
        reinterpret_cast<const fp16*>(weight_ptr),
        rows, K, eps, q);
}
