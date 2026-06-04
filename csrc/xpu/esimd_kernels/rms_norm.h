/* rms_norm.h — ESIMD RMSNorm kernel.
 *
 * Replaces PyTorch kernel dispatches (float cast, pow, mean, rsqrt,
 * mul) with a single ESIMD kernel. For [rows, V]:
 *   output[i] = rmsnorm(x[i]) * weight
 * where rmsnorm(x) = x / rms(x).
 */

#pragma once
#include "utils.h"

inline void select_rms_norm_vl_ks(uint32_t rows, uint32_t V, int& vl, int& ks) {
    vl = 512;
    ks = 1;

    if ( V <= 512){
        vl = 256;
        ks = 1;
    } else if ( V <= 2560){
        vl = 128;
        ks = 10;
    } else if ( V <= 5120){
        vl = 256;
        ks = 10;
     }
    else {
        vl = 1024;
        ks = 1;
    }

    int kpt = V / ks;
    while (vl > kpt || kpt % vl != 0) {
        if (vl > 128) {
            vl /= 2;
        } else if (ks > 1) {
            ks /= 2;
            kpt = V / ks;
        } else {
            break;
        }
    }
}

template <typename scalar_t, int VL, int KS>
struct RmsNorm_kernel {
    const scalar_t* x_ptr;       // [rows, V]
    const scalar_t* weight_ptr;  // [V]
    scalar_t* output_ptr;        // [rows, V]
    int rows;
    int V;
    float eps;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        slm_init<KS * sizeof(float)>();

        int row = item.get_group(0);
        int lid = item.get_local_id(0);
        if (row >= rows) return;

        int n_chunks = V / VL;
        int base = row * V;

        float sum_sq = 0.0f;
        for (int c = lid; c < n_chunks; c += KS) {
            int offset = base + c * VL;
            simd<float, VL> x_f = block_load<scalar_t, VL>(x_ptr + offset);
            simd<float, VL> x_sq = x_f * x_f;
            sum_sq += reduce<float>(x_sq, std::plus<>());
        }

        slm_block_store<float, 1>(lid * sizeof(float), simd<float, 1>(sum_sq));
        barrier();

        float inv_rms = 0.0f;
        if (lid == 0) {
            simd<float, KS> parts = slm_block_load<float, KS>(0);
            float total_sum_sq = reduce<float>(parts, std::plus<>());
            inv_rms = sycl::ext::intel::esimd::rsqrt(
                simd<float, 8>(total_sum_sq / (float)V + eps))[0];
            slm_block_store<float, 1>(0, simd<float, 1>(inv_rms));
        }
        barrier();
        inv_rms = slm_block_load<float, 1>(0)[0];
        barrier();

        for (int c = lid; c < n_chunks; c += KS) {
            int offset = base + c * VL;
            simd<float, VL> x_f = block_load<scalar_t, VL>(x_ptr + offset);
            simd<float, VL> w_f = block_load<scalar_t, VL>(weight_ptr + c * VL);
            simd<float, VL> result = x_f * inv_rms * w_f;
            block_store<scalar_t, VL>(output_ptr + offset, simd<scalar_t, VL>(result));
        }
    }
};

template <typename scalar_t, int VL, int KS>
inline void rms_norm_host_impl(
    const scalar_t* x_ptr,
    const scalar_t* weight_ptr,
    scalar_t* output_ptr,
    int rows,
    int V,
    float eps,
    sycl::queue& q)
{
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>({(size_t)(rows * KS)}, {(size_t)KS}),
            RmsNorm_kernel<scalar_t, VL, KS>{x_ptr, weight_ptr, output_ptr, rows, V, eps});
    });
}

template <typename scalar_t>
inline void rms_norm_256_host(
    const scalar_t* x_ptr,
    const scalar_t* weight_ptr,
    scalar_t* output_ptr,
    int rows,
    int V,
    float eps,
    sycl::queue& q);

inline void rms_norm_host(
    const uint8_t* x_ptr,
    const uint8_t* weight_ptr,
    uint8_t* output_ptr,
    int rows,
    int V,
    float eps,
    int vl,
    int ks,
    bool is_bf16,
    sycl::queue& q);

inline void rms_norm_host(
    const uint8_t* x_ptr,
    const uint8_t* weight_ptr,
    uint8_t* output_ptr,
    int rows,
    int V,
    float eps,
    bool is_bf16,
    sycl::queue& q);



inline void rms_norm_host(
    const uint8_t* x_ptr,
    const uint8_t* weight_ptr,
    uint8_t* output_ptr,
    int rows,
    int V,
    float eps,
    int vl,
    int ks,
    bool is_bf16,
    sycl::queue& q)
{
    if (V == 256 && vl == 256 && ks == 1) {
        if (is_bf16) {
            rms_norm_256_host<bf16>(
                reinterpret_cast<const bf16*>(x_ptr),
                reinterpret_cast<const bf16*>(weight_ptr),
                reinterpret_cast<bf16*>(output_ptr),
                rows,
                V,
                eps,
                q);
        } else {
            rms_norm_256_host<fp16>(
                reinterpret_cast<const fp16*>(x_ptr),
                reinterpret_cast<const fp16*>(weight_ptr),
                reinterpret_cast<fp16*>(output_ptr),
                rows,
                V,
                eps,
                q);
        }
        return;
    }

    if (is_bf16) {
        auto* typed_x = reinterpret_cast<const bf16*>(x_ptr);
        auto* typed_w = reinterpret_cast<const bf16*>(weight_ptr);
        auto* typed_out = reinterpret_cast<bf16*>(output_ptr);

        if (vl == 1024 && ks == 1) { rms_norm_host_impl<bf16, 1024, 1>(typed_x, typed_w, typed_out, rows, V, eps, q); }
        else if (vl == 1024 && ks == 2) { rms_norm_host_impl<bf16, 1024, 2>(typed_x, typed_w, typed_out, rows, V, eps, q); }
        else if (vl == 1024 && ks == 4) { rms_norm_host_impl<bf16, 1024, 4>(typed_x, typed_w, typed_out, rows, V, eps, q); }
        else if (vl == 1024 && ks == 5) { rms_norm_host_impl<bf16, 1024, 5>(typed_x, typed_w, typed_out, rows, V, eps, q); }
        else if (vl == 1024 && ks == 8) { rms_norm_host_impl<bf16, 1024, 8>(typed_x, typed_w, typed_out, rows, V, eps, q); }
        else if (vl == 1024 && ks == 10) { rms_norm_host_impl<bf16, 1024, 10>(typed_x, typed_w, typed_out, rows, V, eps, q); }
        else if (vl == 512 && ks == 1) { rms_norm_host_impl<bf16, 512, 1>(typed_x, typed_w, typed_out, rows, V, eps, q); }
        else if (vl == 512 && ks == 2) { rms_norm_host_impl<bf16, 512, 2>(typed_x, typed_w, typed_out, rows, V, eps, q); }
        else if (vl == 512 && ks == 4) { rms_norm_host_impl<bf16, 512, 4>(typed_x, typed_w, typed_out, rows, V, eps, q); }
        else if (vl == 512 && ks == 5) { rms_norm_host_impl<bf16, 512, 5>(typed_x, typed_w, typed_out, rows, V, eps, q); }
        else if (vl == 512 && ks == 8) { rms_norm_host_impl<bf16, 512, 8>(typed_x, typed_w, typed_out, rows, V, eps, q); }
        else if (vl == 512 && ks == 10) { rms_norm_host_impl<bf16, 512, 10>(typed_x, typed_w, typed_out, rows, V, eps, q); }
        else if (vl == 256 && ks == 1) { rms_norm_host_impl<bf16, 256, 1>(typed_x, typed_w, typed_out, rows, V, eps, q); }
        else if (vl == 256 && ks == 2) { rms_norm_host_impl<bf16, 256, 2>(typed_x, typed_w, typed_out, rows, V, eps, q); }
        else if (vl == 256 && ks == 4) { rms_norm_host_impl<bf16, 256, 4>(typed_x, typed_w, typed_out, rows, V, eps, q); }
        else if (vl == 256 && ks == 5) { rms_norm_host_impl<bf16, 256, 5>(typed_x, typed_w, typed_out, rows, V, eps, q); }
        else if (vl == 256 && ks == 8) { rms_norm_host_impl<bf16, 256, 8>(typed_x, typed_w, typed_out, rows, V, eps, q); }
        else if (vl == 256 && ks == 10) { rms_norm_host_impl<bf16, 256, 10>(typed_x, typed_w, typed_out, rows, V, eps, q); }
        else if (vl == 128 && ks == 1) { rms_norm_host_impl<bf16, 128, 1>(typed_x, typed_w, typed_out, rows, V, eps, q); }
        else if (vl == 128 && ks == 2) { rms_norm_host_impl<bf16, 128, 2>(typed_x, typed_w, typed_out, rows, V, eps, q); }
        else if (vl == 128 && ks == 4) { rms_norm_host_impl<bf16, 128, 4>(typed_x, typed_w, typed_out, rows, V, eps, q); }
        else if (vl == 128 && ks == 5) { rms_norm_host_impl<bf16, 128, 5>(typed_x, typed_w, typed_out, rows, V, eps, q); }
        else if (vl == 128 && ks == 8) { rms_norm_host_impl<bf16, 128, 8>(typed_x, typed_w, typed_out, rows, V, eps, q); }
        else if (vl == 128 && ks == 10) { rms_norm_host_impl<bf16, 128, 10>(typed_x, typed_w, typed_out, rows, V, eps, q); }
        else { throw std::runtime_error("rms_norm_host: unsupported vl/ks"); }
        return;
    }

    auto* typed_x = reinterpret_cast<const fp16*>(x_ptr);
    auto* typed_w = reinterpret_cast<const fp16*>(weight_ptr);
    auto* typed_out = reinterpret_cast<fp16*>(output_ptr);

    if (vl == 1024 && ks == 1) { rms_norm_host_impl<fp16, 1024, 1>(typed_x, typed_w, typed_out, rows, V, eps, q); }
    else if (vl == 1024 && ks == 2) { rms_norm_host_impl<fp16, 1024, 2>(typed_x, typed_w, typed_out, rows, V, eps, q); }
    else if (vl == 1024 && ks == 4) { rms_norm_host_impl<fp16, 1024, 4>(typed_x, typed_w, typed_out, rows, V, eps, q); }
    else if (vl == 1024 && ks == 5) { rms_norm_host_impl<fp16, 1024, 5>(typed_x, typed_w, typed_out, rows, V, eps, q); }
    else if (vl == 1024 && ks == 8) { rms_norm_host_impl<fp16, 1024, 8>(typed_x, typed_w, typed_out, rows, V, eps, q); }
    else if (vl == 1024 && ks == 10) { rms_norm_host_impl<fp16, 1024, 10>(typed_x, typed_w, typed_out, rows, V, eps, q); }
    else if (vl == 512 && ks == 1) { rms_norm_host_impl<fp16, 512, 1>(typed_x, typed_w, typed_out, rows, V, eps, q); }
    else if (vl == 512 && ks == 2) { rms_norm_host_impl<fp16, 512, 2>(typed_x, typed_w, typed_out, rows, V, eps, q); }
    else if (vl == 512 && ks == 4) { rms_norm_host_impl<fp16, 512, 4>(typed_x, typed_w, typed_out, rows, V, eps, q); }
    else if (vl == 512 && ks == 5) { rms_norm_host_impl<fp16, 512, 5>(typed_x, typed_w, typed_out, rows, V, eps, q); }
    else if (vl == 512 && ks == 8) { rms_norm_host_impl<fp16, 512, 8>(typed_x, typed_w, typed_out, rows, V, eps, q); }
    else if (vl == 512 && ks == 10) { rms_norm_host_impl<fp16, 512, 10>(typed_x, typed_w, typed_out, rows, V, eps, q); }
    else if (vl == 256 && ks == 1) { rms_norm_host_impl<fp16, 256, 1>(typed_x, typed_w, typed_out, rows, V, eps, q); }
    else if (vl == 256 && ks == 2) { rms_norm_host_impl<fp16, 256, 2>(typed_x, typed_w, typed_out, rows, V, eps, q); }
    else if (vl == 256 && ks == 4) { rms_norm_host_impl<fp16, 256, 4>(typed_x, typed_w, typed_out, rows, V, eps, q); }
    else if (vl == 256 && ks == 5) { rms_norm_host_impl<fp16, 256, 5>(typed_x, typed_w, typed_out, rows, V, eps, q); }
    else if (vl == 256 && ks == 8) { rms_norm_host_impl<fp16, 256, 8>(typed_x, typed_w, typed_out, rows, V, eps, q); }
    else if (vl == 256 && ks == 10) { rms_norm_host_impl<fp16, 256, 10>(typed_x, typed_w, typed_out, rows, V, eps, q); }
    else if (vl == 128 && ks == 1) { rms_norm_host_impl<fp16, 128, 1>(typed_x, typed_w, typed_out, rows, V, eps, q); }
    else if (vl == 128 && ks == 2) { rms_norm_host_impl<fp16, 128, 2>(typed_x, typed_w, typed_out, rows, V, eps, q); }
    else if (vl == 128 && ks == 4) { rms_norm_host_impl<fp16, 128, 4>(typed_x, typed_w, typed_out, rows, V, eps, q); }
    else if (vl == 128 && ks == 5) { rms_norm_host_impl<fp16, 128, 5>(typed_x, typed_w, typed_out, rows, V, eps, q); }
    else if (vl == 128 && ks == 8) { rms_norm_host_impl<fp16, 128, 8>(typed_x, typed_w, typed_out, rows, V, eps, q); }
    else if (vl == 128 && ks == 10) { rms_norm_host_impl<fp16, 128, 10>(typed_x, typed_w, typed_out, rows, V, eps, q); }
    else { throw std::runtime_error("rms_norm_host: unsupported vl/ks"); }
}

inline void rms_norm_host(
    const uint8_t* x_ptr,
    const uint8_t* weight_ptr,
    uint8_t* output_ptr,
    int rows,
    int V,
    float eps,
    bool is_bf16,
    sycl::queue& q){
        int vl = 128;
        int ks = 1;
        select_rms_norm_vl_ks(rows, V, vl, ks);
        using RmsNormDispatchFn = void (*)(
            const uint8_t*,
            const uint8_t*,
            uint8_t*,
            int,
            int,
            float,
            int,
            int,
            bool,
            sycl::queue&);
        RmsNormDispatchFn dispatch = &rms_norm_host;
        dispatch(x_ptr, weight_ptr, output_ptr, rows, V, eps, vl, ks, is_bf16, q);
    }

template <typename scalar_t>
struct RmsNorm256_kernel {
    static constexpr int VEC = 256;

    const scalar_t* x_ptr;   // [rows, 256]
    const scalar_t* weight_ptr;  // [256]
    scalar_t* output_ptr;    // [rows, 256]
    int rows;
    float eps;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        int row = item.get_group(0);
        if (row >= rows) return;

        const int offset = row * VEC;

        simd<float, 256> x_f = block_load<scalar_t, 256>(x_ptr + offset);
        simd<float, 256> w_f = block_load<scalar_t, 256>(weight_ptr);

        simd<float, 256> x_sq = x_f * x_f;
        x_sq.select<128, 1>(0) += x_sq.select<128, 1>(128);
        x_sq.select<64, 1>(0) += x_sq.select<64, 1>(64);
        x_sq.select<32, 1>(0) += x_sq.select<32, 1>(32);
        x_sq.select<16, 1>(0) += x_sq.select<16, 1>(16);
        x_sq.select<8, 1>(0) += x_sq.select<8, 1>(8);
        x_sq.select<4, 1>(0) += x_sq.select<4, 1>(4);
        x_sq.select<2, 1>(0) += x_sq.select<2, 1>(2);
        float sum_sq = x_sq[0] + x_sq[1];
        float inv_rms = sycl::ext::intel::esimd::rsqrt(
            simd<float, 8>(sum_sq / (float)VEC + eps))[0];

        simd<float, 256> result = x_f * inv_rms * w_f;
        block_store<scalar_t, 256>(output_ptr + offset, simd<scalar_t, 256>(result));
    }
};

template <typename scalar_t>
inline void rms_norm_256_host(
    const scalar_t* x_ptr,
    const scalar_t* weight_ptr,
    scalar_t* output_ptr,
    int rows,
    int V,
    float eps,
    sycl::queue& q)
{
    if (V != RmsNorm256_kernel<scalar_t>::VEC) {
        throw std::runtime_error("rms_norm_256_host: only supports V=256");
    }

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>({(size_t)rows}, {1}),
            RmsNorm256_kernel<scalar_t>{x_ptr, weight_ptr, output_ptr, rows, eps});
    });
}