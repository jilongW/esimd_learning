#pragma once
#include "utils.h"
#include <cmath>

// Fused QKV Split + RMSNorm + RoPE kernel
//
// Optimized design:
// 1) Host-side Q/K/V split dispatch (no Q/K/V branching in device code).
// 2) Compile-time KV shared specialization for K/V kernels.
// 3) Shared helpers for RoPE and RMS computations.
// 4) Dedicated 512 path using 2x256 chunks to keep backend compilation stable.

#define QKV_LOCATION_Q 0
#define QKV_LOCATION_K 1
#define QKV_LOCATION_V 2

template <int HALF_DIM>
ESIMD_INLINE void load_rope_cos_sin(
    fp16* ropeCosSinCache,
    uint32_t rowOffset,
    simd<float, HALF_DIM>& fcos,
    simd<float, HALF_DIM>& fsin) {
    constexpr int CHUNK32 = HALF_DIM / 32;
#pragma unroll
    for (int kk = 0; kk < CHUNK32; kk++) {
        simd<fp16, 32> c16 = block_load<fp16, 32>(ropeCosSinCache + rowOffset + 32 * kk);
        fcos.template select<32, 1>(32 * kk) = c16;
    }
#pragma unroll
    for (int kk = 0; kk < CHUNK32; kk++) {
        simd<fp16, 32> s16 = block_load<fp16, 32>(ropeCosSinCache + rowOffset + HALF_DIM + 32 * kk);
        fsin.template select<32, 1>(32 * kk) = s16;
    }
}

template <int HEAD_DIM>
ESIMD_INLINE void apply_rope_inplace(
    simd<float, HEAD_DIM>& x,
    fp16* ropeCosSinCache,
    uint32_t rowOffset) {
    constexpr int HALF_DIM = HEAD_DIM / 2;
    simd<float, HALF_DIM> fcos, fsin;
    load_rope_cos_sin<HALF_DIM>(ropeCosSinCache, rowOffset, fcos, fsin);

    simd<float, HALF_DIM> x1 = x.template select<HALF_DIM, 1>(0);
    simd<float, HALF_DIM> x2 = x.template select<HALF_DIM, 1>(HALF_DIM);
    x.template select<HALF_DIM, 1>(0) = x1 * fcos - x2 * fsin;
    x.template select<HALF_DIM, 1>(HALF_DIM) = x2 * fcos + x1 * fsin;
}

ESIMD_INLINE void apply_rope_inplace_512(
    simd<float, 256>& x0,
    simd<float, 256>& x1,
    fp16* ropeCosSinCache,
    uint32_t rowOffset) {
    constexpr int HALF_DIM = 256;
    simd<float, HALF_DIM> fcos, fsin;
    load_rope_cos_sin<HALF_DIM>(ropeCosSinCache, rowOffset, fcos, fsin);

    simd<float, HALF_DIM> t0 = x0;
    simd<float, HALF_DIM> t1 = x1;
    x0 = t0 * fcos - t1 * fsin;
    x1 = t1 * fcos + t0 * fsin;
}

template <int HEAD_DIM>
ESIMD_INLINE float rms_scale(simd<float, HEAD_DIM>& x, float eps) {
    simd<float, HEAD_DIM> sq = x * x;
    float acc = sycl::ext::intel::esimd::detail::sum<float, float, HEAD_DIM>(sq) / (float)HEAD_DIM;
    return __ESIMD_NS::rsqrt(acc + eps);
}

ESIMD_INLINE float rms_scale_512(simd<float, 256>& x0, simd<float, 256>& x1, float eps) {
    simd<float, 256> sq0 = x0 * x0;
    simd<float, 256> sq1 = x1 * x1;
    float acc0 = sycl::ext::intel::esimd::detail::sum<float, float, 256>(sq0);
    float acc1 = sycl::ext::intel::esimd::detail::sum<float, float, 256>(sq1);
    return __ESIMD_NS::rsqrt((acc0 + acc1) / 512.0f + eps);
}

template <int HEAD_DIM>
ESIMD_INLINE void q_kernel_impl(
    uint8_t* qkvStateQ,
    uint8_t* qState,
    uint8_t* normWq,
    uint32_t* ropePos,
    fp16* ropeCosSinCache,
    uint32_t hiddenDim,
    uint32_t qHead,
    uint32_t rotaryDim,
    sycl::nd_item<2>& ndi) {

    constexpr float eps = 1e-6f;
    uint32_t headIdx = ndi.get_group(0);
    uint32_t tokIdx = ndi.get_group(1);

    if (rotaryDim != (uint32_t)HEAD_DIM) {
        return;
    }

    uint32_t inputOffset = tokIdx * hiddenDim + headIdx * HEAD_DIM;
    uint32_t outputOffset = qHead * HEAD_DIM * tokIdx + headIdx * HEAD_DIM;
    uint32_t rowOffset = ropePos[tokIdx] * rotaryDim;

    simd<fp16, HEAD_DIM> activation = block_load<fp16, HEAD_DIM>((fp16*)qkvStateQ + inputOffset);
    simd<float, HEAD_DIM> out = activation;

    simd<fp16, HEAD_DIM> wq = block_load<fp16, HEAD_DIM>((fp16*)normWq);
    float scale = rms_scale<HEAD_DIM>(out, eps);
    out = out * simd<float, HEAD_DIM>(wq) * scale;

    apply_rope_inplace<HEAD_DIM>(out, ropeCosSinCache, rowOffset);

    block_store<fp16, HEAD_DIM>((fp16*)qState + outputOffset, out);
}

template <int HEAD_DIM, bool KV_SHARED>
ESIMD_INLINE void k_kernel_impl(
    uint8_t* qkvStateK,
    uint8_t* kState,
    uint8_t* normWk,
    uint32_t* ropePos,
    fp16* ropeCosSinCache,
    uint32_t hiddenDim,
    uint32_t kvHead,
    uint32_t rotaryDim,
    sycl::nd_item<2>& ndi) {

    constexpr float eps = 1e-6f;
    uint32_t headIdx = ndi.get_group(0);
    uint32_t tokIdx = ndi.get_group(1);

    if (rotaryDim != (uint32_t)HEAD_DIM) {
        return;
    }

    uint32_t inputOffset = tokIdx * hiddenDim + headIdx * HEAD_DIM;
    uint32_t outputOffset = kvHead * HEAD_DIM * tokIdx + headIdx * HEAD_DIM;
    uint32_t rowOffset = ropePos[tokIdx] * rotaryDim;

    simd<fp16, HEAD_DIM> activation = block_load<fp16, HEAD_DIM>((fp16*)qkvStateK + inputOffset);
    simd<float, HEAD_DIM> out = activation;

    if constexpr (!KV_SHARED) {
        simd<fp16, HEAD_DIM> wk = block_load<fp16, HEAD_DIM>((fp16*)normWk);
        float scale = rms_scale<HEAD_DIM>(out, eps);
        out = out * simd<float, HEAD_DIM>(wk) * scale;
    }

    apply_rope_inplace<HEAD_DIM>(out, ropeCosSinCache, rowOffset);
    block_store<fp16, HEAD_DIM>((fp16*)kState + outputOffset, out);
}

template <int HEAD_DIM, bool KV_SHARED>
ESIMD_INLINE void v_kernel_impl(
    uint8_t* qkvStateV,
    uint8_t* vState,
    uint32_t hiddenDim,
    uint32_t kvHead,
    sycl::nd_item<2>& ndi) {

    constexpr float eps = 1e-6f;
    uint32_t headIdx = ndi.get_group(0);
    uint32_t tokIdx = ndi.get_group(1);

    uint32_t inputOffset = tokIdx * hiddenDim + headIdx * HEAD_DIM;
    uint32_t outputOffset = kvHead * HEAD_DIM * tokIdx + headIdx * HEAD_DIM;

    simd<fp16, HEAD_DIM> activation = block_load<fp16, HEAD_DIM>((fp16*)qkvStateV + inputOffset);

    if constexpr (KV_SHARED) {
        block_store<fp16, HEAD_DIM>((fp16*)vState + outputOffset, activation);
    } else {
        simd<float, HEAD_DIM> out = activation;
        float scale = rms_scale<HEAD_DIM>(out, eps);
        out = out * scale;
        block_store<fp16, HEAD_DIM>((fp16*)vState + outputOffset, out);
    }
}

template <bool KV_SHARED>
ESIMD_INLINE void q_kernel_impl_512(
    uint8_t* qkvStateQ,
    uint8_t* qState,
    uint8_t* normWq,
    uint32_t* ropePos,
    fp16* ropeCosSinCache,
    uint32_t hiddenDim,
    uint32_t qHead,
    uint32_t rotaryDim,
    sycl::nd_item<2>& ndi) {

    constexpr float eps = 1e-6f;
    constexpr int HEAD_DIM = 512;
    constexpr int CHUNK = 256;

    uint32_t headIdx = ndi.get_group(0);
    uint32_t tokIdx = ndi.get_group(1);

    if (rotaryDim != (uint32_t)HEAD_DIM) {
        return;
    }

    uint32_t inputOffset = tokIdx * hiddenDim + headIdx * HEAD_DIM;
    uint32_t outputOffset = qHead * HEAD_DIM * tokIdx + headIdx * HEAD_DIM;
    uint32_t rowOffset = ropePos[tokIdx] * rotaryDim;

    simd<fp16, CHUNK> a0 = block_load<fp16, CHUNK>((fp16*)qkvStateQ + inputOffset);
    simd<fp16, CHUNK> a1 = block_load<fp16, CHUNK>((fp16*)qkvStateQ + inputOffset + CHUNK);

    simd<float, CHUNK> out0 = a0;
    simd<float, CHUNK> out1 = a1;

    simd<fp16, CHUNK> wq0 = block_load<fp16, CHUNK>((fp16*)normWq);
    simd<fp16, CHUNK> wq1 = block_load<fp16, CHUNK>((fp16*)normWq + CHUNK);

    float scale = rms_scale_512(out0, out1, eps);
    out0 = out0 * simd<float, CHUNK>(wq0) * scale;
    out1 = out1 * simd<float, CHUNK>(wq1) * scale;

    apply_rope_inplace_512(out0, out1, ropeCosSinCache, rowOffset);

    block_store<fp16, CHUNK>((fp16*)qState + outputOffset, out0);
    block_store<fp16, CHUNK>((fp16*)qState + outputOffset + CHUNK, out1);
}

template <bool KV_SHARED>
ESIMD_INLINE void k_kernel_impl_512(
    uint8_t* qkvStateK,
    uint8_t* kState,
    uint8_t* normWk,
    uint32_t* ropePos,
    fp16* ropeCosSinCache,
    uint32_t hiddenDim,
    uint32_t kvHead,
    uint32_t rotaryDim,
    sycl::nd_item<2>& ndi) {

    constexpr float eps = 1e-6f;
    constexpr int HEAD_DIM = 512;
    constexpr int CHUNK = 256;

    uint32_t headIdx = ndi.get_group(0);
    uint32_t tokIdx = ndi.get_group(1);

    if (rotaryDim != (uint32_t)HEAD_DIM) {
        return;
    }

    uint32_t inputOffset = tokIdx * hiddenDim + headIdx * HEAD_DIM;
    uint32_t outputOffset = kvHead * HEAD_DIM * tokIdx + headIdx * HEAD_DIM;
    uint32_t rowOffset = ropePos[tokIdx] * rotaryDim;

    simd<fp16, CHUNK> a0 = block_load<fp16, CHUNK>((fp16*)qkvStateK + inputOffset);
    simd<fp16, CHUNK> a1 = block_load<fp16, CHUNK>((fp16*)qkvStateK + inputOffset + CHUNK);

    simd<float, CHUNK> out0 = a0;
    simd<float, CHUNK> out1 = a1;

    if constexpr (!KV_SHARED) {
        simd<fp16, CHUNK> wk0 = block_load<fp16, CHUNK>((fp16*)normWk);
        simd<fp16, CHUNK> wk1 = block_load<fp16, CHUNK>((fp16*)normWk + CHUNK);
        float scale = rms_scale_512(out0, out1, eps);
        out0 = out0 * simd<float, CHUNK>(wk0) * scale;
        out1 = out1 * simd<float, CHUNK>(wk1) * scale;
    }

    apply_rope_inplace_512(out0, out1, ropeCosSinCache, rowOffset);

    block_store<fp16, CHUNK>((fp16*)kState + outputOffset, out0);
    block_store<fp16, CHUNK>((fp16*)kState + outputOffset + CHUNK, out1);
}

template <bool KV_SHARED>
ESIMD_INLINE void v_kernel_impl_512(
    uint8_t* qkvStateV,
    uint8_t* vState,
    uint32_t hiddenDim,
    uint32_t kvHead,
    sycl::nd_item<2>& ndi) {

    constexpr float eps = 1e-6f;
    constexpr int HEAD_DIM = 512;
    constexpr int CHUNK = 256;

    uint32_t headIdx = ndi.get_group(0);
    uint32_t tokIdx = ndi.get_group(1);

    uint32_t inputOffset = tokIdx * hiddenDim + headIdx * HEAD_DIM;
    uint32_t outputOffset = kvHead * HEAD_DIM * tokIdx + headIdx * HEAD_DIM;

    simd<fp16, CHUNK> a0 = block_load<fp16, CHUNK>((fp16*)qkvStateV + inputOffset);
    simd<fp16, CHUNK> a1 = block_load<fp16, CHUNK>((fp16*)qkvStateV + inputOffset + CHUNK);

    if constexpr (KV_SHARED) {
        block_store<fp16, CHUNK>((fp16*)vState + outputOffset, a0);
        block_store<fp16, CHUNK>((fp16*)vState + outputOffset + CHUNK, a1);
    } else {
        simd<float, CHUNK> out0 = a0;
        simd<float, CHUNK> out1 = a1;
        float scale = rms_scale_512(out0, out1, eps);
        out0 = out0 * scale;
        out1 = out1 * scale;
        block_store<fp16, CHUNK>((fp16*)vState + outputOffset, out0);
        block_store<fp16, CHUNK>((fp16*)vState + outputOffset + CHUNK, out1);
    }
}

// Host dispatcher: submits Q/K/V kernels to the SYCL queue.
inline void qkv_split_norm_rope_gemma_host(
    uint8_t* qkvState,
    uint8_t* qState,
    uint8_t* kState,
    uint8_t* vState,
    uint8_t* normWq,
    uint8_t* normWk,
    uint32_t* ropePos,
    fp16* ropeCosSinCache,
    uint32_t ntoks,
    uint32_t hiddenDim,
    uint32_t headDim,
    uint32_t qHead,
    uint32_t kvHead,
    bool isKvSharedLayer,
    uint32_t rotaryDim,
    sycl::queue& q) {

    sycl::range<2> localRange(1, 1);

    uint8_t* qkvQ = qkvState;
    uint8_t* qkvK = qkvState + (size_t)qHead * headDim * sizeof(fp16);
    uint8_t* qkvV = qkvState + (size_t)(qHead + kvHead) * headDim * sizeof(fp16);

    auto submit_q = [&](auto qFunc) {
        if (qHead == 0) {
            return;
        }
        sycl::range<2> globalRange(qHead, ntoks);
        q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<2>(globalRange, localRange), qFunc);
        });
    };

    auto submit_k = [&](auto kFunc) {
        if (kvHead == 0) {
            return;
        }
        sycl::range<2> globalRange(kvHead, ntoks);
        q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<2>(globalRange, localRange), kFunc);
        });
    };

    auto submit_v = [&](auto vFunc) {
        if (kvHead == 0) {
            return;
        }
        sycl::range<2> globalRange(kvHead, ntoks);
        q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<2>(globalRange, localRange), vFunc);
        });
    };

    switch (headDim) {
        case 64:
            submit_q([=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL {
                q_kernel_impl<64>(qkvQ, qState, normWq, ropePos, ropeCosSinCache, hiddenDim, qHead, rotaryDim, ndi);
            });
            if (isKvSharedLayer) {
                submit_k([=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL {
                    k_kernel_impl<64, true>(qkvK, kState, normWk, ropePos, ropeCosSinCache, hiddenDim, kvHead, rotaryDim, ndi);
                });
                submit_v([=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL {
                    v_kernel_impl<64, true>(qkvV, vState, hiddenDim, kvHead, ndi);
                });
            } else {
                submit_k([=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL {
                    k_kernel_impl<64, false>(qkvK, kState, normWk, ropePos, ropeCosSinCache, hiddenDim, kvHead, rotaryDim, ndi);
                });
                submit_v([=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL {
                    v_kernel_impl<64, false>(qkvV, vState, hiddenDim, kvHead, ndi);
                });
            }
            break;
        case 128:
            submit_q([=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL {
                q_kernel_impl<128>(qkvQ, qState, normWq, ropePos, ropeCosSinCache, hiddenDim, qHead, rotaryDim, ndi);
            });
            if (isKvSharedLayer) {
                submit_k([=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL {
                    k_kernel_impl<128, true>(qkvK, kState, normWk, ropePos, ropeCosSinCache, hiddenDim, kvHead, rotaryDim, ndi);
                });
                submit_v([=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL {
                    v_kernel_impl<128, true>(qkvV, vState, hiddenDim, kvHead, ndi);
                });
            } else {
                submit_k([=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL {
                    k_kernel_impl<128, false>(qkvK, kState, normWk, ropePos, ropeCosSinCache, hiddenDim, kvHead, rotaryDim, ndi);
                });
                submit_v([=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL {
                    v_kernel_impl<128, false>(qkvV, vState, hiddenDim, kvHead, ndi);
                });
            }
            break;
        case 256:
            submit_q([=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL {
                q_kernel_impl<256>(qkvQ, qState, normWq, ropePos, ropeCosSinCache, hiddenDim, qHead, rotaryDim, ndi);
            });
            if (isKvSharedLayer) {
                submit_k([=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL {
                    k_kernel_impl<256, true>(qkvK, kState, normWk, ropePos, ropeCosSinCache, hiddenDim, kvHead, rotaryDim, ndi);
                });
                submit_v([=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL {
                    v_kernel_impl<256, true>(qkvV, vState, hiddenDim, kvHead, ndi);
                });
            } else {
                submit_k([=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL {
                    k_kernel_impl<256, false>(qkvK, kState, normWk, ropePos, ropeCosSinCache, hiddenDim, kvHead, rotaryDim, ndi);
                });
                submit_v([=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL {
                    v_kernel_impl<256, false>(qkvV, vState, hiddenDim, kvHead, ndi);
                });
            }
            break;
        case 512:
            submit_q([=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL {
                q_kernel_impl_512<false>(qkvQ, qState, normWq, ropePos, ropeCosSinCache, hiddenDim, qHead, rotaryDim, ndi);
            });
            if (isKvSharedLayer) {
                submit_k([=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL {
                    k_kernel_impl_512<true>(qkvK, kState, normWk, ropePos, ropeCosSinCache, hiddenDim, kvHead, rotaryDim, ndi);
                });
                submit_v([=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL {
                    v_kernel_impl_512<true>(qkvV, vState, hiddenDim, kvHead, ndi);
                });
            } else {
                submit_k([=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL {
                    k_kernel_impl_512<false>(qkvK, kState, normWk, ropePos, ropeCosSinCache, hiddenDim, kvHead, rotaryDim, ndi);
                });
                submit_v([=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL {
                    v_kernel_impl_512<false>(qkvV, vState, hiddenDim, kvHead, ndi);
                });
            }
            break;
        default:
            return;
    }
}
