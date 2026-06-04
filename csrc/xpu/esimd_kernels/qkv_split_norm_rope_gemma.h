#pragma once
#include "utils.h"
#include <cmath>

// Fused QKV Split + RMSNorm + RoPE kernel
// Adapted from qkv.split/qkv.split.h for project conventions.
//
// Operations per head:
//   Q heads: RMSNorm(weight eps=1e-6) → RoPE(theta=10M)
//   K heads: shared=false => RMSNorm(weight) + RoPE, shared=true => copy only
//   V heads: shared=false => RMSNorm(no weight), shared=true => copy only
//
// Work decomposition: 2D dispatch (totalHeads, nTokens), one WG per (head, token)
//   totalHeads = qHead + 2*kvHead

#define QKV_LOCATION_Q 0
#define QKV_LOCATION_K 1
#define QKV_LOCATION_V 2

template <int HEAD_DIM>
ESIMD_INLINE void qkv_split_norm_rope_gemma_kernel_impl(
    uint8_t* qkvState,
    uint8_t* qState,
    uint8_t* kState,
    uint8_t* vState,
    uint8_t* normWq,
    uint8_t* normWk,
    uint32_t* ropePos,
    fp16* ropeCosSinCache,  // [max_pos, rotaryDim] fp16 — first half cos, second half sin
    uint32_t ntoks,
    uint32_t hiddenDim,
    uint32_t qHead,
    uint32_t kvHead,
    uint32_t rotaryDim,
    bool isKvSharedLayer,
    sycl::nd_item<2>& ndi) {

    constexpr float eps = 1e-6f;
    uint32_t rotaryHalf = rotaryDim / 2;

    int32_t headIdx = ndi.get_group(0);
    int32_t tokIdx  = ndi.get_group(1);

    uint32_t outHead = headIdx;
    uint32_t whereAmI = QKV_LOCATION_Q;

    if (rotaryDim != (uint32_t)HEAD_DIM) {
        return;
    }

    uint32_t inputOffset = tokIdx * hiddenDim + headIdx * HEAD_DIM;
    uint32_t outputOffset;

    uint32_t i32RopeCoord = ropePos[tokIdx];
    float fp32RopeCoord = (float)i32RopeCoord;

    simd<fp16, HEAD_DIM> activation;

    // Load one head from QKV buffer.
    activation = block_load<fp16, HEAD_DIM>((fp16*)qkvState + inputOffset);

    // Determine which output this head maps to (no gating)
    if ((uint32_t)headIdx < qHead) {
        whereAmI = QKV_LOCATION_Q;
        outHead = headIdx;
    } else if ((uint32_t)headIdx < (qHead + kvHead)) {
        whereAmI = QKV_LOCATION_K;
        outHead = headIdx - qHead;
    } else {
        whereAmI = QKV_LOCATION_V;
        outHead = headIdx - qHead - kvHead;
    }

    if (whereAmI == QKV_LOCATION_Q) {
        // RMSNorm + partial RoPE for Q
        simd<fp16, HEAD_DIM> fp16RmsWeights;
        simd<float, HEAD_DIM> fp32RmsWeights;
        simd<float, HEAD_DIM> outputTemp;

        outputOffset = qHead * HEAD_DIM * tokIdx + outHead * HEAD_DIM;
        outputTemp = activation;
        simd<float, HEAD_DIM> outputSq = outputTemp * outputTemp;

        // RMSNorm: x * (weight + 1.0) / rms
        fp16RmsWeights = block_load<fp16, HEAD_DIM>((fp16*)normWq);
        float acc = sycl::ext::intel::esimd::detail::sum<float, float, HEAD_DIM>(outputSq) / (float)HEAD_DIM;
        float scale = __ESIMD_NS::rsqrt(acc + eps);
        fp32RmsWeights = fp16RmsWeights + 1.0f;
        outputTemp = outputTemp * fp32RmsWeights;
        outputTemp.template select<HEAD_DIM, 1>(0) = outputTemp.template select<HEAD_DIM, 1>(0) * scale;

        // RoPE: read from rotary_emb.cos_sin_cache [max_pos, rotaryDim]
        // Layout: [cos(rotaryHalf), sin(rotaryHalf)] per row
        {
            // Row offset in fp16 elements: position * rotaryDim
            uint32_t row_offset = i32RopeCoord * rotaryDim;
            constexpr int HALF_DIM = HEAD_DIM / 2;
            constexpr int CHUNK32 = HALF_DIM / 32;
            simd<float, HALF_DIM> fcos, fsin;
#pragma unroll
            for (int kk = 0; kk < CHUNK32; kk++) {
                simd<fp16, 32> c16 = block_load<fp16, 32>(ropeCosSinCache + row_offset + 32 * kk);
                fcos.template select<32, 1>(32 * kk) = c16;
            }
#pragma unroll
            for (int kk = 0; kk < CHUNK32; kk++) {
                simd<fp16, 32> s16 = block_load<fp16, 32>(ropeCosSinCache + row_offset + HALF_DIM + 32 * kk);
                fsin.template select<32, 1>(32 * kk) = s16;
            }

            simd<float, HALF_DIM> x1 = outputTemp.template select<HALF_DIM, 1>(0);
            simd<float, HALF_DIM> x2 = outputTemp.template select<HALF_DIM, 1>(HALF_DIM);
            outputTemp.template select<HALF_DIM, 1>(0)        = x1 * fcos - x2 * fsin;
            outputTemp.template select<HALF_DIM, 1>(HALF_DIM) = x2 * fcos + x1 * fsin;
        }

        activation = outputTemp;
        block_store<fp16, HEAD_DIM>((fp16*)qState + outputOffset, activation);
    }
    else if (whereAmI == QKV_LOCATION_K) {
        outputOffset = kvHead * HEAD_DIM * tokIdx + outHead * HEAD_DIM;
        if (isKvSharedLayer) {
            // Shared KV layer: keep K unchanged.
            block_store<fp16, HEAD_DIM>((fp16*)kState + outputOffset, activation);
        } else {
            // Non-shared KV layer: RMSNorm + RoPE for K.
            simd<fp16, HEAD_DIM> fp16RmsWeights;
            simd<float, HEAD_DIM> fp32RmsWeights;
            simd<float, HEAD_DIM> outputTemp;

            outputTemp = activation;
            simd<float, HEAD_DIM> kOutputSq = outputTemp * outputTemp;

            fp16RmsWeights = block_load<fp16, HEAD_DIM>((fp16*)normWk);
            float acc = sycl::ext::intel::esimd::detail::sum<float, float, HEAD_DIM>(kOutputSq) / (float)HEAD_DIM;
            float scale = __ESIMD_NS::rsqrt(acc + eps);
            fp32RmsWeights = fp16RmsWeights + 1.0f;
            outputTemp = outputTemp * fp32RmsWeights;
            outputTemp.template select<HEAD_DIM, 1>(0) = outputTemp.template select<HEAD_DIM, 1>(0) * scale;

            // RoPE: read from cos_sin_cache (same as Q)
            {
                uint32_t krow_offset = i32RopeCoord * rotaryDim;
                constexpr int HALF_DIM = HEAD_DIM / 2;
                constexpr int CHUNK32 = HALF_DIM / 32;
                simd<float, HALF_DIM> kfcos, kfsin;
#pragma unroll
                for (int kk = 0; kk < CHUNK32; kk++) {
                    simd<fp16, 32> kc16 = block_load<fp16, 32>(ropeCosSinCache + krow_offset + 32 * kk);
                    kfcos.template select<32, 1>(32 * kk) = kc16;
                }
#pragma unroll
                for (int kk = 0; kk < CHUNK32; kk++) {
                    simd<fp16, 32> ks16 = block_load<fp16, 32>(ropeCosSinCache + krow_offset + HALF_DIM + 32 * kk);
                    kfsin.template select<32, 1>(32 * kk) = ks16;
                }

                simd<float, HALF_DIM> kx1 = outputTemp.template select<HALF_DIM, 1>(0);
                simd<float, HALF_DIM> kx2 = outputTemp.template select<HALF_DIM, 1>(HALF_DIM);
                outputTemp.template select<HALF_DIM, 1>(0)        = kx1 * kfcos - kx2 * kfsin;
                outputTemp.template select<HALF_DIM, 1>(HALF_DIM) = kx2 * kfcos + kx1 * kfsin;
            }

            activation = outputTemp;
            block_store<fp16, HEAD_DIM>((fp16*)kState + outputOffset, activation);
        }
    }
    else if (whereAmI == QKV_LOCATION_V) {
        outputOffset = kvHead * HEAD_DIM * tokIdx + outHead * HEAD_DIM;
        if (!isKvSharedLayer) {
            // Non-shared KV layer: RMSNorm for V (no weight).
            simd<float, HEAD_DIM> outputTemp;

            outputTemp = activation;
            simd<float, HEAD_DIM> vOutputSq = outputTemp * outputTemp;

            float acc = sycl::ext::intel::esimd::detail::sum<float, float, HEAD_DIM>(vOutputSq) / (float)HEAD_DIM;
            float scale = __ESIMD_NS::rsqrt(acc + eps);
            outputTemp.template select<HEAD_DIM, 1>(0) = outputTemp.template select<HEAD_DIM, 1>(0) * scale;

            activation = outputTemp;
        }

        // Shared KV layer keeps V unchanged; non-shared stores normalized V.
        block_store<fp16, HEAD_DIM>((fp16*)vState + outputOffset, activation);
    }
}

ESIMD_INLINE void qkv_split_norm_rope_gemma_kernel(
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
    uint32_t rotaryDim,
    bool isKvSharedLayer,
    sycl::nd_item<2>& ndi) {

    switch (headDim) {
        case 64:
            qkv_split_norm_rope_gemma_kernel_impl<64>(
                qkvState, qState, kState, vState,
                normWq, normWk, ropePos, ropeCosSinCache,
                ntoks, hiddenDim, qHead, kvHead, rotaryDim, isKvSharedLayer, ndi);
            break;
        case 128:
            qkv_split_norm_rope_gemma_kernel_impl<128>(
                qkvState, qState, kState, vState,
                normWq, normWk, ropePos, ropeCosSinCache,
                ntoks, hiddenDim, qHead, kvHead, rotaryDim, isKvSharedLayer, ndi);
            break;
        case 256:
            qkv_split_norm_rope_gemma_kernel_impl<256>(
                qkvState, qState, kState, vState,
                normWq, normWk, ropePos, ropeCosSinCache,
                ntoks, hiddenDim, qHead, kvHead, rotaryDim, isKvSharedLayer, ndi);
            break;
        default:
            return;
    }
}

// Host dispatcher: submits the 2D kernel to the SYCL queue
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
    uint32_t totalHeads = qHead + 2 * kvHead;

    sycl::range<2> globalRange(totalHeads, ntoks);
    sycl::range<2> localRange(1, 1);

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<2>(globalRange, localRange),
            [=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL {
                qkv_split_norm_rope_gemma_kernel(
                    qkvState, qState, kState, vState,
                    normWq, normWk, ropePos, ropeCosSinCache,
                    ntoks, hiddenDim, headDim, qHead, kvHead, rotaryDim, isKvSharedLayer, ndi);
            });
    });
}
