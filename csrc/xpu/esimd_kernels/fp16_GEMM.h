#pragma once

#include "utils.h"

// ============================================================================
// FP16 GEMM
// Input : [M, K] fp16
// Weight: [N, K] fp16
// Output: [M, N] fp16
// ============================================================================

inline void normalize_fp16_gemm_vl_ks(uint32_t K, int &vl, int &ks) {
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

inline void select_fp16_gemm_vl_ks_xe3(uint32_t N, uint32_t K, int &vl, int &ks) {
	(void)N;
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
	} else if (K >= 2560) {
		vl = 512;
		ks = 1;
	} else if (K >= 2048) {
		vl = 256;
		ks = 2;
	}

	normalize_fp16_gemm_vl_ks(K, vl, ks);
}

inline void select_fp16_gemm_vl_ks_xe2(uint32_t N, uint32_t K, int &vl, int &ks) {
	select_fp16_gemm_vl_ks_xe3(N, K, vl, ks);
}

inline void select_fp16_gemm_vl_ks(
	uint32_t N,
	uint32_t K,
	int &vl,
	int &ks,
	const sycl::device *dev = nullptr) {
	if (dev != nullptr) {
		auto arch = dev->get_info<sycl::ext::oneapi::experimental::info::device::architecture>();
		if (is_ptl_architecture_device(arch)) {
			select_fp16_gemm_vl_ks_xe3(N, K, vl, ks);
		} else {
			select_fp16_gemm_vl_ks_xe2(N, K, vl, ks);
		}
		return;
	}
	select_fp16_gemm_vl_ks_xe2(N, K, vl, ks);
}

inline void select_fp16_gemm_vl_ks_m1(
	uint32_t N,
	uint32_t K,
	int &vl,
	int &ks,
	const sycl::device *dev = nullptr) {
	select_fp16_gemm_vl_ks(N, K, vl, ks, dev);
	ks = 1;
}

// ----------------------------------------------------------------------------
// Regime A (small M): batched GEMV-like split-K reduction
// nd_range<2>({N * K_SPLIT, M}, {K_SPLIT, 1})
// ----------------------------------------------------------------------------
template<int VL, int K_SPLIT>
struct GEMM_fp16_batched_kernel {
	const fp16 *input;   // [M, K]
	const fp16 *weight;  // [N, K]
	fp16 *output;        // [M, N]
	int M, N, K;

	void operator()(sycl::nd_item<2> item) const SYCL_ESIMD_KERNEL {
		if constexpr (K_SPLIT > 1) {
			slm_init<K_SPLIT * sizeof(float)>();
		}

		int n = item.get_group(0);
		int m = item.get_group(1);
		int lid = item.get_local_id(0);
		if (n >= N || m >= M) return;

		int kp = K / K_SPLIT;
		int ks = lid * kp;
		int vec_end = ks + (kp / VL) * VL;

		simd<float, VL> acc0 = 0.0f;
		simd<float, VL> acc1 = 0.0f;
		float tail = 0.0f;
		size_t in_base = (size_t)m * K;
		size_t w_base = (size_t)n * K;

		int k = ks;
		for (; k + 2 * VL <= vec_end; k += 2 * VL) {
			simd<fp16, VL> iv0 = block_load<fp16, VL>(input + in_base + k);
			simd<fp16, VL> wv0 = block_load<fp16, VL>(weight + w_base + k);
			acc0 += simd<float, VL>(iv0) * simd<float, VL>(wv0);

			simd<fp16, VL> iv1 = block_load<fp16, VL>(input + in_base + k + VL);
			simd<fp16, VL> wv1 = block_load<fp16, VL>(weight + w_base + k + VL);
			acc1 += simd<float, VL>(iv1) * simd<float, VL>(wv1);
		}

		for (; k < vec_end; k += VL) {
			simd<fp16, VL> iv = block_load<fp16, VL>(input + in_base + k);
			simd<fp16, VL> wv = block_load<fp16, VL>(weight + w_base + k);
			acc0 += simd<float, VL>(iv) * simd<float, VL>(wv);
		}

		for (int k = vec_end; k < ks + kp; ++k) {
			tail += static_cast<float>(input[in_base + k]) *
					static_cast<float>(weight[w_base + k]);
		}

		float my_sum = reduce<float>(acc0 + acc1, std::plus<>()) + tail;

		if constexpr (K_SPLIT == 1) {
			output[(size_t)m * N + n] = fp16(my_sum);
		} else {
			slm_block_store<float, 1>(lid * sizeof(float), simd<float, 1>(my_sum));
			barrier();
			if (lid == 0) {
				simd<float, K_SPLIT> parts = slm_block_load<float, K_SPLIT>(0);
				output[(size_t)m * N + n] = fp16(reduce<float>(parts, std::plus<>()));
			}
		}
	}
};

// ----------------------------------------------------------------------------
// Regime A0 (M=1): dedicated 1D launch
// nd_range<1>({N * K_SPLIT}, {K_SPLIT})
// ----------------------------------------------------------------------------
template<int VL, int K_SPLIT>
struct GEMM_fp16_m1_kernel {
	const fp16 *input;   // [K]
	const fp16 *weight;  // [N, K]
	fp16 *output;        // [N]
	int N, K;

	void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
		if constexpr (K_SPLIT > 1) {
			slm_init<K_SPLIT * sizeof(float)>();
		}

		int n = item.get_group(0);
		int lid = item.get_local_id(0);
		if (n >= N) return;

		int kp = K / K_SPLIT;
		int ks = lid * kp;
		int vec_end = ks + (kp / VL) * VL;

		simd<float, VL> acc0 = 0.0f;
		simd<float, VL> acc1 = 0.0f;
		float tail = 0.0f;
		size_t w_base = (size_t)n * K;

		int k = ks;
		for (; k + 2 * VL <= vec_end; k += 2 * VL) {
			simd<fp16, VL> iv0 = block_load<fp16, VL>(input + k);
			simd<fp16, VL> wv0 = block_load<fp16, VL>(weight + w_base + k);
			acc0 += simd<float, VL>(iv0) * simd<float, VL>(wv0);

			simd<fp16, VL> iv1 = block_load<fp16, VL>(input + k + VL);
			simd<fp16, VL> wv1 = block_load<fp16, VL>(weight + w_base + k + VL);
			acc1 += simd<float, VL>(iv1) * simd<float, VL>(wv1);
		}

		for (; k < vec_end; k += VL) {
			simd<fp16, VL> iv = block_load<fp16, VL>(input + k);
			simd<fp16, VL> wv = block_load<fp16, VL>(weight + w_base + k);
			acc0 += simd<float, VL>(iv) * simd<float, VL>(wv);
		}

		for (int k_tail = vec_end; k_tail < ks + kp; ++k_tail) {
			tail += static_cast<float>(input[k_tail]) *
					static_cast<float>(weight[w_base + k_tail]);
		}

		float my_sum = reduce<float>(acc0 + acc1, std::plus<>()) + tail;

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

// ----------------------------------------------------------------------------
// Regime B/C (general M): weight-stationary
// nd_range<2>({N, ceil(M / TILE_M)}, {1, 1})
// ----------------------------------------------------------------------------
template<int VL, int TILE_M>
struct GEMM_fp16_ws_kernel {
	const fp16 *input;   // [M, K]
	const fp16 *weight;  // [N, K]
	fp16 *output;        // [M, N]
	int M, N, K;

	void operator()(sycl::nd_item<2> item) const SYCL_ESIMD_KERNEL {
		int n = item.get_group(0);
		int m_tile = item.get_group(1);
		int m_start = m_tile * TILE_M;
		if (n >= N) return;

		simd<float, VL> acc[TILE_M];
		float tail[TILE_M];

		#pragma unroll
		for (int i = 0; i < TILE_M; i++) {
			acc[i] = 0.0f;
			tail[i] = 0.0f;
		}

		int vec_end = (K / VL) * VL;
		for (int k = 0; k < vec_end; k += VL) {
			simd<fp16, VL> wv = block_load<fp16, VL>(weight + (size_t)n * K + k);
			simd<float, VL> wf = wv;

			#pragma unroll
			for (int i = 0; i < TILE_M; i++) {
				if (m_start + i < M) {
					simd<fp16, VL> iv = block_load<fp16, VL>(
						input + (size_t)(m_start + i) * K + k);
					acc[i] += simd<float, VL>(iv) * wf;
				}
			}
		}

		for (int k = vec_end; k < K; ++k) {
			float wf = static_cast<float>(weight[(size_t)n * K + k]);
			#pragma unroll
			for (int i = 0; i < TILE_M; i++) {
				if (m_start + i < M) {
					tail[i] += static_cast<float>(input[(size_t)(m_start + i) * K + k]) * wf;
				}
			}
		}

		#pragma unroll
		for (int i = 0; i < TILE_M; i++) {
			if (m_start + i < M) {
				float sum = reduce<float>(acc[i], std::plus<>()) + tail[i];
				output[(size_t)(m_start + i) * N + n] = fp16(sum);
			}
		}
	}
};

template<int VL, int K_SPLIT>
inline void launch_fp16_batched(
	const fp16 *p_in,
	const fp16 *p_w,
	fp16 *p_out,
	int M,
	int N,
	int K,
	sycl::queue &q) {
	int global0 = N * K_SPLIT;
	int global1 = M;
	int local0 = K_SPLIT;
	int local1 = 1;

	q.submit([&](sycl::handler &h) {
		h.parallel_for(
			sycl::nd_range<2>({(size_t)global0, (size_t)global1}, {(size_t)local0, (size_t)local1}),
			GEMM_fp16_batched_kernel<VL, K_SPLIT>{p_in, p_w, p_out, M, N, K});
	});
}

template<int VL, int K_SPLIT>
inline void launch_fp16_m1(
	const fp16 *p_in,
	const fp16 *p_w,
	fp16 *p_out,
	int N,
	int K,
	sycl::queue &q) {
	int global = N * K_SPLIT;
	int local = K_SPLIT;

	q.submit([&](sycl::handler &h) {
		h.parallel_for(
			sycl::nd_range<1>((size_t)global, (size_t)local),
			GEMM_fp16_m1_kernel<VL, K_SPLIT>{p_in, p_w, p_out, N, K});
	});
}

template<int VL, int TILE_M>
inline void launch_fp16_ws(
	const fp16 *p_in,
	const fp16 *p_w,
	fp16 *p_out,
	int M,
	int N,
	int K,
	sycl::queue &q) {
	int groups_m = (M + TILE_M - 1) / TILE_M;
	q.submit([&](sycl::handler &h) {
		h.parallel_for(
			sycl::nd_range<2>({(size_t)N, (size_t)groups_m}, {(size_t)1, (size_t)1}),
			GEMM_fp16_ws_kernel<VL, TILE_M>{p_in, p_w, p_out, M, N, K});
	});
}

inline void GEMM_fp16_host_impl(
	uint8_t *input_data,
	uint8_t *weight_data,
	uint8_t *output_data,
	uint32_t M,
	uint32_t N,
	uint32_t K,
	int vl,
	int ks,
	sycl::queue &q) {

	auto *p_in = reinterpret_cast<const fp16 *>(input_data);
	auto *p_w = reinterpret_cast<const fp16 *>(weight_data);
	auto *p_out = reinterpret_cast<fp16 *>(output_data);

	auto launch_small_m = [&](int v, int s) {
		if (v == 512 && s == 1) launch_fp16_batched<512, 1>(p_in, p_w, p_out, (int)M, (int)N, (int)K, q);
		else if (v == 512 && s == 2) launch_fp16_batched<512, 2>(p_in, p_w, p_out, (int)M, (int)N, (int)K, q);
		else if (v == 256 && s == 1) launch_fp16_batched<256, 1>(p_in, p_w, p_out, (int)M, (int)N, (int)K, q);
		else if (v == 256 && s == 2) launch_fp16_batched<256, 2>(p_in, p_w, p_out, (int)M, (int)N, (int)K, q);
		else if (v == 256 && s == 4) launch_fp16_batched<256, 4>(p_in, p_w, p_out, (int)M, (int)N, (int)K, q);
		else if (v == 128 && s == 2) launch_fp16_batched<128, 2>(p_in, p_w, p_out, (int)M, (int)N, (int)K, q);
		else if (v == 128 && s == 4) launch_fp16_batched<128, 4>(p_in, p_w, p_out, (int)M, (int)N, (int)K, q);
		else launch_fp16_batched<128, 1>(p_in, p_w, p_out, (int)M, (int)N, (int)K, q);
	};

	auto launch_m1 = [&](int v, int s) {
		if (v == 512 && s == 1) launch_fp16_m1<512, 1>(p_in, p_w, p_out, (int)N, (int)K, q);
		else if (v == 512 && s == 2) launch_fp16_m1<512, 2>(p_in, p_w, p_out, (int)N, (int)K, q);
		else if (v == 256 && s == 1) launch_fp16_m1<256, 1>(p_in, p_w, p_out, (int)N, (int)K, q);
		else if (v == 256 && s == 2) launch_fp16_m1<256, 2>(p_in, p_w, p_out, (int)N, (int)K, q);
		else if (v == 256 && s == 4) launch_fp16_m1<256, 4>(p_in, p_w, p_out, (int)N, (int)K, q);
		else if (v == 128 && s == 2) launch_fp16_m1<128, 2>(p_in, p_w, p_out, (int)N, (int)K, q);
		else if (v == 128 && s == 4) launch_fp16_m1<128, 4>(p_in, p_w, p_out, (int)N, (int)K, q);
		else launch_fp16_m1<128, 1>(p_in, p_w, p_out, (int)N, (int)K, q);
	};

	auto launch_ws = [&](int v, int tile_m) {
		if (v == 512 && tile_m == 8) launch_fp16_ws<512, 8>(p_in, p_w, p_out, (int)M, (int)N, (int)K, q);
		else if (v == 256 && tile_m == 8) launch_fp16_ws<256, 8>(p_in, p_w, p_out, (int)M, (int)N, (int)K, q);
		else if (v == 128 && tile_m == 8) launch_fp16_ws<128, 8>(p_in, p_w, p_out, (int)M, (int)N, (int)K, q);
		else if (v == 512 && tile_m == 16) launch_fp16_ws<512, 16>(p_in, p_w, p_out, (int)M, (int)N, (int)K, q);
		else if (v == 256 && tile_m == 16) launch_fp16_ws<256, 16>(p_in, p_w, p_out, (int)M, (int)N, (int)K, q);
		else launch_fp16_ws<128, 16>(p_in, p_w, p_out, (int)M, (int)N, (int)K, q);
	};

	if (M == 1) {
		launch_small_m(vl, ks);
		return;
	}

	if (M <= 4) {
		launch_small_m(vl, ks);
		return;
	}

	if (M <= 8) {
		launch_ws(vl, 8);
		return;
	}

	launch_ws(vl, 16);
}

inline void GEMM_fp16_host(
	uint8_t *input_data,
	uint8_t *weight_data,
	uint8_t *output_data,
	uint32_t M,
	uint32_t N,
	uint32_t K,
	sycl::queue &q,
	int vl = 0,
	int ks = 0) {
	if (vl <= 0 || ks <= 0) {
		const sycl::device dev = q.get_device();
		select_fp16_gemm_vl_ks(N, K, vl, ks, &dev);
	}

	GEMM_fp16_host_impl(
		input_data,
		weight_data,
		output_data,
		M,
		N,
		K,
		vl,
		ks,
		q);
}
