#pragma once

#include <torch/all.h>
#include <c10/xpu/XPUStream.h>

#include <cute/tensor.hpp>

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/collective/xe_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/util/packed_stride.hpp"
#include "gemm_sycl_tla_fp8_policy.hpp"

namespace XeGemmFp8 {
using namespace cute;

template <typename Policy, typename ElementInputB>
inline cutlass::Status run_cutlass_gemm_fp8_with_policy(
    at::Tensor& ptr_A,
    at::Tensor& ptr_B,
    at::Tensor& ptr_D,
    int m,
    int n,
    int k,
    sycl::queue& q) {
  using ElementAccumulator = float;
  using ElementComputeEpilogue = float;
  using ElementInputA = half_t;
  // ElementInputB is cutlass::float_e4m3_t or cutlass::float_e5m2_t
  using ElementOutput = half_t;

  // A=[M,K] RowMajor, B=[N,K] ColumnMajor (K-contiguous)
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  // Auto-select copy atoms: block_2d_selector handles FP8->FP16 resize
  using GmemTiledCopyA = void;
  using GmemTiledCopyB = void;

  constexpr int PipelineStages = 2;
  using GEMMDispatchPolicy = cutlass::gemm::MainloopXeL1Staged<PipelineStages>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeGeneric;

  using TileShape = typename Policy::WGTile;
  using TiledMma = typename TiledMMAHelper<
      MMA_Atom<XE_DPAS_TT<8, float, cute::half_t>>,
      Layout<TileShape>,
      typename Policy::SGLayout>::TiledMMA;

  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementOutput,
      ElementComputeEpilogue,
      ElementAccumulator,
      ElementAccumulator,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using FusionCallbacks = cutlass::epilogue::fusion::FusionCallbacks<
      EpilogueDispatchPolicy,
      EpilogueOp,
      TileShape,
      decltype(tile_shape(TiledMma()))>;

  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
      EpilogueDispatchPolicy,
      TileShape,
      void,
      ElementOutput,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      ElementOutput,
      cutlass::gemm::TagToStrideC_t<LayoutD>,
      FusionCallbacks,
      void,
      void>;

  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
      GEMMDispatchPolicy,
      TileShape,
      ElementInputA,
      cutlass::gemm::TagToStrideA_t<LayoutA>,
      ElementInputB,
      cutlass::gemm::TagToStrideB_t<LayoutB>,
      TiledMma,
      GmemTiledCopyA, void, void, cute::identity,
      GmemTiledCopyB, void, void, cute::identity>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
  using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;

  ProblemShapeType problem_size{m, n, k, 1};

  auto stride_A =
      cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
  auto stride_B =
      cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
  auto stride_C =
      cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(m, n, 1));
  auto stride_D =
      cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(m, n, 1));

  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
          hw_info.device_id);

  typename Gemm::GemmKernel::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size,
      {reinterpret_cast<ElementInputA*>(ptr_A.data_ptr()),
       stride_A,
       reinterpret_cast<ElementInputB*>(ptr_B.data_ptr()),
       stride_B},
      {{ElementComputeEpilogue(1.f), ElementComputeEpilogue(0.f)},
       reinterpret_cast<half_t*>(ptr_D.data_ptr()),
       stride_C,
       reinterpret_cast<half_t*>(ptr_D.data_ptr()),
       stride_D},
      hw_info};

  Gemm gemm_op;

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  if (workspace_size != 0) {
    return cutlass::Status::kErrorInternal;
  }

  if (gemm_op.can_implement(arguments) != cutlass::Status::kSuccess) {
    return cutlass::Status::kErrorInvalidProblem;
  }

  auto st = gemm_op.initialize(arguments, nullptr, &q);
  if (st != cutlass::Status::kSuccess) {
    return st;
  }

  st = gemm_op.run(&q);
  return st;
}

template <typename ElementInputB>
inline cutlass::Status dispatch_cutlass_gemm_sycl_tla_fp8(
    at::Tensor& ptr_A,
    at::Tensor& ptr_B,
    at::Tensor& ptr_D,
    int m,
    int n,
    int k,
    sycl::queue& q) {
  // Use fp8-specific policy choices for GEMM shape dispatch
  if (m <= 16 && n <= 15360) {
    return run_cutlass_gemm_fp8_with_policy<sycl_tla_fp8_policy_m_16_small_n, ElementInputB>(
        ptr_A, ptr_B, ptr_D, m, n, k, q);
  }
  else if (m <= 16 && n > 15360) {
    return run_cutlass_gemm_fp8_with_policy<sycl_tla_fp8_policy_m_16_large_n, ElementInputB>(
        ptr_A, ptr_B, ptr_D, m, n, k, q);
  }
  else if (m <= 32 && n <= 15360) {
    return run_cutlass_gemm_fp8_with_policy<sycl_tla_fp8_policy_m_32_small_n, ElementInputB>(
        ptr_A, ptr_B, ptr_D, m, n, k, q);
  }
  else if (m <= 32 && n > 15360) {
    return run_cutlass_gemm_fp8_with_policy<sycl_tla_fp8_policy_m_32_large_n, ElementInputB>(
        ptr_A, ptr_B, ptr_D, m, n, k, q);
  }
  else if (n <= 64) {
    return run_cutlass_gemm_fp8_with_policy<sycl_tla_fp8_policy_n_64, ElementInputB>(
        ptr_A, ptr_B, ptr_D, m, n, k, q);
  }
  else if (n <= 128) {
    return run_cutlass_gemm_fp8_with_policy<sycl_tla_fp8_policy_n_128, ElementInputB>(
        ptr_A, ptr_B, ptr_D, m, n, k, q);
  }
  else{
    return run_cutlass_gemm_fp8_with_policy<sycl_tla_fp8_policy_default, ElementInputB>(
        ptr_A, ptr_B, ptr_D, m, n, k, q);
  }
  
}

inline torch::Tensor cutlass_gemm_sycl_tla_fp8_impl(
    at::Tensor& ptr_A,
    at::Tensor& ptr_B,
    const c10::optional<at::Tensor>& ptr_bias,
    at::Tensor& ptr_D,
    int64_t N,
    int64_t K) {
  TORCH_CHECK(!ptr_bias.has_value(), "cutlass_gemm_sycl_tla_fp8 currently does not support bias");

  TORCH_CHECK(ptr_A.dim() == 2, "ptr_A must be 2D [M, K]");
  TORCH_CHECK(ptr_B.dim() == 2, "ptr_B must be 2D [N, K] ColumnMajor (K-contiguous)");
  TORCH_CHECK(ptr_D.dim() == 2, "ptr_D must be 2D [M, N]");

  TORCH_CHECK(ptr_A.is_contiguous(), "ptr_A must be contiguous");
  TORCH_CHECK(ptr_B.is_contiguous(), "ptr_B must be contiguous");
  TORCH_CHECK(ptr_D.is_contiguous(), "ptr_D must be contiguous");

  TORCH_CHECK(ptr_A.dtype() == at::kHalf, "ptr_A must be fp16");
  TORCH_CHECK(ptr_B.dtype() == at::kFloat8_e4m3fn || ptr_B.dtype() == at::kFloat8_e5m2,
              "ptr_B must be fp8 (e4m3 or e5m2)");
  TORCH_CHECK(ptr_D.dtype() == at::kHalf, "ptr_D must be fp16");

  int m = static_cast<int>(ptr_A.size(0));
  int k_a = static_cast<int>(ptr_A.size(1));
  int n_b = static_cast<int>(ptr_B.size(0));
  int k_b = static_cast<int>(ptr_B.size(1));

  TORCH_CHECK(k_a == static_cast<int>(K), "ptr_A.size(1) must match K");
  TORCH_CHECK(k_b == static_cast<int>(K), "ptr_B.size(1) must match K");
  TORCH_CHECK(n_b == static_cast<int>(N), "ptr_B.size(0) must match N");
  TORCH_CHECK(static_cast<int>(ptr_D.size(0)) == m, "ptr_D.size(0) must match M");
  TORCH_CHECK(static_cast<int>(ptr_D.size(1)) == n_b, "ptr_D.size(1) must match N");

  TORCH_CHECK(n_b >= 16, "N must be >= 16 for FP8 input with F16 MMA");

  auto& q = c10::xpu::getCurrentXPUStream(ptr_A.device().index()).queue();

  cutlass::Status st;
  if (ptr_B.dtype() == at::kFloat8_e4m3fn) {
    st = dispatch_cutlass_gemm_sycl_tla_fp8<cutlass::float_e4m3_t>(
        ptr_A, ptr_B, ptr_D, m, static_cast<int>(N), static_cast<int>(K), q);
  } else {
    st = dispatch_cutlass_gemm_sycl_tla_fp8<cutlass::float_e5m2_t>(
        ptr_A, ptr_B, ptr_D, m, static_cast<int>(N), static_cast<int>(K), q);
  }

  TORCH_CHECK(st == cutlass::Status::kSuccess, "cutlass_gemm_sycl_tla_fp8: run failed");

  return ptr_D;
}

}  // namespace XeGemmFp8
