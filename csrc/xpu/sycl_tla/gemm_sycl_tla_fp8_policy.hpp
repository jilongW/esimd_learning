#pragma once

#include "cute/atom/mma_atom.hpp"
#include "cutlass/numeric_types.h"

namespace XeGemmFp8 {
using namespace cute;

// Policy set for SYCL-TLA fp8 weight GEMM dispatch.
// A=FP16 input, B=FP8 weight [N,K] ColumnMajor (K-contiguous)
// Uses MainloopXeL1Staged with auto-select copy atoms (void).
// block_2d_selector handles FP8->FP16 resize, reorder() handles type conversion.
class sycl_tla_fp8_policy_base {
 public:
  using WGTile = Shape<_256, _256, _32>;
  using SGLayout = Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>;
};

class sycl_tla_fp8_policy_default : public sycl_tla_fp8_policy_base {};

class sycl_tla_fp8_policy_n_128 : public sycl_tla_fp8_policy_base {
 public:
  using WGTile = Shape<_256, _128, _32>;
  using SGLayout = Layout<Shape<_8, _2, _1>, Stride<_2, _1, _0>>;
};

class sycl_tla_fp8_policy_n_64 : public sycl_tla_fp8_policy_base {
 public:
  using WGTile = Shape<_256, _64, _32>;
  using SGLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _0>>;
};

class sycl_tla_fp8_policy_m_16_large_n : public sycl_tla_fp8_policy_base {
 public:
  using WGTile = Shape<_8, _512, _128>;
  using SGLayout = Layout<Shape<_1, _32, _1>, Stride<_32, _1, _0>>;
};

class sycl_tla_fp8_policy_m_16_small_n : public sycl_tla_fp8_policy_base {
 public:
  using WGTile = Shape<_8, _128, _64>;
  using SGLayout = Layout<Shape<_1, _8, _1>, Stride<_8, _1, _0>>;
};

class sycl_tla_fp8_policy_m_32_small_n : public sycl_tla_fp8_policy_base {
 public:
  using WGTile = Shape<_32, _64, _32>;
  using SGLayout = Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>;
};
class sycl_tla_fp8_policy_m_32_large_n : public sycl_tla_fp8_policy_base {
 public:
  using WGTile = Shape<_32, _256, _32>;
  using SGLayout = Layout<Shape<_4, _8, _1>, Stride<_8, _1, _0>>;
};

}  // namespace XeGemmFp8
