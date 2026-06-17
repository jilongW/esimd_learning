#pragma once

#include "cute/atom/mma_atom.hpp"
#include "cutlass/numeric_types.h"

namespace XeGemm {
using namespace cute;

// Policy set for SYCL-TLA fp16 GEMM dispatch.
class sycl_tla_policy_base {
 public:
  using WGTile = Shape<_256, _256, _32>;
  using SGLayout = Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>;

  // Copy atoms can be specialized later if needed.
  using GmemTiledCopyA = void;
  using GmemTiledCopyB = void;
  using GmemTiledCopyD = void;
};

class sycl_tla_policy_default : public sycl_tla_policy_base {};

class sycl_tla_policy_n_128 : public sycl_tla_policy_base {
 public:
  using WGTile = Shape<_256, _128, _32>;
  using SGLayout = Layout<Shape<_8, _2, _1>, Stride<_2, _1, _0>>;
};

class sycl_tla_policy_n_64 : public sycl_tla_policy_base {
 public:
  using WGTile = Shape<_256, _64, _32>;
  using SGLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _0>>;
};

class sycl_tla_policy_m_16 : public sycl_tla_policy_base {
 public:
  using WGTile = Shape<_16, _256, _32>;
  using SGLayout = Layout<Shape<_2, _16, _1>, Stride<_16, _1, _0>>;
};

class sycl_tla_policy_m_32 : public sycl_tla_policy_base {
 public:
  using WGTile = Shape<_32, _64, _32>;
  using SGLayout = Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>;
};

}  // namespace XeGemm
