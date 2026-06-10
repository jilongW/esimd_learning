#pragma once

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/oneapi/experimental/device_architecture.hpp>

using namespace sycl::ext::intel::esimd;
using fp16 = sycl::half;
using bf16 = sycl::ext::oneapi::bfloat16;
using namespace sycl;

inline bool is_ptl_architecture_device(sycl::ext::oneapi::experimental::architecture arch) {
	using sycl::ext::oneapi::experimental::architecture;
	return arch == architecture::intel_gpu_ptl_h || arch == architecture::intel_gpu_ptl_u;
}
