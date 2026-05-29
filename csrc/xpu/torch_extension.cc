#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/all.h>
#include <torch/library.h>
#include <Python.h>

#include "kernel_ops.h"

TORCH_LIBRARY(custom_esimd_kernels_vllm, m) {
  m.def("esimd_gemv_fp8_pern(Tensor input, Tensor weight, Tensor weight_scale, "
        "Tensor output, int N, int K, int vl, int ks) -> Tensor");
  m.impl("esimd_gemv_fp8_pern", torch::kXPU, &esimd_gemv_fp8_pern);

  m.def("esimd_gemv_fp8(Tensor input, Tensor weight, Tensor weight_scale, "
        "Tensor output) -> Tensor");
  m.impl("esimd_gemv_fp8", torch::kXPU, &esimd_gemv_fp8);

  m.def("esimd_gemv_fp8_pert(Tensor input, Tensor weight, Tensor weight_scale, "
        "Tensor output, int N, int K, int vl, int ks) -> Tensor");
  m.impl("esimd_gemv_fp8_pert", torch::kXPU, &esimd_gemv_fp8_pert);
  m.def("esimd_fused_add_rms_norm_batched(Tensor hidden_states, Tensor residual, "
        "Tensor weight, float eps) -> Tensor");
  m.impl("esimd_fused_add_rms_norm_batched", torch::kXPU, &esimd_fused_add_rms_norm_batched);

        m.def("esimd_rms_norm(Tensor hidden_states, Tensor weight, float eps, Tensor output, int vl, int ks) -> Tensor");
  m.impl("esimd_rms_norm", torch::kXPU, &esimd_rms_norm);

  m.def("esimd_norm_gemv_fp8_pert(Tensor hidden_states, Tensor norm_weight, "
        "Tensor gemv_weight, Tensor gemv_scale, Tensor output, float eps, int vl, int ks) -> Tensor");
  m.impl("esimd_norm_gemv_fp8_pert", torch::kXPU, &esimd_norm_gemv_fp8_pert);

  m.def("esimd_norm_gemv2_fp8_pert(Tensor hidden_states, Tensor norm_weight, "
        "Tensor gemv_weight0, Tensor gemv_scale0, Tensor gemv_weight1, Tensor gemv_scale1, float eps, int vl, int ks) -> (Tensor, Tensor)");
  m.impl("esimd_norm_gemv2_fp8_pert", torch::kXPU, &esimd_norm_gemv2_fp8_pert);

      m.def("esimd_gelu_tanh_and_mul(Tensor input, Tensor output, int vl, int ks) -> Tensor");
  m.impl("esimd_gelu_tanh_and_mul", torch::kXPU, &esimd_gelu_tanh_and_mul);

  m.def("esimd_resadd_norm_gemv_fp8_pert(Tensor hidden_states, Tensor residual, "
        "Tensor norm_weight, Tensor gemv_weight, Tensor gemv_scale, "
        "Tensor output, Tensor normed_out, float eps) -> Tensor");
  m.impl("esimd_resadd_norm_gemv_fp8_pert", torch::kXPU, &esimd_resadd_norm_gemv_fp8_pert);
}

PyMODINIT_FUNC PyInit_custom_esimd_kernels() {
    static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "custom_esimd_kernels", nullptr, 0, nullptr};
    return PyModule_Create(&module);
}
