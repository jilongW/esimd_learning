#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/all.h>
#include <torch/library.h>
#include <Python.h>

#include "kernel_ops.h"

TORCH_LIBRARY(custom_esimd_kernels_vllm, m) {
  m.def("esimd_gemv_fp8_pern(Tensor input, Tensor weight, Tensor weight_scale, "
        "Tensor output, int N, int K) -> Tensor");
  m.impl("esimd_gemv_fp8_pern", torch::kXPU, &esimd_gemv_fp8_pern);

  m.def("esimd_gemv_fp8_pert(Tensor input, Tensor weight, Tensor weight_scale, "
        "Tensor output) -> Tensor");
  m.impl("esimd_gemv_fp8_pert", torch::kXPU, &esimd_gemv_fp8_pert);
  m.def("esimd_fused_add_rms_norm_batched(Tensor hidden_states, Tensor residual, "
        "Tensor weight, float eps) -> Tensor");
  m.impl("esimd_fused_add_rms_norm_batched", torch::kXPU, &esimd_fused_add_rms_norm_batched);

        m.def("esimd_rms_norm(Tensor hidden_states, Tensor weight, float eps, Tensor output) -> Tensor");
  m.impl("esimd_rms_norm", torch::kXPU, &esimd_rms_norm);

  m.def("esimd_resadd_norm_gemv_fp8_pert(Tensor hidden_states, Tensor residual, "
        "Tensor norm_weight, Tensor gemv_weight, Tensor gemv_scale, "
        "Tensor output, Tensor normed_out, float eps) -> Tensor");
  m.impl("esimd_resadd_norm_gemv_fp8_pert", torch::kXPU, &esimd_resadd_norm_gemv_fp8_pert);
}

PyMODINIT_FUNC PyInit_custom_esimd_kernels() {
    static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "custom_esimd_kernels", nullptr, 0, nullptr};
    return PyModule_Create(&module);
}
