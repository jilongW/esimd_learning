#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/all.h>
#include <torch/library.h>
#include <Python.h>

#include "kernel_ops.h"

TORCH_LIBRARY(custom_esimd_kernels_vllm, m) {
  m.def("esimd_gemv_fp8_pert(Tensor input, Tensor weight, Tensor weight_scale, "
        "Tensor output) -> Tensor");
  m.impl("esimd_gemv_fp8_pert", torch::kXPU, &esimd_gemv_fp8_pert);
}

PyMODINIT_FUNC PyInit_custom_esimd_kernels() {
    static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "custom_esimd_kernels", nullptr, 0, nullptr};
    return PyModule_Create(&module);
}
