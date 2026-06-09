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

      m.def("esimd_rms_norm_res(Tensor hidden_states, Tensor res, Tensor weight, float eps, Tensor output, int vl, int ks) -> Tensor");
      m.impl("esimd_rms_norm_res", torch::kXPU, &esimd_rms_norm_res);

      m.def("esimd_rms_norm_res_scale(Tensor hidden_states, Tensor res, Tensor weight, Tensor scale, float eps, Tensor output, int vl, int ks) -> Tensor");
      m.impl("esimd_rms_norm_res_scale", torch::kXPU, &esimd_rms_norm_res_scale);

      m.def("esimd_norm_gemv_fp8_pert(Tensor hidden_states, Tensor norm_weight, "
            "Tensor gemv_weight, Tensor gemv_scale, Tensor output, float eps, int vl, int ks) -> Tensor");
      m.impl("esimd_norm_gemv_fp8_pert", torch::kXPU, &esimd_norm_gemv_fp8_pert);

      m.def("esimd_qkv_split_norm_rope_gemma(Tensor qkv_state, Tensor q_out, "
            "Tensor k_out, Tensor v_out, Tensor norm_wq, Tensor norm_wk, Tensor positions, int q_heads, int kv_heads, int rotary_dim, bool isKvSharedLayer, Tensor cos_sin_cache) -> Tensor");
      m.impl("esimd_qkv_split_norm_rope_gemma", torch::kXPU, &esimd_qkv_split_norm_rope_gemma);

      m.def("esimd_gemv_gelu_tanh_mul_fp8_pert(Tensor hidden_states, "
            "Tensor gemv_weight, Tensor gemv_scale, Tensor output, int vl, int ks) -> Tensor");
      m.impl("esimd_gemv_gelu_tanh_mul_fp8_pert", torch::kXPU,
            static_cast<at::Tensor (*)(at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t)>(
                &esimd_gemv_gelu_tanh_mul_fp8_pert));

      m.def("esimd_gemv_gelu_tanh_mul_y_fp8_pert(Tensor hidden_states, "
            "Tensor gemv_weight, Tensor gemv_scale, Tensor per_layer_input, Tensor output, int vl, int ks) -> Tensor");
      m.impl("esimd_gemv_gelu_tanh_mul_y_fp8_pert", torch::kXPU,
            static_cast<at::Tensor (*)(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t)>(
                &esimd_gemv_gelu_tanh_mul_y_fp8_pert));

      m.def("esimd_norm_gemv2_geglu_fp8_pert(Tensor hidden_states, Tensor norm_weight, "
            "Tensor gemv_weight0, Tensor gemv_scale0, Tensor gemv_weight1, Tensor gemv_scale1, Tensor output, float eps, int vl, int ks) -> Tensor");
      m.impl("esimd_norm_gemv2_geglu_fp8_pert", torch::kXPU, &esimd_norm_gemv2_geglu_fp8_pert);

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
