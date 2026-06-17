import os
import sys
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import SyclExtension
from esimd_build_extention import BuildExtension

root = Path(__file__).parent.resolve()

import torch
torch_include = str(Path(torch.__file__).parent / "include")

# Allow external CUTLASS/SYCL-TLA source override (compatible with vllm-xpu-kernels style).
sycl_tla_root = Path(os.environ.get("VLLM_CUTLASS_SRC_DIR", "/home/edgeai/sycl-tla")).resolve()
sycl_tla_include = sycl_tla_root / "include"
sycl_tla_examples_common = sycl_tla_root / "examples" / "common"
sycl_tla_tools_util_include = sycl_tla_root / "tools" / "util" / "include"
sycl_tla_applications = sycl_tla_root / "applications"

required_sycl_tla_dirs = [
    sycl_tla_include,
    sycl_tla_examples_common,
    sycl_tla_tools_util_include,
    sycl_tla_applications,
]
missing_sycl_tla_dirs = [str(path) for path in required_sycl_tla_dirs if not path.exists()]
if missing_sycl_tla_dirs:
    raise RuntimeError(
        "Missing SYCL-TLA/CUTLASS include directories: "
        + ", ".join(missing_sycl_tla_dirs)
        + ". Set VLLM_CUTLASS_SRC_DIR to a valid sycl-tla/cutlass source tree."
    )

cutlass_common_defines = [
    "-DCUTLASS_ENABLE_HEADERS_ONLY",
    "-DCUTLASS_ENABLE_SYCL",
    "-DSYCL_INTEL_TARGET",
    "-DCUTLASS_VERSIONS_GENERATED",
]

ext_modules = [
    SyclExtension(
        name="custom_esimd_kernels_vllm.custom_esimd_kernels",
        sources=[
            "csrc/xpu/esimd_kernel.sycl",
            "csrc/xpu/torch_extension.cc",
        ],
        include_dirs=[
            root / "include",
            root / "csrc",
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "sycl": ["-ffast-math", "-fsycl-device-code-split=per_kernel",
                     f"-I{torch_include}"],
        },
        extra_link_args=["-Wl,-rpath,$ORIGIN/../../torch/lib"],
        py_limited_api=False,
    )
]


### CUTLASS GEMM extension
ext_modules.append(
    SyclExtension(
        name="custom_esimd_kernels_vllm.custom_esimd_kernels_cutlass_gemm",
        sources=[
            "csrc/xpu/torch_extension_cutlass_gemm.sycl",
        ],
        include_dirs=[
            root / "include",
            root / "csrc",
            sycl_tla_include,
            sycl_tla_examples_common,
            sycl_tla_tools_util_include,
            sycl_tla_applications,
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17", *cutlass_common_defines],
            "sycl": [
                "-fsycl",
                "-ffast-math",
                "-fsycl-device-code-split=per_kernel",
                "-fsycl-targets=spir64_gen",
                "-Xs", "-device ptl -options -doubleGRF",
                "-fno-sycl-instrument-device-code",
                *cutlass_common_defines,
                f"-I{torch_include}",
            ],
        },
        extra_link_args=["-Wl,-rpath,$ORIGIN/../../torch/lib"],
        py_limited_api=False,
    )
)

ext_modules.append(
    SyclExtension(
        name="custom_esimd_kernels_vllm.custom_esimd_kernels_gemm",
        sources=[
            "csrc/xpu/esimd_kernel_gemm.sycl",
            "csrc/xpu/torch_extension_gemm.cc",
        ],
        include_dirs=[
            root / "include",
            root / "csrc",
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "sycl": ["-fsycl", "-ffast-math", "-fsycl-device-code-split=per_kernel",
                     "-fsycl-targets=spir64_gen", "-Xs", "-device ptl -options -doubleGRF",
                     f"-I{torch_include}"],
        },
        extra_link_args=["-Wl,-rpath,$ORIGIN/../../torch/lib"],
        py_limited_api=False,
    )
)

### FP8 GEMM kernels

setup(
    name="custom-esimd-kernels-vllm",
    version="0.1.0",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
