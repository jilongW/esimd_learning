# custom-esimd-kernels-vllm

这个仓库当前保留的是一个最小化的 XPU SYCL 扩展，只导出一个算子：`esimd_gemv_fp8`。

## 目录说明

- `setup.py`：编译入口。
- `esimd_build_extention.py`：本地 BuildExtension，负责调用 PyTorch 的扩展编译流程。
- `csrc/xpu/esimd_kernel.sycl`：`esimd_gemv_fp8_*` 的 SYCL 入口实现。
- `csrc/xpu/torch_extension.cc`：PyTorch dispatcher 注册。
- `python/custom_esimd_kernels_vllm/`：Python 导入与包装层。
- `tests/test_gemv_fp8.py`：最小测试入口。

## 环境要求

- Linux
- 安装了带 XPU 支持的 PyTorch
- 可用的 Intel oneAPI / SYCL 编译环境
- `ninja`
- `pytest`

典型环境初始化：

```bash
source /home/edgeai/miniforge3/etc/profile.d/conda.sh
conda activate down
source /opt/intel/oneapi/setvars.sh
```

如果当前 shell 里已经执行过一次 `setvars.sh`，再次 `source` 时会打印提示；这不影响后续编译命令继续执行。

## 怎么编译

在仓库根目录执行：


```bash
cd /home/edgeai/esimd_learning
source /home/edgeai/miniforge3/etc/profile.d/conda.sh
conda activate down
source /opt/intel/oneapi/setvars.sh
TORCH_XPU_ARCH_LIST=ptl pip install -e . --no-build-isolation
```

如果你只想本地重编译扩展，也可以直接运行：

```bash
cd /home/edgeai/esimd_learning
source /home/edgeai/miniforge3/etc/profile.d/conda.sh
conda activate down
source /opt/intel/oneapi/setvars.sh
TORCH_XPU_ARCH_LIST=ptl python setup.py build_ext --inplace
```

这里显式固定 `TORCH_XPU_ARCH_LIST=ptl`，避免多架构 device-link 把 `mtl-h` 等目标一起带进来后触发编译失败。
## 怎么运行测试

```bash
cd /home/edgeai/esimd_learning
source /home/edgeai/miniforge3/etc/profile.d/conda.sh
conda activate down
source /opt/intel/oneapi/setvars.sh
python tests/test_gemv_fp8.py
```

如果你要顺手拉起 standalone OpenCL FP16 GEMM 压测，可以运行新增的 [tests/test_cm_fp16_gemm.py](/llm/cm/esimd_learning/tests/test_cm_fp16_gemm.py)。这个测试默认不会参与常规 pytest，需要显式打开；推荐从 [cm.gemm.examples.kernels](cm.gemm.examples.kernels) 目录触发，这样编译和运行入口都放在 GEMM 工程这一侧：

如果你希望先把 FP16 GEMM 工程本身装成 editable package，也可以先执行：

```bash
source /llm/cm/miniforge3/bin/activate
conda activate test
cd /llm/cm/cm.gemm.examples.kernels
TORCH_XPU_ARCH_LIST=ptl pip install -e . --no-build-isolation
```

这条命令已经在 `test` conda 环境里验证通过。

```bash
source /llm/cm/miniforge3/bin/activate
conda activate test
cd /llm/cm/cm.gemm.examples.kernels
CM_GEMM_RUN=1 \
CM_GEMM_LIB=/llm/cm/cm.gemm.examples.kernels/standalone/fp16.gemm/build_pytest/libcm_fp16_gemm.so \
CM_GEMM_KERNEL_BIN=/llm/cm/cm.gemm.examples.kernels/standalone/fp16.gemm/build_pytest/kernel.cm.bin \
python ../esimd_learning/tests/test_cm_fp16_gemm.py
```

常用环境变量：

- `CM_GEMM_LIB`：复用已经编好的 `libcm_fp16_gemm.so`
- `CM_GEMM_KERNEL_BIN`：复用现成的 `kernel.cm.bin`
- `CM_GEMM_CASES`：指定压测形状，格式如 `5120x2560x5120x100x512x256;2048x2048x2048x200`

这个脚本现在会直接加载现成的 `libcm_fp16_gemm.so` 和 `kernel.cm.bin`，并通过导出的 `cm_fp16_gemm_run` 函数执行单次 GEMM。共享库和 kernel binary 都属于 [cm.gemm.examples.kernels](cm.gemm.examples.kernels) 这一侧的产物；性能循环和正确性验证放在 Python 层完成，不再通过子进程启动可执行文件。
