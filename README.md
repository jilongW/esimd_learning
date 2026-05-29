# custom-esimd-kernels-vllm

这个仓库当前保留的是一个最小化的 XPU SYCL 扩展，当前导出以下几个算子：

- `esimd_gemv_fp8`
- `esimd_gemv_fp8_pern`
- `esimd_gemv_fp8_pert`
- `esimd_gemm_fp8_pert`
- `esimd_fused_add_rms_norm_batched`
- `esimd_rms_norm`

其中 `esimd_gemv_fp8` 会根据 `scale` 自动分流：`scale.numel()==1` 走 per-tensor 路径，`scale.numel()==N` 走 per-N 路径；`esimd_gemv_fp8_pern` 和 `esimd_gemv_fp8_pert` 保留给需要显式路径或手工 sweep 配置的场景。

当前 GEMV FP8 的约定是：

- `esimd_gemv_fp8(input, weight, scale, output)`：统一入口，内部根据 `scale` 形状自动选择 pern 或 pert 路径，并自动选择 `vl/ks`。
- `esimd_gemv_fp8_pern(input, weight, scale, output, N, K, vl, ks)`：显式 per-N 路径，`scale` 是 `[N]` 的 `fp16`。
- `esimd_gemv_fp8_pert(input, weight, scale, output, N, K, vl, ks)`：显式 per-tensor 路径，`scale` 是单个 `float` 标量。

`pern` 和 `pert` 的自动选参现在是两套独立 heuristic；统一入口会按实际路径分别使用对应的 `select_vl_ks_pern` 或 `select_vl_ks_pert`。

其中 `esimd_fused_add_rms_norm_batched` 支持 `fp16` 和 `bf16`。它的 dtype 判断方式不是靠 Python 侧额外传字符串或枚举，而是直接在 XPU 入口里根据 `hidden_states.scalar_type()` 判定；`residual` 和 `weight` 必须与 `hidden_states` 保持同 dtype。

`esimd_rms_norm` 也支持 `fp16` 和 `bf16`，输入是 `[..., K]` 的 tensor、`[K]` 的 weight，以及预分配好的 output。内核会把最后一维当成 hidden size `K`，把前面的维度展平成 `rows` 后执行；`vl/ks` 在 `csrc/xpu/esimd_kernels/rms_norm.h` 的 host path 里自动选择，当前要求 `K` 能被 `128` 整除。

## 目录说明

- `setup.py`：编译入口。
- `esimd_build_extention.py`：本地 BuildExtension，负责调用 PyTorch 的扩展编译流程。
- `csrc/xpu/esimd_kernel.sycl`：`esimd_gemv_fp8_*`、`esimd_fused_add_rms_norm_batched` 和 `esimd_rms_norm` 的 SYCL 入口实现。
- `csrc/xpu/torch_extension.cc`：PyTorch dispatcher 注册。
- `python/custom_esimd_kernels_vllm/`：Python 导入与包装层。
- `tests/test_gemv_fp8.py`：GEMV FP8 主测试入口，会打印自动规则、最优配置和所有候选 `vl/ks`。
- `tests/test_gemm_fp8.py`：GEMM FP8 per-tensor 测试，并对比 GEMM、GEMV 和 vLLM。
- `tests/test_fused_add_rms_norm_batched_fp8.py`：fused residual add + RMSNorm 的正确性与性能测试。
- `tests/test_rms_norm.py`：standalone RMSNorm 的正确性与性能测试，并对比 `torch.ops._C.rms_norm`。

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

如果你把命令写成一行，`source /opt/intel/oneapi/setvars.sh` 这一步不要和后面的命令用 `&&` 强绑定，推荐写成 `source /opt/intel/oneapi/setvars.sh; ...`，避免 shell 因 `setvars.sh` 的返回码中断后续流程。

## 怎么编译

在仓库根目录执行：


```bash
cd /home/edgeai/esimd_learning
source /home/edgeai/miniforge3/etc/profile.d/conda.sh
conda activate down
source /opt/intel/oneapi/setvars.sh
TORCH_XPU_ARCH_LIST=ptl pip install -e . --no-build-isolation
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

`tests/test_gemv_fp8.py` 当前会直接执行 `test_esimd_vs_vllm()`，并额外打印：

- `rule=vl:ks`：自动 heuristic 选出来的配置
- `best=vl:ks`：本次 sweep 测到的最优配置
- `all`：所有被搜索到的候选配置及其 latency
- `in_search=True/False`：自动选择的 `vl/ks` 是否确实出现在候选列表里

如果你想看 GEMM per-tensor 路径与 GEMV/vLLM 的对比，可以直接运行：

```bash
cd /home/edgeai/esimd_learning
source /home/edgeai/miniforge3/etc/profile.d/conda.sh
conda activate down
source /opt/intel/oneapi/setvars.sh
python tests/test_gemm_fp8.py
```

如果你要验证 standalone RMSNorm，可以运行：

```bash
cd /home/edgeai/esimd_learning
source /home/edgeai/miniforge3/etc/profile.d/conda.sh
conda activate down
source /opt/intel/oneapi/setvars.sh
python tests/test_rms_norm.py
```

这个脚本会先做正确性检查，再打印 `esimd_rms_norm` 和 `torch.ops._C.rms_norm` 的 latency、TFLOPS、memory bandwidth，并在 `Config` 列里显示当前自动选择出的 `auto=vl:ks`。

Python 侧最小调用方式如下：

```python
import torch
from custom_esimd_kernels_vllm import esimd_rms_norm

hidden = torch.randn(2, 64, 2560, device="xpu", dtype=torch.float16)
weight = torch.randn(2560, device="xpu", dtype=torch.float16)
output = torch.empty_like(hidden)

esimd_rms_norm(hidden, weight, 1e-6, output)
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
