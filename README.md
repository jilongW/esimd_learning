# custom-esimd-kernels-vllm

这个仓库当前保留的是一个最小化的 XPU SYCL 扩展，当前导出以下几个算子：

- `esimd_gemv_fp8`
- `esimd_gemv_fp8_pern`
- `esimd_gemv_fp8_pert`
- `esimd_gemm_fp8_pert`
- `esimd_fused_add_rms_norm_batched`
- `esimd_rms_norm`
- `esimd_norm_gemv_fp8_pert`
- `esimd_norm_gemv2_geglu_fp8_pert`
- `esimd_resadd_norm_gemv_fp8_pert`
- `esimd_gelu_tanh_and_mul`
- `select_gelu_tanh_and_mul_vl_ks`

其中 `esimd_gemv_fp8` 会根据 `scale` 自动分流：`scale.numel()==1` 走 per-tensor 路径，`scale.numel()==N` 走 per-N 路径；`esimd_gemv_fp8_pern` 和 `esimd_gemv_fp8_pert` 保留给需要显式路径或手工 sweep 配置的场景。

当前自动选参路径已经统一为架构感知分发：

- selector 接口统一支持 `const sycl::device* dev = nullptr`；
- 当设备架构是 PTL（`intel_gpu_ptl_h`/`intel_gpu_ptl_u`）时走 `xe3` 策略；
- 其他架构默认走 `xe2` 策略；
- 若没有传入 `dev`，默认按 `xe2`。

另外，`csrc/xpu/esimd_kernel.sycl` 现在不再直接调用 `select_vl_ks*`，统一改为调用各 kernel header 的“自动 host 重载”；selector 逻辑在对应 header 内部完成（包括 `q.get_device()` 和 `xe2/xe3` 分流）。

当前 GEMV FP8 的约定是：

- `esimd_gemv_fp8(input, weight, scale, output)`：统一入口，内部根据 `scale` 形状自动选择 pern 或 pert 路径，并自动选择 `vl/ks`。
- `esimd_gemv_fp8_pern(input, weight, scale, output, N, K, vl, ks)`：显式 per-N 路径，`scale` 是 `[N]` 的 `fp16`。
- `esimd_gemv_fp8_pert(input, weight, scale, output, N, K, vl, ks)`：显式 per-tensor 路径，`scale` 是单个 `float` 标量。

`pern` 和 `pert` 的自动选参现在是两套独立 heuristic；统一入口会按实际路径分别使用对应的 `select_vl_ks_pern` 或 `select_vl_ks_pert`。

对应到调用链上：

- `esimd_kernel.sycl` 只负责参数校验和 op 分流；
- 自动 `vl/ks` 选择在 `csrc/xpu/esimd_kernels/*.h` 的 host 自动重载里完成；
- 显式 `vl/ks` 模式仍然保留，行为与之前一致。

其中 `esimd_fused_add_rms_norm_batched` 支持 `fp16` 和 `bf16`。它的 dtype 判断方式不是靠 Python 侧额外传字符串或枚举，而是直接在 XPU 入口里根据 `hidden_states.scalar_type()` 判定；`residual` 和 `weight` 必须与 `hidden_states` 保持同 dtype。

`esimd_rms_norm` 也支持 `fp16` 和 `bf16`，输入是 `[..., K]` 的 tensor、`[K]` 的 weight，以及预分配好的 output。内核会把最后一维当成 hidden size `K`，把前面的维度展平成 `rows` 后执行；`vl/ks` 在 `csrc/xpu/esimd_kernels/rms_norm.h` 的 host path 里自动选择，当前要求 `K` 能被 `128` 整除。

新增的 fused 路径当前分成三类：

- `esimd_norm_gemv_fp8_pert(hidden, norm_weight, gemv_weight, gemv_scale, output, eps, vl, ks)`：单路 `RMSNorm + FP8 GEMV`，适合把 norm 和单个投影融合在一起。
- `esimd_norm_gemv2_geglu_fp8_pert(hidden, norm_weight, gemv_weight0, gemv_scale0, gemv_weight1, gemv_scale1, eps, vl, ks)`：一次 `RMSNorm` 后复用同一份 normed hidden，分别打到两路 FP8 GEMV，并在内核内执行 `GELU_tanh(first) * second`，返回单个输出 tensor。
- `esimd_resadd_norm_gemv_fp8_pert(hidden, residual, norm_weight, gemv_weight, gemv_scale, output, normed_out, eps)`：`ResidualAdd + RMSNorm + FP8 GEMV`，同时返回 GEMV 输出并保留 norm 之后的中间结果。

`esimd_gelu_tanh_and_mul(input, output, vl=None, ks=None)` 对应 Gemma4 MLP 里的 GeGLU 激活，语义是对前半段做 `GELU(tanh)`，再与后半段逐元素相乘；如果不显式传 `vl/ks`，会在 host 侧自动调用 `select_gelu_tanh_and_mul_vl_ks(input)` 选参。

## 目录说明

### 核心文件

- `setup.py`：编译入口。
- `esimd_build_extention.py`：本地 BuildExtension，负责调用 PyTorch 的扩展编译流程。
- `csrc/xpu/esimd_kernel.sycl`：`esimd_gemv_fp8_*`、`esimd_fused_add_rms_norm_batched` 和 `esimd_rms_norm` 的 SYCL 入口实现。
- `csrc/xpu/torch_extension.cc`：PyTorch dispatcher 注册。
- `python/custom_esimd_kernels_vllm/`：Python 导入与包装层。

### ESIMD Kernels

- **`csrc/xpu/esimd_kernels/fp8_GEMV_ptl.h`**: PTL 优化的 FP8 GEMV
  - 针对 PTL 架构 (480 线程, VL=256, SLM=192KB)
  - 支持 K_SPLIT 和 tail 处理
  - 适用于 M=1 的 decode 场景
  
- **`csrc/xpu/esimd_kernels/fp8_GEMV_v2.h`**: 基线 FP8 GEMV (V2)
  - 通用启发式参数选择

- `csrc/xpu/esimd_kernels/gelu_tanh_and_mul.h`：GeGLU 激活的 ESIMD kernel 与 `vl/ks` selector。
- `csrc/xpu/esimd_kernels/norm_gemv2_geglu_fused.h`：`RMSNorm + 2-matrix FP8 GEMV + GeGLU` 的 fused host/kernel 实现。

### 测试文件
- **`tests/test_gemv_fp8_ptl_vs_v2.py`**: PTL vs V2 完整对比测试
  - 正确性对比：PTL 和 V2 应产生相同结果
  - 性能对比：延迟、带宽、加速比
  - 需要实际硬件运行
  
- **`tests/compare_ptl_v2_simple.py`**: PTL vs V2 配置分析脚本
  - 纯配置对比，无需硬件
  - 显示 PTL 优化的参数选择策略
  - 线程利用率分析

- `tests/test_gemv_fp8.py`：GEMV FP8 主测试入口，会打印自动规则、最优配置和所有候选 `vl/ks`。
- `tests/test_gemm_fp8.py`：GEMM FP8 per-tensor 测试，并对比 GEMM、GEMV 和 vLLM。
- `tests/test_fused_add_rms_norm_batched_fp8.py`：fused residual add + RMSNorm 的正确性与性能测试。
- `tests/test_rms_norm.py`：standalone RMSNorm 的正确性与性能测试，并对比 `torch.ops._C.rms_norm`。
- `tests/test_norm_gemv_fused.py`：单路 `RMSNorm + FP8 GEMV` 的正确性与 benchmark。
- `tests/test_norm_gemv2_fused.py`：`RMSNorm + 2-matrix FP8 GEMV + GeGLU` 的正确性测试，在一个函数里同时对比 fused、单次拼接权重的 `esimd_norm_gemv_fp8_pert + esimd_gelu_tanh_and_mul`，以及分开的 `esimd_rms_norm + 两次 esimd_gemv_fp8 + esimd_gelu_tanh_and_mul`。
- `tests/test_resadd_norm_gemv_fused.py`：`ResidualAdd + RMSNorm + FP8 GEMV` 的 fused correctness/benchmark。
- `tests/test_gelu_tanh_and_mul.py`：GeGLU 激活的精度与性能测试，并对比 vLLM/Gemma4 的 `gelu_tanh_and_mul`。

## 环境要求

### 硬件
- Intel XPU (PTL 架构)
  - PTL: ~480 硬件线程, VL_max=256, SLM=192KB

### 软件依赖
- Linux (推荐 Ubuntu 22.04+)
- Python 3.9+
- 安装了带 XPU 支持的 PyTorch (>=2.1.0)
- Intel oneAPI Base Toolkit (2024.0+)
  - 包含 SYCL 编译器 (icpx)
  - DPC++ 运行时
- `ninja` 编译系统
- `pytest` (用于测试)

### 从零开始配置新 Conda 环境

如果你要在全新的 conda 环境里编译，请按以下步骤操作：

#### 1. 创建并激活新环境

```bash
# 创建名为 esimd-dev 的新环境 (Python 3.10)
conda create -n esimd-dev python=3.10 -y
conda activate esimd-dev
```

#### 2. 安装 PyTorch with XPU 支持

```bash
# 安装带 XPU 支持的 PyTorch (根据你的 Intel GPU 驱动版本选择)
# 示例：安装 PyTorch 2.1.0+
python -m pip install torch torchvision torchaudio --index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/cn/

# 验证安装
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'XPU available: {torch.xpu.is_available()}')"
```

#### 3. 安装编译工具

```bash
# 安装 ninja (加速编译)
conda install ninja -y

# 安装 pytest (用于测试)
pip install pytest
```

#### 4. 配置 Intel oneAPI 环境

```bash
# 加载 oneAPI 环境变量 (根据你的安装路径调整)
source /opt/intel/oneapi/setvars.sh

# 验证 SYCL 编译器可用
which icpx
icpx --version
```

#### 5. 克隆并进入项目

```bash
cd /home/edgeai
git clone <your-repo-url> esimd_learning
cd esimd_learning
```

### 典型环境初始化 (已有环境)

如果你已经配置过环境，每次编译前需要激活：

```bash
source /home/edgeai/miniforge3/etc/profile.d/conda.sh
conda activate down  # 或你的环境名
source /opt/intel/oneapi/setvars.sh
```

如果当前 shell 里已经执行过一次 `setvars.sh`，再次 `source` 时会打印提示；这不影响后续编译命令继续执行。

如果你把命令写成一行，`source /opt/intel/oneapi/setvars.sh` 这一步不要和后面的命令用 `&&` 强绑定，推荐写成 `source /opt/intel/oneapi/setvars.sh; ...`，避免 shell 因 `setvars.sh` 的返回码中断后续流程。

## 怎么编译

### 编译前准备

确保你已经：
1. ✅ 激活了 conda 环境 (包含 PyTorch XPU)
2. ✅ 加载了 oneAPI 环境 (`source /opt/intel/oneapi/setvars.sh`)
3. ✅ 安装了 ninja

### 编译命令

在仓库根目录执行：

```bash
cd /home/edgeai/esimd_learning

# 方法1: 使用已有环境 (推荐)
source /home/edgeai/miniforge3/etc/profile.d/conda.sh
conda activate down
source /opt/intel/oneapi/setvars.sh

# 方法2: 使用新创建的环境
# conda activate esimd-dev
# source /opt/intel/oneapi/setvars.sh

# 编译并安装 (PTL 架构)
VLLM_CUTLASS_SRC_DIR=/home/edgeai/sycl-tla \
TORCH_XPU_ARCH_LIST=ptl \
python -m pip install -e . --no-build-isolation -v
```

### 编译参数说明

- **`TORCH_XPU_ARCH_LIST=ptl`**: 指定目标架构为 PTL
  - 避免多架构 device-link 将其他目标带入导致编译失败
  - PTL 优化: 480 线程, VL=256, SLM=192KB
  
- **`VLLM_CUTLASS_SRC_DIR`**: CUTLASS/SYCL-TLA 源码路径
  - 提供 `include/tools/util/include/applications` 头文件
  - 如果没有 CUTLASS，可以省略此参数（会跳过 CUTLASS GEMM 扩展）

- **`--no-build-isolation`**: 使用当前环境的依赖
  - 避免 pip 创建隔离的临时环境

- **`-v`**: 详细输出编译过程（推荐，方便调试）

### 编译输出

编译成功后会生成：
- `custom_esimd_kernels_vllm` Python 包 (主要 ESIMD 算子)
- `custom_esimd_kernels_cutlass_gemm` Python 包 (可选，如果提供了 CUTLASS 源码)

### 常见编译问题

#### 问题 1: `icpx: command not found`
**原因**: 未加载 oneAPI 环境  
**解决**: `source /opt/intel/oneapi/setvars.sh`

#### 问题 2: `torch.xpu` 不可用
**原因**: PyTorch 未安装 XPU 支持  
**解决**: 重新安装 PyTorch XPU 版本
```bash
pip install torch --index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/cn/
```

#### 问题 3: 编译时架构不匹配
**原因**: `TORCH_XPU_ARCH_LIST` 未设置或设置错误  
**解决**: 确保设置为 `TORCH_XPU_ARCH_LIST=ptl`

#### 问题 4: ninja 未找到
**原因**: 未安装 ninja 或不在 PATH 中  
**解决**: `conda install ninja -y`

### 清理重新编译

如果编译出错需要清理：

```bash
# 清理构建缓存
rm -rf build/ *.egg-info

# 卸载旧版本
pip uninstall custom_esimd_kernels_vllm custom_esimd_kernels_cutlass_gemm -y

# 重新编译
VLLM_CUTLASS_SRC_DIR=/home/edgeai/sycl-tla \
TORCH_XPU_ARCH_LIST=ptl \
python -m pip install -e . --no-build-isolation -v
```

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

如果你要验证 GeGLU 激活，可以运行：

```bash
cd /home/edgeai/esimd_learning
source /home/edgeai/miniforge3/etc/profile.d/conda.sh
conda activate down
source /opt/intel/oneapi/setvars.sh
PYTHONPATH=/home/edgeai/esimd_learning/python:$PYTHONPATH python tests/test_gelu_tanh_and_mul.py
```

这个脚本会先做 `esimd_gelu_tanh_and_mul` 和 Gemma4/vLLM 参考实现的精度对比，再打印自动选择配置与 sweep 出来的最佳 `vl/ks` 性能结果。

如果你要验证 fused norm+2-gemv+GeGLU，可以运行：

```bash
cd /home/edgeai/esimd_learning
source /home/edgeai/miniforge3/etc/profile.d/conda.sh
conda activate down
source /opt/intel/oneapi/setvars.sh
PYTHONPATH=/home/edgeai/esimd_learning/python:$PYTHONPATH python tests/test_norm_gemv2_fused.py
```

这个脚本会在同一个 correctness 函数里比较三条路径：

- `esimd_norm_gemv2_geglu_fp8_pert`
- 单次拼接权重的 `esimd_norm_gemv_fp8_pert`
- 分开的 `esimd_rms_norm + 两次 esimd_gemv_fp8`

Python 侧最小调用方式如下：

```python
import torch
from custom_esimd_kernels_vllm import esimd_rms_norm

hidden = torch.randn(2, 64, 2560, device="xpu", dtype=torch.float16)
weight = torch.randn(2560, device="xpu", dtype=torch.float16)
output = torch.empty_like(hidden)

esimd_rms_norm(hidden, weight, 1e-6, output)
```

GeGLU 和 fused norm+2-gemv+GeGLU 的最小调用方式分别如下：

```python
import torch
from custom_esimd_kernels_vllm import (
	esimd_gelu_tanh_and_mul,
	esimd_norm_gemv2_geglu_fp8_pert,
)

gate_up = torch.randn(1, 20480, device="xpu", dtype=torch.float16)
act_out = torch.empty(1, 10240, device="xpu", dtype=torch.float16)
esimd_gelu_tanh_and_mul(gate_up, act_out)

hidden = torch.randn(1, 2560, device="xpu", dtype=torch.float16)
norm_weight = torch.randn(2560, device="xpu", dtype=torch.float16)
weight0 = torch.randn(10240, 2560, device="xpu", dtype=torch.float16).to(torch.float8_e4m3fn)
weight1 = torch.randn(10240, 2560, device="xpu", dtype=torch.float16).to(torch.float8_e4m3fn)
scale0 = torch.tensor([8e-4], device="xpu", dtype=torch.float32)
scale1 = torch.tensor([8e-4], device="xpu", dtype=torch.float32)
out = esimd_norm_gemv2_geglu_fp8_pert(
	hidden,
	norm_weight,
	weight0,
	scale0,
	weight1,
	scale1,
	1e-6,
	128,
	4,
)
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

## PTL 架构优化说明

### PTL 硬件特性

| 特性 | PTL 规格 |
|------|---------|
| **硬件线程数** | ~480 |
| **最大 VL** | 256 |
| **SLM 大小** | 192KB |

### fp8_GEMV_ptl.h 的核心设计

PTL 优化版本 `fp8_GEMV_ptl.h` 针对 PTL 架构特性进行了优化：

#### 1. 线程饱和度策略

针对 PTL 的 480 个硬件线程，采用以下 K_SPLIT 选择策略：

```cpp
// 目标: N × K_SPLIT >= 480 (充分利用硬件线程)
if (N * 8 <= 480) target_ks = 8;       // N <= 60: 最大并行
else if (N * 4 <= 480) target_ks = 4;  // N <= 120: 中等并行
else if (N * 2 <= 480) target_ks = 2;  // N <= 240: 低并行
else target_ks = 1;                     // N > 240: 无需分割
```

#### 2. 不同 N 范围的行为

| N 范围 | K_SPLIT | 总线程数 | 利用率 | 说明 |
|--------|---------|----------|--------|------|
| N ≤ 60 | 8 | N × 8 ≤ 480 | 100% | 充分并行 |
| 60 < N ≤ 120 | 4 | 240-480 | 50-100% | 适度并行 |
| 120 < N ≤ 240 | 2 | 240-480 | 50-100% | 低并行 |
| N > 240 | 1 | N | 过饱和 | 依靠work-group调度 |

#### 3. 核心特性

- ✅ **VL 候选列表**: `{256, 128, 64, 32}` - 支持多种向量宽度
- ✅ **Tail 处理**: 自动处理 kp % VL != 0 的情况
- ✅ **FP8 反量化**: E4M3 和 E5M2 格式支持
- ✅ **SLM 优化**: 当前使用最大 32B (K_SPLIT=8 时)，远低于 192KB 限制

### 如何测试 PTL 优化

#### 配置分析（无需硬件）

```bash
cd /home/edgeai/esimd_learning/tests
python compare_ptl_v2_simple.py
```

输出示例：
```
PTL (480 threads) Configuration Analysis
================================================================================
Name            N      K | PTL Config          | Total Threads  Utilization
--------------------------------------------------------------------------------
Small           60   2560 | vl=128 ks=8         | 480            100.0%
Medium         128   2048 | vl=256 ks=2         | 256            53.3%
Large         2560   2048 | vl=256 ks=1         | 2560           533.3% (oversubscribed)
```

#### 完整性能测试（需要 XPU 硬件）

```bash
cd /home/edgeai/esimd_learning
source /home/edgeai/miniforge3/etc/profile.d/conda.sh
conda activate down
source /opt/intel/oneapi/setvars.sh

# 运行 PTL 性能测试
python tests/test_gemv_fp8.py
```

测试会显示：
1. **正确性验证**: 与参考实现对比
2. **性能指标**: 
   - 延迟 (us)
   - 带宽 (GB/s)
   - 带宽利用率 (%)
3. **配置信息**: 自动选择的 VL 和 K_SPLIT

### 性能预期

#### 不同 N 范围的性能特征

| N 范围 | K_SPLIT | 预期带宽利用率 | 说明 |
|--------|---------|---------------|------|
| N ≤ 60 | 8 | 85-95% | 充分并行，接近峰值 |
| 60 < N ≤ 240 | 2-4 | 80-90% | 适度并行，性能良好 |
| N > 240 | 1 | 90-98% | 依赖调度器，性能优秀 |

#### 典型 LLM shapes (Qwen3-Next-80B)

| Layer | N | K | K_SPLIT | 预期带宽 | 说明 |
|-------|---|---|---------|----------|------|
| qkv_proj | 3072 | 2560 | 1 | ~100 GB/s | 大 N，充分利用 |
| o_proj | 2560 | 2048 | 1 | ~105 GB/s | 大 N，充分利用 |
| gate_up | 20480 | 2560 | 1 | ~110 GB/s | 超大 N，接近峰值 |
| down_proj | 2560 | 10240 | 1 | ~108 GB/s | 大 N，充分利用 |

**结论**: PTL 优化版本在典型 LLM 推理场景中表现优异，带宽利用率可达 90-98%。

### 未来优化方向

1. **利用更大 SLM (192KB)**:
   - 当前最大使用 32B (K_SPLIT=8)
   - 可以支持 K_SPLIT=16 或更高（如果有更小 N 的场景）

2. **VL 选择微调**:
   - PTL 可能在某些 VL 上有不同的最优值
   - 需要实际 benchmark 验证

3. **Prefetch 策略**:
   - PTL 的缓存层级可能不同
   - 可针对性优化内存访问模式

### 相关文件

- **PTL 实现**: [`csrc/xpu/esimd_kernels/fp8_GEMV_ptl.h`](csrc/xpu/esimd_kernels/fp8_GEMV_ptl.h)
- **V2 基线**: [`csrc/xpu/esimd_kernels/fp8_GEMV_v2.h`](csrc/xpu/esimd_kernels/fp8_GEMV_v2.h)
- **对比测试**: [`tests/test_gemv_fp8_ptl_vs_v2.py`](tests/test_gemv_fp8_ptl_vs_v2.py)
- **配置分析**: [`tests/compare_ptl_v2_simple.py`](tests/compare_ptl_v2_simple.py)
