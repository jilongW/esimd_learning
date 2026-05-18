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
source /opt/intel/oneapi/setvars.sh --force
source ~/downstream/bin/activate
```

如果你使用的是虚拟环境，也需要先激活对应 Python 环境。

## 怎么编译

在仓库根目录执行：


```bash
cd /home/edgeai/esimd_learning
TORCH_XPU_ARCH_LIST=ptl pip install -e . --no-build-isolation
```
## 怎么运行测试

```bash
cd /home/edgeai/esimd_learning
python tests/test_gemv_fp8.py
```
