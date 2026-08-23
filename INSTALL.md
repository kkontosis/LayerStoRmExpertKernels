# Build & Install — SM120 Expert Kernels

## Prerequisites

- NVIDIA GPU: SM120 (RTX 5090/5080) or SM89+ (RTX 4090) for FP8 MMA
- CUDA Toolkit: 13.x (or 12.8+ for SM120 target)
- Linux with `apt` package manager

### System packages

```bash
sudo apt-get install libcutlass-dev python3-dev
```

## Python environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### PyTorch (CUDA 13.x systems)

```bash
pip install torch==2.9.1+cu130 --index-url https://download.pytorch.org/whl/cu130
```

For CUDA 12.8 systems:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

### Build dependencies

```bash
pip install ninja pybind11 pytest
```

## Build

```bash
pip install -e . --no-build-isolation
```

Smoke test:

```bash
python -c "import sm120_expert_kernels; print('OK')"
```

## Test

```bash
# GPU kernel tests (requires CUDA GPU)
pytest tests/test_kernels.py -v

# Pure-PyTorch reference tests (CPU only, no GPU needed)
python tests/test_reference.py -v
```

## Troubleshooting

### CUDA version mismatch

`setup.py` bypasses PyTorch's strict CUDA version check. System CUDA 13.1 with PyTorch cu130 is ABI-compatible.

### CUTLASS not found / NVFP4 GEMM fails

CUTLASS **4.4.2+** is required for NVFP4 GEMM on SM120. The pip package `nvidia-cutlass` (4.2.0) has a bug — use a local checkout instead:

```bash
git clone --branch v4.4.2 https://github.com/NVIDIA/cutlass.git /path/to/cutlass
CUTLASS_PATH=/path/to/cutlass pip install -e . --no-build-isolation
```

If you only need FP8 GEMM (not NVFP4), the pip package works:

```bash
pip install nvidia-cutlass
```

### Missing Python.h

```bash
sudo apt-get install python3-dev
```
