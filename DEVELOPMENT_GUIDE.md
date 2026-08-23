# Development Guide — LayerStoRmExpertKernels

This document defines the standards for this project. All conventions mirror the sibling project **LayerStoRmKernels** (SM120 SnapMLA attention kernels) so that both libraries share the same structure, style, and integration patterns.

**Read this document at the start of every session.**

## 1. Project Identity

| Property | LayerStoRmKernels (reference) | LayerStoRmExpertKernels (this project) |
|----------|-------------------------------|----------------------------------------|
| Domain | Attention (SnapMLA) | Expert/MoE (grouped GEMM, FFN) |
| Kernel types | Decode, prefill, prep, graph | Grouped GEMM, activation, permute, dequant |
| Python module | `sm120_mla_kernels` | `sm120_expert_kernels` |
| Namespace root | `sm120::` | `sm120::expert::` |
| Target GPU | SM120 (RTX 5090/5080) | SM120 (RTX 5090/5080) |
| Quantization | FP8 e4m3 (KV cache) | NVFP4 E2M1, FP8 e4m3/e5m2 |

Both are consumed by LayerStoRm3 as git submodule OBJECT libraries.

## 2. Directory Structure

```
csrc/                               # C++ CUDA kernel sources
  sm120/                            # SM120-specific kernels
    gemm/                           # Quantized GEMM kernels
      grouped_gemm.h                # NVFP4 + FP8 grouped GEMM param structs + launch
      nvfp4/                        # NVFP4 (FP4 E2M1) GEMM
        nvfp4_gemm.h                # Single GEMM params + GemmOutputDtype enum
        nvfp4_gemm.cu               # CUTLASS 3.x BlockScaledTensorOp, M-dispatch
        nvfp4_grouped_gemm.cu       # Grouped GEMM via PtrArray + TMA warp-specialized
      fp8/                          # FP8 (E4M3) blockwise-scaled GEMM
        fp8_gemm.h                  # Single GEMM params
        fp8_gemm.cu                 # CUTLASS 3.x OpClassTensorOp + blockwise scales
        fp8_grouped_gemm.cu         # Per-expert dispatch loop (CUTLASS grouped TBD)
    gating/                         # SM120-optimized gating kernels
      topk_gating.h                 # TopkGatingParams + launch
      topk_gating.cu                # Warp-level packed uint64_t top-K selection
  smxx/                             # Arch-generic kernels
    utils.h                         # CHECK_CUDA, FLASH_ASSERT macros
    activation/                     # Activation kernels
      fused_swiglu.h                # FusedSwigluParams + launch
      fused_swiglu.cu               # Vectorized 128-bit BF16/FP16 SwiGLU
    permute/                        # Expert routing permute/unpermute
      moe_permute.h                 # Permute/unpermute params + launch
      moe_permute.cu                # CUB radix sort + vectorized scatter/gather
    gating/                         # Arch-generic gating kernels
      adaptive_gating.h             # AdaptiveGatingParams + launch
      adaptive_gating.cu            # Register-bound threshold pruning
    quant/                          # Quantization kernels
      dynamic_fp8_quant.h           # DynamicFp8QuantParams + launch
      dynamic_fp8_quant.cu          # BF16→FP8 E4M3 per-block quantization

tests/                              # Test suite
  test_reference.py                 # Pure-PyTorch reference (CPU, no GPU needed)
  test_kernels.py                   # GPU kernel tests
  helpers/
    __init__.py
    load_sample_data.py

benchmarks/                         # Performance benchmarking
  benchmark_speed.py                # Kernel latency benchmarking
  5080/                             # RTX 5080 results
  5090/                             # RTX 5090 results

sample-data/                        # Sample activations for testing
  texts/                            # Text samples
  generate_samples.py               # Script to generate sample data

test-data/                          # Model configs
  DeepSeek-V3.2/
  config/

samples/                            # Documented usage examples
tools/                              # Utility scripts
ref/                                # Reference implementations (read-only, gitignored)
```

### Parallel with LayerStoRmKernels

| LayerStoRmKernels | This project | Purpose |
|-------------------|-------------|---------|
| `csrc/sm120/decode/sparse_fp8/` | `csrc/sm120/gemm/nvfp4/` | Primary kernel variant |
| `csrc/sm120/decode/dense_fp8/` | `csrc/sm120/gemm/fp8/` | Secondary kernel variant |
| `csrc/sm120/prep/` | `csrc/smxx/quant/` | Quantization |
| `csrc/smxx/mla_combine.cu` | `csrc/smxx/activation/fused_swiglu.cu` | Supporting computation |
| `csrc/smxx/` | `csrc/smxx/` | Arch-generic utilities |

## 3. Naming Conventions

### Files
- Kernel implementations: `snake_case.cu` (e.g., `grouped_gemm.cu`, `swiglu.cu`)
- Headers: `snake_case.h` (e.g., `expert_config.h`, `params.h`)
- Instantiations: `{model}_{variant}.cu` (e.g., `v32_nvfp4_n2048.cu`)
- Tests: `test_{domain}.py` (e.g., `test_reference.py`, `test_kernels.py`)
- Benchmarks: `benchmark_{what}.py` (e.g., `benchmark_speed.py`)

### Namespaces

Follow the same hierarchical pattern as LayerStoRmKernels:

```cpp
// LayerStoRmKernels pattern:
// sm120::decode::sparse_fp8    — sparse FP8 decode
// sm120::decode::dense_fp8     — dense FP8 decode
// sm120::prep                  — prep kernels
// sm120::graph                 — CUDA graph runner

// This project:
sm120::expert::gemm::nvfp4     // NVFP4 grouped GEMM
sm120::expert::gemm::fp8       // FP8 grouped GEMM
sm120::expert::activation      // SwiGLU, etc.
sm120::expert::permute         // Expert routing permute/unpermute
sm120::expert::dequant         // Weight dequantization
sm120::expert::config          // Shared configuration (ModelType, dims)
```

### Param Structs

Each kernel defines its own param struct in `params.h`:

```cpp
// LayerStoRmKernels example (csrc/sm120/decode/sparse_fp8/params.h):
namespace sm120::decode::sparse_fp8 {
struct SparseAttnDecodeParams {
    __nv_fp8_e4m3* q_nope;
    __nv_bfloat16* q_rope;
    float* q_scales;
    // ...
};
} // namespace sm120::decode::sparse_fp8

// This project should follow the same pattern:
namespace sm120::expert::gemm::nvfp4 {
struct GroupedGemmParams {
    const void* weights;        // NVFP4 packed weights
    const void* scales;         // Dequantization scales
    const void* activations;    // Input activations (BF16 or FP8)
    void* output;               // Output buffer
    const int* expert_offsets;  // Per-expert token counts
    int num_experts;
    int hidden_dim;
    int intermediate_dim;
    // ...
};
} // namespace sm120::expert::gemm::nvfp4
```

### Launch Functions

Same pattern: param struct + CUDA stream.

```cpp
// LayerStoRmKernels example:
namespace sm120::prep {
void run_fused_q_quant(const FusedQQuantParams& params, cudaStream_t stream);
}

// This project:
namespace sm120::expert::gemm::nvfp4 {
template <ModelType M>
void run_grouped_gemm(const GroupedGemmParams& params, cudaStream_t stream);
}
```

## 4. Build System

### setup.py Structure

Mirror the LayerStoRmKernels `setup.py` exactly. Key elements:

```python
# 1. CUTLASS header detection (same find_cutlass_include function)
# 2. CUDA version mismatch bypass (same _noop_check trick)
# 3. Single compilation unit strategy (bindings.cu includes all inline kernels)
# 4. Separate instantiation .cu files for template-heavy kernels
# 5. SM120 arch detection with SM89 fallback

setup(
    name="sm120_expert_kernels",
    version="0.1.0",
    description="SM120 Expert/MoE CUDA kernels with Python bindings",
    ext_modules=[
        CUDAExtension(
            name="sm120_expert_kernels",
            sources=[
                "csrc/bindings.cu",
                # Add instantiation .cu files here
            ],
            include_dirs=[csrc_dir, cutlass_include],
            extra_compile_args={
                "cxx": ["-std=c++17", "-O2"],
                "nvcc": nvcc_flags,
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.10",
)
```

### Compiler Flags (identical to LayerStoRmKernels)

```
-std=c++17 -O2 --expt-relaxed-constexpr --expt-extended-lambda
-U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__
-U__CUDA_NO_BFLOAT16_CONVERSIONS__ -U__CUDA_NO_HALF2_OPERATORS__
-arch=sm_120  (or sm_89 fallback)
```

### CMake Integration (for LayerStoRm3)

Same pattern as `LAYERSTORM_KERNELS_ADDITION.md`:

```cmake
# In LayerStoRm3's src/CMakeLists.txt:
add_library(expert_kernels OBJECT
    deps/LayerStoRmExpertKernels/csrc/sm120/gemm/nvfp4/instantiations/*.cu
    deps/LayerStoRmExpertKernels/csrc/sm120/gemm/fp8/instantiations/*.cu
    # ...
)
target_include_directories(expert_kernels PRIVATE
    deps/LayerStoRmExpertKernels/csrc
    ${CUTLASS_INCLUDE}
)
```

## 5. Kernel Implementation Patterns

### Shared Memory Budget

SM120 has **99KB** (101,376 bytes) shared memory. All kernels must fit within this.

```cpp
// LayerStoRmKernels documents smem usage in traits.h:
// BF16 path: ~90KB (V at 65KB dominates)
// FP8 path:  ~53KB (V at 32KB, leaves headroom)

// This project: document smem layout in traits.h for each grouped GEMM variant
```

### Thread Configuration

LayerStoRmKernels uses 256 threads (8 warps) with producer/consumer split:
```
Warps 0-3 (consumer): CuTe MMA
Warps 4-7 (producer): global memory loads
```

Expert kernels should document their thread configuration similarly in the kernel file header.

### MMA Atoms (CuTe)

```cpp
// From LayerStoRmKernels (csrc/sm120/components/helpers.h):
using MmaAtomBF16 = MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>;
using MmaAtomFP8  = MMA_Atom<SM89_16x8x32_F32F8F8F32_TN>;

// NVFP4 uses different atoms — document in components/helpers.h
```

### Swizzled Shared Memory

```cpp
// LayerStoRmKernels conventions:
// BF16: Swizzle<3,3,3>, 8 BF16 elements per 128-bit line
// FP8:  Swizzle<3,4,3>, 16 FP8 elements per 128-bit line

// NVFP4: document the appropriate swizzle pattern
```

### `__ldg()` Read-Only Cache Hints

LayerStoRmKernels found `__ldg()` gave -36% on dense decode. Apply to all global loads:

```cpp
// LayerStoRmKernels pattern (csrc/sm120/decode/dense_fp8/splitkv_mla.cu):
auto val = __ldg(&global_ptr[idx]);
```

### Error Checking Macros

```cpp
// From csrc/smxx/utils.h (copy to this project):
#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        abort(); \
    } \
} while (0)

#define FLASH_ASSERT(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "Assertion failed: %s at %s:%d\n", #cond, __FILE__, __LINE__); \
        abort(); \
    } \
} while (0)
```

## 6. Test Conventions

### Reference Tests (CPU, no GPU needed)

File: `tests/test_reference.py`

Pure-PyTorch implementations that establish error budgets. Run standalone.

```python
# LayerStoRmKernels example (tests/test_snapmla_reference.py):
# 6 reference tests: Q-quant roundtrip, K-append roundtrip,
# FP8 vs BF16, longer context, LSE merge, sparse vs dense

# This project: reference tests for each kernel
# - NVFP4 grouped GEMM vs torch.mm (with manual dequant)
# - FP8 grouped GEMM vs torch.mm
# - SwiGLU vs manual gate * silu(up) * down
# - Permute/unpermute round-trip
# - NVFP4 dequant accuracy (max rel err, cosine)
```

Run with: `python tests/test_reference.py -v`

### GPU Kernel Tests

File: `tests/test_kernels.py`

Per-kernel validation against reference. Same structure as LayerStoRmKernels:

```python
# LayerStoRmKernels pattern (tests/test_kernels.py):
def test_kernel_fused_q_quant():
    # 1. Generate inputs on GPU
    # 2. Run CUDA kernel
    # 3. Run PyTorch reference on CPU
    # 4. Compare: cosine > threshold, max_rel_err < budget
```

### Error Budget Reference

| Stage | Metric | Expected | Explanation |
|-------|--------|----------|-------------|
| NVFP4 dequant round-trip | max rel err | Model-dependent | 4-bit mantissa + group scales |
| FP8 GEMM vs BF16 | cosine | >0.995 | FP8 accumulation |
| SwiGLU precision | cosine | >0.999 | BF16 activation math |
| Permute round-trip | abs diff | 0 | Exact index gather/scatter |
| Full expert forward | cosine vs BF16 | >0.99 | End-to-end |

### Benchmark Conventions

File: `benchmarks/benchmark_speed.py`

```python
# LayerStoRmKernels pattern (benchmarks/benchmark_speed.py):
# - CUDA events for timing (not wall clock)
# - 10 warmup + 100 timed iterations
# - Report: median, min, p95, mean, std (microseconds)
# - Output: JSON file with structured results
# - Per-model, per-sequence-length measurements
```

GPU-specific results stored in `benchmarks/5080/` and `benchmarks/5090/`.

## 7. Documentation Standards

### AGENTS.md — Architecture Documentation

Full project documentation. Must include:
1. What this project is (one paragraph)
2. Expert data structures (weight format, activation layout)
3. SM120 hardware constraints (same as LayerStoRmKernels — 99KB smem, no GMMA, etc.)
4. Project structure (complete file listing)
5. Kernel inventory table
6. Expert pipeline phases
7. Model dimensions table
8. Namespace listing
9. Template parameters
10. Build requirements

### USAGE.md — API Documentation

User-facing API docs. Must include:
1. What this is
2. Weight format specification
3. Kernel inventory table (file, precision, purpose)
4. Expert forward flow
5. C++ API (param structs + launch functions)
6. Python API (function signatures)
7. Model dimensions table
8. Build instructions

### TESTING.md — Validation Plan

Phased testing plan. Must include:
1. Prerequisites
2. What exists today
3. Phase 1: Build system + Python bindings
4. Phase 2: Per-kernel unit tests (with code examples)
5. Phase 3: Numerical consistency tests
6. Phase 4: End-to-end validation
7. Phase 5: Stress testing
8. Error budget reference table
9. File reference

### NOTICE.md — Assumptions & Known Issues

Format requirements, implicit assumptions, TODOs. Update as discovered.

### INSTALL.md — Build & Installation

Same structure as LayerStoRmKernels:
1. Prerequisites (hardware, CUDA, packages)
2. Python environment setup
3. Build command
4. Smoke test
5. Test commands
6. Troubleshooting

## 8. Python Bindings Pattern

### bindings.cu — Main Compilation Unit

```cpp
// LayerStoRmKernels pattern (csrc/bindings.cu):
// Includes all prep kernels and graph runner as a single TU
// to avoid duplicate symbol issues with inline definitions

// This project:
#include "sm120/activation/swiglu.cu"
#include "sm120/dequant/nvfp4_dequant.cu"
#include "sm120/dequant/fp8_dequant.cu"
#include "sm120/permute/expert_permute.cu"
// Instantiation .cu files compiled separately (template-heavy)
```

### bindings_python.cu — pybind11 Module

```cpp
// LayerStoRmKernels pattern (csrc/bindings_python.cu):
#include <torch/extension.h>

// Wrapper functions accepting PyTorch tensors:
torch::Tensor grouped_gemm_nvfp4(
    torch::Tensor weights,
    torch::Tensor scales,
    torch::Tensor activations,
    torch::Tensor expert_offsets,
    int hidden_dim,
    int intermediate_dim
) {
    // 1. Extract raw pointers via .data_ptr()
    // 2. Populate param struct
    // 3. Call launch function
    // 4. Return output tensor
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("grouped_gemm_nvfp4", &grouped_gemm_nvfp4, "NVFP4 grouped GEMM");
    // ... add other kernel wrappers
}
```

## 9. LayerStoRm3 Integration Pattern

Same as `LAYERSTORM_KERNELS_ADDITION.md`:

1. **Git submodule**: `git submodule add ../LayerStoRmExpertKernels deps/LayerStoRmExpertKernels`
2. **CMake OBJECT library**: compile instantiation `.cu` files
3. **C++ wrapper headers**: thin wrappers in `layerstorm::compute` namespace
4. **Tests**: gtest, `REQUIRES_GPU()` macro

### C++ Wrapper Pattern

```cpp
// LayerStoRm3's src/compute/kernels/expert/expert_gemm.h:
#pragma once
#include "sm120/gemm/nvfp4/params.h"

namespace layerstorm::compute {
void launch_grouped_gemm_nvfp4(/* params */);
void launch_grouped_gemm_fp8(/* params */);
void launch_swiglu(/* params */);
void launch_expert_permute(/* params */);
void launch_expert_unpermute(/* params */);
} // namespace layerstorm::compute
```

## 10. Code Style

- C++17 (same as LayerStoRmKernels)
- `#pragma once` for all headers
- `__nv_bfloat16`, `__nv_fp8_e4m3`, `__nv_fp8_e5m2` for CUDA numeric types
- Raw pointers in param structs (no smart pointers — kernel launch interface)
- `const __restrict__` on all read-only kernel parameters
- `__forceinline__ __device__` for small device helpers
- `__noinline__` for functions that cause register pressure (see LayerStoRmKernels INV-10a)
- Apache-2.0 attribution when adapting reference code

## 11. Commit & Branch Conventions

- Descriptive commit messages (what + why)
- Feature branches for kernel development
- Optimization passes tracked in OPTIMIZATION_IMPROVEMENTS.md with benchmark data
