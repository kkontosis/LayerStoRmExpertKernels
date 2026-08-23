// SM120 Expert Kernels — single compilation unit.
//
// Arch-generic kernel sources (csrc/smxx/) are #included here for single-TU
// compilation. SM120-specific CUTLASS-heavy kernels are compiled as separate
// TUs listed in setup.py.

// Arch-generic kernel implementations (local)
#include "smxx/activation/fused_swiglu.cu"
#include "smxx/permute/moe_permute.cu"
#include "smxx/gating/adaptive_gating.cu"

// SM120 kernel headers — local
#include "sm120/gating/topk_gating.h"

// Python bindings (pybind11 module definition)
#include "bindings_python.cu"
