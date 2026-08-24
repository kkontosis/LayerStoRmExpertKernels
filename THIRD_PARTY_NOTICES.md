# Third-Party Notices — LayerStoRmExpertKernels

LayerStoRmExpertKernels is licensed under the Apache License 2.0 (see
`LICENSE.md`). Portions of this repository are derived from or adapted from
the third-party projects listed below. Where a section says "see MIT License
text below", the full license text in Appendix A applies together with that
section's copyright line(s).

---

## vLLM

- Upstream: https://github.com/vllm-project/vllm
- License: Apache-2.0 — Copyright contributors to the vLLM project
- What was derived: the fused SwiGLU activation kernel
  (`csrc/smxx/activation/fused_swiglu.{cu,h}`, from
  `csrc/activation_kernels.cu`), the MoE permute/unpermute kernels
  (`csrc/smxx/permute/moe_permute.{cu,h}`, from
  `csrc/moe/permute_unpermute_kernels/`), and the grouped top-k gating
  structure (`csrc/sm120/gating/topk_gating.cu`, from
  `grouped_topk_kernels.cu`).

## NVIDIA TensorRT-LLM

- Upstream: https://github.com/NVIDIA/TensorRT-LLM
- License: Apache-2.0 — Copyright (c) 2011-2025 NVIDIA CORPORATION &
  AFFILIATES. All rights reserved.
- What was derived: the top-k gating kernels
  (`csrc/sm120/gating/topk_gating.{cu,h}`, adapted from
  `RoutingKernelTopK.cuh`).
- TensorRT-LLM ships no Apache-2.0 NOTICE file at its repository root, so
  there are no NOTICE contents to reproduce under Apache-2.0 §4(d).

## llama.cpp / ggml

- Upstream: https://github.com/ggerganov/llama.cpp
- License: MIT — Copyright (c) 2023-2026 The ggml authors (see MIT License
  text below)
- What was derived: the DeepSeek-V4 SwiGLU clamp semantics
  (`ggml_swiglu_split`, `LLM_ARCH_DEEPSEEK4`) implemented in
  `csrc/smxx/activation/fused_swiglu.{cu,h}`.

## FlashMLA

- Upstream: https://github.com/deepseek-ai/FlashMLA
- License: MIT — Copyright (c) 2025 DeepSeek (see MIT License text below)
- What was derived: the CUDA error-check / assert macros in
  `csrc/smxx/utils.h`.

## NVIDIA CUTLASS

- Upstream: https://github.com/NVIDIA/cutlass (consumed as the
  `3rd-party/cutlass` git submodule; not vendored in this tree)
- License: BSD-3-Clause — Copyright (c) 2017 - 2026 NVIDIA CORPORATION &
  AFFILIATES. All rights reserved. (full text in Appendix B)
- What is used: CUTLASS/CuTe headers are a build dependency of the CUDA
  kernels in `csrc/`. Binaries built from this repository incorporate CUTLASS
  header code; the BSD-3-Clause notice applies to such binaries.

---

## Appendix A — MIT License text

The following license text applies to the MIT-licensed material identified
above, together with the copyright lines given in each section:

```
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Appendix B — BSD-3-Clause (NVIDIA CUTLASS)

```
Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: BSD-3-Clause

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Appendix C — Apache License 2.0

This repository is licensed under the MIT License — see `LICENSE.md`.
It applies both to LayerStoRmExpertKernels itself (Copyright 2026 Kimon
Kontosis) and to the Apache-2.0-licensed upstream material identified above
(vLLM, NVIDIA TensorRT-LLM).
