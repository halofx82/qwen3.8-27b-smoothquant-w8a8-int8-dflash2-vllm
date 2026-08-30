---
license: apache-2.0
base_model:
  - Freaksterz/Qwen3.8-27B-SmoothQuant-W8A8-INT8
tags:
  - vllm
  - qwen3.8
  - dflash2
  - speculative-decoding
  - adapter
---

# Freaksterz DFlash2 basis adapter

This repository provides the runtime adapter used by
[`halofx82/qwen3.8-27b-smoothquant-w8a8-int8-dflash2-vllm`](https://github.com/halofx82/qwen3.8-27b-smoothquant-w8a8-int8-dflash2-vllm).

It bridges the rotated residual basis of
[Freaksterz/Qwen3.8-27B-SmoothQuant-W8A8-INT8](https://huggingface.co/Freaksterz/Qwen3.8-27B-SmoothQuant-W8A8-INT8)
with the native basis expected by
[z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)
and [lued/Qwen3.8-27B-DFlash2-W8](https://huggingface.co/lued/Qwen3.8-27B-DFlash2-W8).

It is not a standalone model. Use it only through the linked runtime project,
which downloads this file automatically at container startup.
