# Hardware compatibility matrix

This project is **evaluation-backed**, not "works everywhere." The only
configuration a maintainer has verified end-to-end is listed as `verified`.
Everything else is honestly marked `community` (reported by a user, not re-run by
a maintainer) or `unknown` (not tested). Do not assume an unlisted combination
works — file a ROCm-compatibility issue with your `rocm-smi` / `rocminfo` output.

## Verified by a maintainer

| GPU | gfx | ROCm | Backend | Status | Source |
|---|---|---|---|---|---|
| Radeon RX 7900 series (gfx1100, 48 GB ×4) | gfx1100 | 7.2 | llama.cpp (HIP, BF16 GGUF) | **verified** | full 1651-page run = 92.09; canary 93.33 |
| Radeon RX 7900 series (gfx1100) | gfx1100 | 7.2 | vLLM (Flash-Attn ViT) | **verified (canary only)** | canary 148 = 94.81; full-set invalid (server crashes) |
| Radeon RX 7900 series (gfx1100) | gfx1100 | 7.2 | transformers (SDPA ViT) | **verified (canary only)** | canary 148 = 94.11; full-set not run (~40 h) |

Stack details (torch/hip/transformers/vLLM) are pinned in
[`reproducibility.lock.yaml`](../reproducibility.lock.yaml).

## Community / unknown

| GPU | gfx | ROCm | Backend | Status | Notes |
|---|---|---|---|---|---|
| RX 7900 XTX | gfx1100 | ? | llama.cpp | community | expected to work (same gfx1100); please report |
| RX 7900 XT / GRE | gfx1100 | ? | llama.cpp | unknown | same arch; untested |
| Other RDNA3 (gfx1101/1102/1150) | gfx110x/1150 | ? | any | unknown | not tested; ViT pixel cap may differ |
| RDNA2 / gfx103x | gfx103x | ? | any | unknown | not tested |
| CDNA (MI250/MI300) | gfx90a/942 | ? | any | unknown | not tested; different memory model |
| Minimum VRAM | — | — | — | unknown | **not measured** — do not assume a number |

## Known platform-specific findings

- **>14k ViT instability** (transformers/ROCm SDPA path): a sharp threshold
  (~14,200 patches) on the tested gfx1100 stack; not pinned to a single SDPA
  kernel, no NVIDIA control. The llama.cpp C++ ViT is deterministic and uncapped.
  See [ROCm/ROCm#6416](https://github.com/ROCm/ROCm/issues/6416).
- **rocWMMA**: the llama.cpp build uses rocWMMA FATTN on gfx1100; fall back to a
  non-rocWMMA build (drop `-DGGML_HIP_ROCWMMA_FATTN=ON`) if your stack lacks the
  headers.

If you verify a new combination, open an issue with the row + evidence and we will
promote it to `verified`.
