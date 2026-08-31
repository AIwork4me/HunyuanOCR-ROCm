# Production performance — real Muse q_proj (K=6656, N=4096, bf16, W7900D)

Median of 200 timed calls (50 warmup), CUDA events. det = production build defaults (no env var); legacy = same build with VLLM_RDNA3_W4A16_LEGACY_ATOMIC=1.

| M | legacy atomic µs | deterministic µs | delta | det distinct/100 | legacy distinct/100 | det max abs err | legacy max abs err |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 43.1 | 46.4 | +7.8% | 1 | 100 | 0.0061 | 0.0280 |
| 8 | 70.0 | 73.8 | +5.3% | 1 | 100 | 0.0070 | 0.0350 |
| 16 | 99.7 | 101.6 | +1.9% | 1 | 100 | 0.0118 | 0.0367 |
| 64 | 270.2 | 276.4 | +2.3% | 1 | 100 | 0.0140 | 0.0546 |
| 128 | 174.2 | 172.1 | -1.2% | 1 | 1 | 0.0173 | 0.0248 |
| 512 | 528.7 | 534.5 | +1.1% | 1 | 1 | 0.0173 | 0.0385 |

Cosine vs fp32 dequant reference: det 0.9999976–0.9999986 vs legacy 0.9999788–0.9999942 (det better at every M). All outputs finite.

Determinism spot-sweep on the production build, K=1024 (Z=4): M in {2,3,4,5,7,9,15,32} all bit-deterministic (plus M=1,8,16,64,128,512 above).

Reference arms measured earlier on the same box/shape: single-writer Z=1 567.6µs (13.1×, rejected); WMMA forced K_SPLIT=1: M=16 +187%, M=512 +13%; scalar-vs-WMMA crossover between M=64 and M=96 (drives the M>=65 routing threshold).

E2E engine wall (Muse both depths, eager, full harness, 3 engines): legacy atomic 115.4 s; **production build 132.5 s (+14.8%)**; root-cause prototype 171.5 s; Triton fallback 188.3 s. Note the E2E includes the M=8192 profiling/prefill shapes where the deterministic WMMA path tiles; decode-only overhead is the M=1 row.
