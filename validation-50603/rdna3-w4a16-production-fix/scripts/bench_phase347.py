"""Phases 3/4/7 bench: single-writer + Z-sweep (scalar) and WMMA force-K_SPLIT1.

Mode scalar: real q_proj (K=6656,N=4096), M=1 (plus M=4,8 for z=1):
  - stock gptq_gemm_rdna3 (atomic, z=26)
  - gptq_gemm_rdna3_zexp z in {1,2,4,8}
  latency (CUDA events, 50 warmup + 200 timed), distinct/100, correctness vs
  fp32 dequant reference (z=1 only).
Mode wmma: M in {16,32,64,128,512}: latency + distinct/100 with the
  process-wide RDNA3_WMMA_FORCE_KSPLIT1 state (set by the caller).
Usage: bench_phase347.py scalar|wmma <tag>
"""
import hashlib
import json
import os
import sys

import torch
from safetensors import safe_open

SNAP = "/workspace/vllm-50603-version-ab/models/muse"
BASE = "model.language_model.layers.0.self_attn.q_proj"


def sha(t):
    return hashlib.sha256(
        t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def main():
    mode, tag = sys.argv[1], sys.argv[2]
    from vllm.config import VllmConfig, set_current_vllm_config
    _cm = set_current_vllm_config(VllmConfig()); _cm.__enter__()
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29733")
    from vllm.distributed import (init_distributed_environment,
                                  initialize_model_parallel)
    init_distributed_environment(backend="cpu:gloo,cuda:hccl", world_size=1,
                                 rank=0, local_rank=0,
                                 distributed_init_method="env://")
    initialize_model_parallel(tensor_model_parallel_size=1)
    from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
        MPLinearLayerConfig,)
    from vllm.model_executor.kernels.linear.mixed_precision.rdna3_w4a16 import (
        RDNA3W4A16LinearKernel,)
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        pack_quantized_values_into_int32,)
    from vllm.model_executor.parameter import (GroupQuantScaleParameter,
                                               PackedvLLMParameter)
    from vllm.scalar_type import scalar_types
    ops = torch.ops._rocm_C

    idx = json.load(open(f"{SNAP}/model.safetensors.index.json"))
    with safe_open(f"{SNAP}/{idx['weight_map'][f'{BASE}.weight_packed']}",
                   framework="pt", device="cpu") as f:
        packed = f.get_tensor(f"{BASE}.weight_packed")
        scale = f.get_tensor(f"{BASE}.weight_scale")
        N, K = f.get_tensor(f"{BASE}.weight_shape").tolist()
    q_nk = torch.stack([(packed >> (4 * i)) & 0xF for i in range(8)],
                       dim=2).reshape(N, K).to(torch.int32)
    q_int4_kn = q_nk.t().contiguous()
    scales_gn = scale.t().contiguous()
    qw = pack_quantized_values_into_int32(q_int4_kn, scalar_types.uint4b8,
                                          packed_dim=0)
    no_loader = lambda *a, **k: None  # noqa: E731
    layer = type("L", (torch.nn.Module,), {})()
    layer.register_parameter("qweight", PackedvLLMParameter(
        data=qw, weight_loader=no_loader, input_dim=0, output_dim=1,
        packed_dim=0, packed_factor=8))
    layer.register_parameter("scales", GroupQuantScaleParameter(
        data=scales_gn.to(torch.bfloat16), weight_loader=no_loader,
        input_dim=0, output_dim=1))
    layer = layer.cuda()
    cfg = MPLinearLayerConfig(
        full_weight_shape=(K, N), partition_weight_shape=(K, N),
        weight_type=scalar_types.uint4b8, act_type=torch.bfloat16,
        group_size=128, zero_points=False, has_g_idx=False)
    k = RDNA3W4A16LinearKernel(cfg, "qweight", "scales", None, None)
    k.process_weights_after_loading(layer)
    w_q, w_s, w_zp, w_g_idx = k._get_weight_params(layer)

    def bench(op, x, n=200):
        for _ in range(50):
            op(x)
        torch.cuda.synchronize()
        ts = []
        for _ in range(n):
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record(); op(x); e1.record()
            torch.cuda.synchronize(); ts.append(e0.elapsed_time(e1))
        ts.sort()
        return ts[len(ts) // 2], ts[20], ts[179]

    def repeats(op, x, n=100):
        hs = []
        for _ in range(n):
            y = op(x); torch.cuda.synchronize(); hs.append(sha(y))
        return len(set(hs))

    if mode == "scalar":
        for M in (1, 4, 8, 16, 32, 64, 128, 512):
            torch.manual_seed(20260831)
            x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
            rec = {"tag": tag, "mode": "scalar", "M": M, "K": K, "N": N}
            med, p10, p90 = bench(lambda t: ops.gptq_gemm_rdna3(
                t, w_q, w_zp, w_s, w_g_idx, False), x)
            rec["atomic"] = {"median_ms": med, "p10_ms": p10, "p90_ms": p90,
                             "distinct100": repeats(lambda t: ops.gptq_gemm_rdna3(
                                 t, w_q, w_zp, w_s, w_g_idx, False), x)}
            for z in (26,):
                med, p10, p90 = bench(lambda t: ops.gptq_gemm_rdna3_zexp(
                    t, w_q, w_zp, w_s, w_g_idx, False, z), x)
                rec[f"zexp_z{z}"] = {
                    "median_ms": med, "p10_ms": p10, "p90_ms": p90,
                    "distinct100": repeats(lambda t: ops.gptq_gemm_rdna3_zexp(
                        t, w_q, w_zp, w_s, w_g_idx, False, z), x)}
            if M == 1:
                W = (q_int4_kn.float() - 8) * scales_gn.float().repeat_interleave(128, dim=0)
                ref = (x.float() @ W.cuda()).squeeze(0)
                y1 = ops.gptq_gemm_rdna3_zexp(x, w_q, w_zp, w_s, w_g_idx, False, 1).float().squeeze(0)
                d = (y1 - ref).abs()
                rec["z1_correctness"] = {
                    "max_abs_err": float(d.max()), "mean_abs_err": float(d.mean()),
                    "cosine": float(torch.nn.functional.cosine_similarity(
                        y1.unsqueeze(0), ref.unsqueeze(0)))}
            print(json.dumps(rec), flush=True)

    elif mode == "wmma":
        forced = os.environ.get("RDNA3_WMMA_FORCE_KSPLIT1") == "1"
        for M in (16, 32, 64, 128, 512):
            torch.manual_seed(20260831)
            x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
            med, p10, p90 = bench(lambda t: ops.gptq_gemm_rdna3(
                t, w_q, w_zp, w_s, w_g_idx, False), x)
            d = repeats(lambda t: ops.gptq_gemm_rdna3(
                t, w_q, w_zp, w_s, w_g_idx, False), x)
            print(json.dumps({"tag": tag, "mode": "wmma", "forced_ksplit1": forced,
                              "M": M, "K": K, "N": N, "median_ms": med,
                              "p10_ms": p10, "p90_ms": p90, "distinct100": d}),
                  flush=True)


if __name__ == "__main__":
    main()
