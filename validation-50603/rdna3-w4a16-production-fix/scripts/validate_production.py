"""Phase 10-11 validation matrices on the production build.

Real Muse q_proj weights (K=6656, N=4096, group 128, bf16).
For M in {1, 8, 16, 64, 128, 512}:
  - latency (median of 200, 50 warmup) of the public op (production dispatch)
  - bit-repeatability: distinct outputs over 100 calls
  - correctness vs fp32 dequant reference (max/mean abs err, cosine, finite)
Run once with defaults (deterministic) and once with
VLLM_RDNA3_W4A16_LEGACY_ATOMIC=1 to capture the legacy arms.
Usage: validate_production.py <tag>
"""
import hashlib
import json
import math
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
    tag = sys.argv[1]
    legacy = os.environ.get("VLLM_RDNA3_W4A16_LEGACY_ATOMIC") == "1"
    from vllm.config import VllmConfig, set_current_vllm_config
    _cm = set_current_vllm_config(VllmConfig()); _cm.__enter__()
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29735")
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

    for M in (1, 8, 16, 64, 128, 512):
        torch.manual_seed(20260831)
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        op = lambda t: ops.gptq_gemm_rdna3(t, w_q, w_zp, w_s, w_g_idx, False)
        for _ in range(50):
            op(x)
        torch.cuda.synchronize()
        ts = []
        for _ in range(200):
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record(); op(x); e1.record()
            torch.cuda.synchronize(); ts.append(e0.elapsed_time(e1))
        ts.sort()
        hs = []
        for _ in range(100):
            y = op(x); torch.cuda.synchronize(); hs.append(sha(y))
        y0 = op(x).float()
        W = (q_int4_kn.float() - 8) * scales_gn.float().repeat_interleave(128, dim=0)
        ref = (x.float() @ W.cuda())
        d = (y0 - ref).abs()
        rec = {
            "tag": tag, "legacy_atomic": legacy, "M": M, "K": K, "N": N,
            "median_ms": ts[len(ts)//2], "p10_ms": ts[20], "p90_ms": ts[179],
            "distinct100": len(set(hs)),
            "max_abs_err": float(d.max()), "mean_abs_err": float(d.mean()),
            "cosine": float(torch.nn.functional.cosine_similarity(
                y0.flatten(), ref.flatten(), dim=0)),
            "finite": bool(torch.isfinite(y0).all()),
        }
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
