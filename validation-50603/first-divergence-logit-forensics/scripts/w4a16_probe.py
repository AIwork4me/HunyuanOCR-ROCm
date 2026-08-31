"""Phase 10 micro-probe: is gptq_gemm_rdna3 (RDNA3W4A16LinearKernel) bit-
repeatable on fixed inputs at the real Muse decode-step shape?

Uses the REAL layer-0 q_proj weights (K=6656, N=4096, group 128) from the
pinned Muse INT4 checkpoint and the layer-building recipe from vLLM's own
tests/kernels/quantization/test_rdna3_w4a16.py (GPTQ int32 packing +
process_weights_after_loading shuffle). Fixed seeded bf16 activations at
M=1 (decode shape) and M=128 (prefill batch).

Usage: w4a16_probe.py <tag> <repeats>   -> prints JSON lines
"""
import hashlib
import json
import sys

import torch
from safetensors import safe_open

from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
    MPLinearLayerConfig,
)
from vllm.model_executor.kernels.linear.mixed_precision.rdna3_w4a16 import (
    RDNA3W4A16LinearKernel,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_quantized_values_into_int32,
)
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    PackedvLLMParameter,
)
from vllm.scalar_type import scalar_types

SNAP = "/workspace/vllm-50603-version-ab/models/muse"
BASE = "model.language_model.layers.0.self_attn.q_proj"
WEIGHT_TYPE = scalar_types.uint4b8
PACK_FACTOR = 8

def build_layer(q_int4_kn, scales_gn, dtype):
    no_loader = lambda *a, **k: None  # noqa: E731
    qweight = pack_quantized_values_into_int32(q_int4_kn, WEIGHT_TYPE, packed_dim=0)

    class DummyLayer(torch.nn.Module):
        pass

    layer = DummyLayer()
    layer.register_parameter("qweight", PackedvLLMParameter(
        data=qweight, weight_loader=no_loader, input_dim=0, output_dim=1,
        packed_dim=0, packed_factor=PACK_FACTOR))
    layer.register_parameter("scales", GroupQuantScaleParameter(
        data=scales_gn.to(dtype), weight_loader=no_loader,
        input_dim=0, output_dim=1))
    return layer

def main():
    tag, repeats = sys.argv[1], int(sys.argv[2])

    # single-rank distributed init (kernel init queries the TP group)
    import os
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29711")
    os.environ.setdefault("RANK", "0")
    from vllm.config import VllmConfig, set_current_vllm_config
    _cfg_cm = set_current_vllm_config(VllmConfig())
    _cfg_cm.__enter__()

    from vllm.distributed import (init_distributed_environment,
                                  initialize_model_parallel)
    init_distributed_environment(backend="cpu:gloo,cuda:hccl",
                                 world_size=1, rank=0,
                                 local_rank=0, distributed_init_method="env://")
    initialize_model_parallel(tensor_model_parallel_size=1)

    # real weights from the checkpoint shard
    shard = None
    import json as _json
    idx = _json.load(open(f"{SNAP}/model.safetensors.index.json"))
    fname = idx["weight_map"][f"{BASE}.weight_packed"]
    with safe_open(f"{SNAP}/{fname}", framework="pt", device="cpu") as f:
        packed = f.get_tensor(f"{BASE}.weight_packed")     # uint8 [N, K/2]
        scale = f.get_tensor(f"{BASE}.weight_scale")       # [N, K/128]
        wshape = f.get_tensor(f"{BASE}.weight_shape").tolist()
    N, K = wshape
    assert (N, K) == (4096, 6656), (N, K)
    # checkpoint: int32 [N, K/8], 8 nibbles per int32 along K -> [N, K] -> [K, N]
    assert packed.dtype == torch.int32 and packed.shape == (N, K // 8), packed.shape
    q_nk = torch.stack([(packed >> (4 * i)) & 0xF for i in range(8)], dim=2)
    q_nk = q_nk.reshape(N, K).to(torch.int32)
    q_int4_kn = q_nk.t().contiguous()                      # [K, N]
    scales_gn = scale.t().contiguous()                     # [K/128, N]

    torch.manual_seed(20260831)
    results = {}
    for M in (1, 128):
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        layer = build_layer(q_int4_kn, scales_gn, torch.bfloat16).cuda()
        cfg = MPLinearLayerConfig(
            full_weight_shape=(K, N), partition_weight_shape=(K, N),
            weight_type=WEIGHT_TYPE, act_type=torch.bfloat16,
            group_size=128, zero_points=False, has_g_idx=False)
        kernel = RDNA3W4A16LinearKernel(
            cfg, w_q_param_name="qweight", w_s_param_name="scales",
            w_zp_param_name=None, w_gidx_param_name=None)
        kernel.process_weights_after_loading(layer)
        torch.cuda.synchronize()

        outs, hashes = [], []
        for _ in range(repeats):
            y = kernel.apply_weights(layer, x)
            torch.cuda.synchronize()
            b = y.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
            hashes.append(hashlib.sha256(b).hexdigest())
            outs.append(y.detach().clone())
        ref = outs[0].float()
        maxerr = max(float((o.float() - ref).abs().max()) for o in outs[1:])
        results[f"M{M}"] = {
            "distinct_outputs": len(set(hashes)),
            "max_abs_err_vs_first": maxerr,
            "first_sha": hashes[0],
        }
        print(json.dumps({"tag": tag, "M": M, "repeats": repeats,
                          "distinct": len(set(hashes)),
                          "max_abs_err_vs_first": maxerr,
                          "sha_first": hashes[0],
                          "shape": [M, N]}), flush=True)

if __name__ == "__main__":
    main()
