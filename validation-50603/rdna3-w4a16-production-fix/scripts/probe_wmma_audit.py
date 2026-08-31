"""Phase 1: WMMA split-K determinism audit on the W7900D.

Real Muse layer-0 q_proj weights (K=6656, N=4096, group 128, bf16) at
M in {16,32,64,128,512} via the public op (M>=16 bf16 -> WMMA path),
plus a forced-K_SPLIT=1 control (synthetic K=256 layer, K-only heuristic
returns 1). 100 repeated fixed-input calls per shape.
Usage: probe_wmma_audit.py <tag>
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


# Replication of the host heuristics (q_gemm_rdna3_wmma.cu) for evidence:
def compute_wmma_k_split(size_k):
    if size_k >= 1024 and size_k % 64 == 0:
        return 4
    if size_k >= 512 and size_k % 32 == 0:
        return 2
    return 1


def compute_wmma_k_split_mn(size_m, size_n, size_k, m_tile, n_tile):
    blocks_xy = ((size_n + n_tile - 1) // n_tile) * ((size_m + m_tile - 1) // m_tile)
    if blocks_xy >= 1500:
        return 1
    if blocks_xy * 2 >= 1500 and size_k >= 512 and size_k % 32 == 0:
        return 2
    if blocks_xy * 4 >= 1500 and size_k >= 1024 and size_k % 64 == 0:
        return 4
    return compute_wmma_k_split(size_k)


def main():
    tag = sys.argv[1]
    from vllm.config import VllmConfig, set_current_vllm_config
    _cm = set_current_vllm_config(VllmConfig()); _cm.__enter__()
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29731")
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

    def real_layer():
        idx = json.load(open(f"{SNAP}/model.safetensors.index.json"))
        with safe_open(f"{SNAP}/{idx['weight_map'][f'{BASE}.weight_packed']}",
                       framework="pt", device="cpu") as f:
            packed = f.get_tensor(f"{BASE}.weight_packed")
            scale = f.get_tensor(f"{BASE}.weight_scale")
            N, K = f.get_tensor(f"{BASE}.weight_shape").tolist()
        q_nk = torch.stack([(packed >> (4 * i)) & 0xF for i in range(8)],
                           dim=2).reshape(N, K).to(torch.int32)
        return q_nk.t().contiguous(), scale.t().contiguous(), K, N

    def synth_layer(K, N):
        torch.manual_seed(1000 + K)
        q = torch.randint(0, 16, (K, N), dtype=torch.int32)
        sc = (torch.randn(K // 128, N) * 0.01 + 0.02).to(torch.bfloat16)
        return q, sc, K, N

    def build(q_int4_kn, scales_gn, K, N):
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
        return k, layer

    # variant tiles per the WMMA launchers (v3 64x16, v4 64x32, v5 64x64)
    def tiles_for(M):
        if M <= 64: return (64, 16)
        if M <= 128: return (64, 32)
        return (64, 64)

    real_q, real_s, K, N = real_layer()
    cases = [(M, "real", real_q, real_s, K, N) for M in (16, 32, 64, 128, 512)]
    sq, ss, sK, sN = synth_layer(256, 4096)
    cases.append((128, "synth_K256_KSPLIT1_control", sq, ss, sK, sN))

    for M, wname, q, sc, K, N in cases:
        k, layer = build(q, sc, K, N)
        w_q, w_s, w_zp, w_g_idx = k._get_weight_params(layer)
        mt, nt = tiles_for(M)
        ksplit = compute_wmma_k_split_mn(M, N, K, mt, nt)
        torch.manual_seed(20260831)
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        hashes, maxd = [], 0.0
        for _ in range(100):
            y = ops.gptq_gemm_rdna3(x, w_q, w_zp, w_s, w_g_idx, False)
            torch.cuda.synchronize()
            hashes.append(sha(y))
            if len(hashes) > 1:
                maxd = max(maxd, float((y.float() - first.float()).abs().max()))
            else:
                first = y.clone()
        rec = {"tag": tag, "M": M, "weights": wname, "K": K, "N": N,
               "m_tile": mt, "n_tile": nt, "computed_K_SPLIT": ksplit,
               "repeats": 100, "distinct": len(set(hashes)),
               "max_abs_diff": maxd, "sha_first": hashes[0]}
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
