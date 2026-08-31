"""Phase 4/6 probe against the modified build (env-rdna3 / vllm-rdna3).

Uses the REAL Muse layer-0 q_proj weights (K=6656, N=4096, group 128) and a
fixed seeded bf16 input, exercising three ops on identical inputs:
  gptq_gemm_rdna3              production atomic path (unchanged semantics)
  gptq_gemm_rdna3_partials     per-split FP32 partials [Z, M, N]
  gptq_gemm_rdna3_deterministic FP32 scratch + fixed-order reduce

Mode A (partials): capture partials repeatedly; test partial determinism,
R1/R2/R3 offline fixed-order reductions, and compare vs atomic outputs.
Mode P (probe): repeated end-to-end op calls, one JSON line per op.

Usage: probe_rdna3.py <mode: partials|probe> <tag> <repeats>
"""
import hashlib
import json
import os
import sys

import torch
from safetensors import safe_open

SNAP = "/workspace/vllm-50603-version-ab/models/muse"
BASE = "model.language_model.layers.0.self_attn.q_proj"
WEIGHT_TYPE = None  # set in main after import


def sha(t):
    return hashlib.sha256(
        t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def build_layer(q_int4_kn, scales_gn, dtype):
    no_loader = lambda *a, **k: None  # noqa: E731
    qweight = pack_quantized_values_into_int32(q_int4_kn, WEIGHT_TYPE, packed_dim=0)

    class DummyLayer(torch.nn.Module):
        pass

    layer = DummyLayer()
    layer.register_parameter("qweight", PackedvLLMParameter(
        data=qweight, weight_loader=no_loader, input_dim=0, output_dim=1,
        packed_dim=0, packed_factor=8))
    layer.register_parameter("scales", GroupQuantScaleParameter(
        data=scales_gn.to(dtype), weight_loader=no_loader, input_dim=0,
        output_dim=1))
    return layer


def main():
    mode, tag, repeats = sys.argv[1], sys.argv[2], int(sys.argv[3])

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.scalar_type import scalar_types
    _cm = set_current_vllm_config(VllmConfig())
    _cm.__enter__()
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29713")
    from vllm.distributed import (init_distributed_environment,
                                  initialize_model_parallel)
    init_distributed_environment(backend="cpu:gloo,cuda:hccl", world_size=1,
                                 rank=0, local_rank=0,
                                 distributed_init_method="env://")
    initialize_model_parallel(tensor_model_parallel_size=1)

    global WEIGHT_TYPE
    WEIGHT_TYPE = scalar_types.uint4b8
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
    globals().update(MPLinearLayerConfig=MPLinearLayerConfig,
                     RDNA3W4A16LinearKernel=RDNA3W4A16LinearKernel,
                     pack_quantized_values_into_int32=pack_quantized_values_into_int32,
                     GroupQuantScaleParameter=GroupQuantScaleParameter,
                     PackedvLLMParameter=PackedvLLMParameter)

    import json as _json
    idx = _json.load(open(f"{SNAP}/model.safetensors.index.json"))
    fname = idx["weight_map"][f"{BASE}.weight_packed"]
    with safe_open(f"{SNAP}/{fname}", framework="pt", device="cpu") as f:
        packed = f.get_tensor(f"{BASE}.weight_packed")   # int32 [N, K/8]
        scale = f.get_tensor(f"{BASE}.weight_scale")     # bf16 [N, K/128]
        N, K = f.get_tensor(f"{BASE}.weight_shape").tolist()
    q_nk = torch.stack([(packed >> (4 * i)) & 0xF for i in range(8)], dim=2)
    q_nk = q_nk.reshape(N, K).to(torch.int32)
    q_int4_kn = q_nk.t().contiguous()
    scales_gn = scale.t().contiguous()

    torch.manual_seed(20260831)
    x = torch.randn(1, K, device="cuda", dtype=torch.bfloat16)

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

    ops = torch.ops._rocm_C

    if mode == "partials":
        # 1) partial determinism (layer params via the kernel's own accessor;
        # the RDNA3 layer synthesizes qzeros internally during
        # process_weights_after_loading)
        w_q, w_s, w_zp, w_g_idx = kernel._get_weight_params(layer)
        phashes = []
        for _ in range(repeats):
            p = ops.gptq_gemm_rdna3_partials(x, w_q, w_zp, w_s, w_g_idx, False)
            torch.cuda.synchronize()
            phashes.append(sha(p))
        pd = len(set(phashes))
        P = p  # last captured partials [Z, M, N] fp32
        Z = P.shape[0]
        print(json.dumps({"tag": tag, "what": "partials", "repeats": repeats,
                          "distinct": pd, "Z": Z}), flush=True)

        # 2) R1/R2/R3 fixed-order reductions from the SAME captured partials
        r1 = P.sum(dim=0)                       # ascending z (torch sum = pairwise! do manual)
        r1m = P[0].clone()
        for z in range(1, Z):
            r1m += P[z]
        r2m = P[Z - 1].clone()
        for z in range(Z - 2, -1, -1):
            r2m += P[z]
        g = torch.Generator(device="cpu").manual_seed(7)
        perm = torch.randperm(Z, generator=g)
        r3m = P[perm[0]].clone()
        for i in range(1, Z):
            r3m += P[perm[i]]
        b1 = r1m.to(torch.bfloat16)
        b2 = r2m.to(torch.bfloat16)
        b3 = r3m.to(torch.bfloat16)
        print(json.dumps({"tag": tag, "what": "reductions",
                          "r1_sha": sha(b1), "r2_sha": sha(b2), "r3_sha": sha(b3),
                          "r1_eq_r2": bool(torch.equal(b1, b2)),
                          "r1_eq_r3": bool(torch.equal(b1, b3)),
                          "r2_eq_r3": bool(torch.equal(b2, b3)),
                          "perm": [int(v) for v in perm.tolist()]}), flush=True)

        # 3) 30 production atomic outputs + their distance to R1
        atomics, ahashes = [], []
        for _ in range(30):
            o = ops.gptq_gemm_rdna3(x, w_q, w_zp, w_s, w_g_idx, False)
            torch.cuda.synchronize()
            atomics.append(o.clone())
            ahashes.append(sha(o))
        d_r1 = torch.stack([((a.float() - b1.float()).abs().max()) for a in atomics])
        a01 = (atomics[0].float() - atomics[1].float()).abs().max()
        print(json.dumps({"tag": tag, "what": "atomic_vs_R1",
                          "atomic_distinct_30": len(set(ahashes)),
                          "max_abs_atomic_vs_R1": float(d_r1.max()),
                          "median_abs_atomic_vs_R1": float(d_r1.median()),
                          "max_abs_atomic0_vs_atomic1": float(a01)}), flush=True)

    elif mode == "correctness":
        w_q, w_s, w_zp, w_g_idx = kernel._get_weight_params(layer)
        # decode the synthesized qzeros: [groups, N/8] uint32, 8 nibbles per word
        z0 = w_zp[0, :2].tolist()
        zeros_stored = {(z >> (4 * i)) & 0xF for z in z0 for i in range(8)}
        # dequant reference: value = (nibble - stored_zero - zero_offset) * scale
        # with zero_offset = 1 (use_v2_format=False)
        W = (q_int4_kn.float() - (sorted(zeros_stored)[0] + 1)) * scales_gn.float().repeat_interleave(128, dim=0)
        ref = (x.float() @ W.cuda()).squeeze(0)          # fp32 reference [N]
        outs = {}
        outs["deterministic"] = ops.gptq_gemm_rdna3_deterministic(x, w_q, w_zp, w_s, w_g_idx, False).float().squeeze(0)
        outs["atomic_run0"] = ops.gptq_gemm_rdna3(x, w_q, w_zp, w_s, w_g_idx, False).float().squeeze(0)
        outs["atomic_run1"] = ops.gptq_gemm_rdna3(x, w_q, w_zp, w_s, w_g_idx, False).float().squeeze(0)
        rec = {"tag": tag, "zeros_stored_unique": sorted(zeros_stored)}
        for name, o in outs.items():
            d = (o - ref).abs()
            denom = ref.abs().clamp_min(1e-6)
            rec[name] = {
                "max_abs_err": float(d.max()),
                "mean_abs_err": float(d.mean()),
                "max_rel_err": float((d / denom).max()),
                "cosine": float(torch.nn.functional.cosine_similarity(
                    o.unsqueeze(0), ref.unsqueeze(0))),
            }
        print(json.dumps(rec), flush=True)

    elif mode == "probe":
        w_q, w_s, w_zp, w_g_idx = kernel._get_weight_params(layer)
        for name, op in (("atomic", ops.gptq_gemm_rdna3),
                         ("deterministic", ops.gptq_gemm_rdna3_deterministic)):
            hashes, maxerr = [], 0.0
            for _ in range(repeats):
                y = op(x, w_q, w_zp, w_s, w_g_idx, False)
                torch.cuda.synchronize()
                hashes.append(sha(y))
                if len(hashes) > 1:
                    maxerr = max(maxerr, float((y.float() - first.float()).abs().max()))
                else:
                    first = y.clone()
            print(json.dumps({"tag": tag, "op": name, "M": 1,
                              "repeats": repeats, "distinct": len(set(hashes)),
                              "max_abs_err_vs_first": maxerr,
                              "sha_first": hashes[0]}), flush=True)


if __name__ == "__main__":
    main()
