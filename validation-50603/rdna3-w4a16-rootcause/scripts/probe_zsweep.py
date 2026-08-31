"""Phase 9: does Z (number of split-K blocks) gate the nondeterminism?

Uses the UNMODIFIED production op gptq_gemm_rdna3 on synthetic GPTQ layers
(N=4096, group 128, symmetric stored-zero=7) with K swept so that
Z = ceil(K/256) takes 1, 2, 4, 8, 26. Fixed seeded bf16 input, M=1.
If the mechanism is concurrent multi-writer accumulation, Z=1 (single writer,
plain CAS add onto zero) must be bit-repeatable and Z>=2 must not.
Usage: probe_zsweep.py <tag> <repeats>
"""
import hashlib
import json
import os
import sys

import torch


def sha(t):
    return hashlib.sha256(
        t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def main():
    tag, repeats = sys.argv[1], int(sys.argv[2])
    from vllm.config import VllmConfig, set_current_vllm_config
    _cm = set_current_vllm_config(VllmConfig())
    _cm.__enter__()
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29717")
    from vllm.distributed import (init_distributed_environment,
                                  initialize_model_parallel)
    init_distributed_environment(backend="cpu:gloo,cuda:hccl", world_size=1,
                                 rank=0, local_rank=0,
                                 distributed_init_method="env://")
    initialize_model_parallel(tensor_model_parallel_size=1)

    from vllm.model_executor.kernels.linear.mixed_precision.rdna3_w4a16 import (
        RDNA3W4A16LinearKernel,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        pack_quantized_values_into_int32,
    )
    from vllm.model_executor.parameter import (GroupQuantScaleParameter,
                                               PackedvLLMParameter)
    from vllm.scalar_type import scalar_types

    WEIGHT_TYPE = scalar_types.uint4b8
    N = 4096
    G = 128

    def build_layer(K):
        torch.manual_seed(1000 + K)          # per-K deterministic weights
        q = torch.randint(0, 16, (K, N), dtype=torch.int32)
        scales = (torch.randn(K // G, N) * 0.01 + 0.02).to(torch.bfloat16)
        qweight = pack_quantized_values_into_int32(q, WEIGHT_TYPE, packed_dim=0)
        no_loader = lambda *a, **k: None  # noqa: E731

        class L(torch.nn.Module):
            pass

        layer = L()
        layer.register_parameter("qweight", PackedvLLMParameter(
            data=qweight, weight_loader=no_loader, input_dim=0, output_dim=1,
            packed_dim=0, packed_factor=8))
        layer.register_parameter("scales", GroupQuantScaleParameter(
            data=scales, weight_loader=no_loader, input_dim=0, output_dim=1))
        return layer

    ops = torch.ops._rocm_C
    for K in (256, 512, 1024, 2048, 6656):
        Z = (K + 255) // 256
        layer = build_layer(K).cuda()
        cfg = __import__(
            "vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel",
            fromlist=["MPLinearLayerConfig"]).MPLinearLayerConfig(
            full_weight_shape=(K, N), partition_weight_shape=(K, N),
            weight_type=WEIGHT_TYPE, act_type=torch.bfloat16,
            group_size=G, zero_points=False, has_g_idx=False)
        kernel = RDNA3W4A16LinearKernel(
            cfg, w_q_param_name="qweight", w_s_param_name="scales",
            w_zp_param_name=None, w_gidx_param_name=None)
        kernel.process_weights_after_loading(layer)
        w_q, w_s, w_zp, w_g_idx = kernel._get_weight_params(layer)
        torch.manual_seed(20260831)
        x = torch.randn(1, K, device="cuda", dtype=torch.bfloat16)
        hashes = []
        for _ in range(repeats):
            y = ops.gptq_gemm_rdna3(x, w_q, w_zp, w_s, w_g_idx, False)
            torch.cuda.synchronize()
            hashes.append(sha(y))
        print(json.dumps({"tag": tag, "K": K, "Z": Z, "M": 1,
                          "repeats": repeats, "distinct": len(set(hashes)),
                          "op": "gptq_gemm_rdna3(production)"}), flush=True)


if __name__ == "__main__":
    main()
