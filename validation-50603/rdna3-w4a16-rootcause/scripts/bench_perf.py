"""Phase 11: latency microbench, atomic vs deterministic, real q_proj weights.
M=1 (decode) and M=512 (prefill-tail). CUDA events, 50 warmup + 200 timed."""
import json, os, sys
import torch
from safetensors import safe_open

def main():
    from vllm.config import VllmConfig, set_current_vllm_config
    _cm = set_current_vllm_config(VllmConfig()); _cm.__enter__()
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1"); os.environ.setdefault("MASTER_PORT", "29719")
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    init_distributed_environment(backend="cpu:gloo,cuda:hccl", world_size=1, rank=0,
                                 local_rank=0, distributed_init_method="env://")
    initialize_model_parallel(tensor_model_parallel_size=1)
    from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import MPLinearLayerConfig
    from vllm.model_executor.kernels.linear.mixed_precision.rdna3_w4a16 import RDNA3W4A16LinearKernel
    from vllm.model_executor.layers.quantization.utils.quant_utils import pack_quantized_values_into_int32
    from vllm.model_executor.parameter import GroupQuantScaleParameter, PackedvLLMParameter
    from vllm.scalar_type import scalar_types

    SNAP = "/workspace/vllm-50603-version-ab/models/muse"
    BASE = "model.language_model.layers.0.self_attn.q_proj"
    idx = json.load(open(f"{SNAP}/model.safetensors.index.json"))
    with safe_open(f"{SNAP}/{idx['weight_map'][f'{BASE}.weight_packed']}", framework="pt", device="cpu") as f:
        packed = f.get_tensor(f"{BASE}.weight_packed"); scale = f.get_tensor(f"{BASE}.weight_scale")
        N, K = f.get_tensor(f"{BASE}.weight_shape").tolist()
    q_nk = torch.stack([(packed >> (4*i)) & 0xF for i in range(8)], dim=2).reshape(N, K).to(torch.int32)
    q_int4_kn = q_nk.t().contiguous(); scales_gn = scale.t().contiguous()
    no_loader = lambda *a, **k: None
    layer0 = type("L", (torch.nn.Module,), {})()
    qw = pack_quantized_values_into_int32(q_int4_kn, scalar_types.uint4b8, packed_dim=0)
    layer0.register_parameter("qweight", PackedvLLMParameter(data=qw, weight_loader=no_loader,
        input_dim=0, output_dim=1, packed_dim=0, packed_factor=8))
    layer0.register_parameter("scales", GroupQuantScaleParameter(data=scales_gn.to(torch.bfloat16),
        weight_loader=no_loader, input_dim=0, output_dim=1))
    layer = layer0.cuda()
    cfg = MPLinearLayerConfig(full_weight_shape=(K, N), partition_weight_shape=(K, N),
        weight_type=scalar_types.uint4b8, act_type=torch.bfloat16, group_size=128,
        zero_points=False, has_g_idx=False)
    k = RDNA3W4A16LinearKernel(cfg, "qweight", "scales", None, None)
    k.process_weights_after_loading(layer)
    w_q, w_s, w_zp, w_g_idx = k._get_weight_params(layer)
    ops = torch.ops._rocm_C
    out = []
    for M in (1, 512):
        torch.manual_seed(20260831)
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        for name, op in (("atomic", ops.gptq_gemm_rdna3), ("deterministic", ops.gptq_gemm_rdna3_deterministic)):
            for _ in range(50): op(x, w_q, w_zp, w_s, w_g_idx, False)
            torch.cuda.synchronize()
            ts = []
            for _ in range(200):
                e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                e0.record(); op(x, w_q, w_zp, w_s, w_g_idx, False); e1.record()
                torch.cuda.synchronize(); ts.append(e0.elapsed_time(e1))
            ts.sort()
            out.append({"M": M, "op": name, "median_ms": ts[len(ts)//2],
                        "p10_ms": ts[20], "p90_ms": ts[179], "min_ms": ts[0]})
            print(json.dumps(out[-1]), flush=True)
if __name__ == "__main__":
    main()
