#!/usr/bin/env python3
# SPDX-LicenseFileCopyrightText: Copyright contributors to the vLLM project
"""Bit-repeatability regression test for the RDNA3 W4A16 split-K epilogue.

Reproduces vllm#50603-class greedy nondeterminism at the kernel level:
with split-K active (Z = K/BLOCK_KN_SIZE >= 4), the production
``gptq_gemm_rdna3`` epilogue atomically accumulates per-split bf16 partials
into the output via a CAS loop, so the bf16 reduction order is not fixed and
repeated fixed-input calls differ (observed on gfx1100: Z=1..2 repeatable,
Z>=4 gives a different output almost every call).

The deterministic split-K path (FP32 partials + fixed-order reduction,
``gptq_gemm_rdna3_deterministic``) must be bit-repeatable.

Run: pytest tests/kernels/quantization/test_rdna3_w4a16_determinism.py
"""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("RDNA3 W4A16 kernel is ROCm-only", allow_module_level=True)

from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (  # noqa: E402
    MPLinearLayerConfig,
)
from vllm.model_executor.kernels.linear.mixed_precision.rdna3_w4a16 import (  # noqa: E402
    RDNA3W4A16LinearKernel,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (  # noqa: E402
    pack_quantized_values_into_int32,
)
from vllm.model_executor.parameter import (  # noqa: E402
    GroupQuantScaleParameter,
    PackedvLLMParameter,
)
from vllm.platforms.rocm import on_gfx1100  # noqa: E402
from vllm.scalar_type import scalar_types  # noqa: E402

device = "cuda"
WEIGHT_TYPE = scalar_types.uint4b8

gfx1100_only = pytest.mark.skipif(
    not (
        on_gfx1100()
        and hasattr(torch.ops, "_rocm_C")
        and hasattr(torch.ops._rocm_C, "gptq_gemm_rdna3")
    ),
    reason="requires gfx1100 with the _rocm_C.gptq_gemm_rdna3 op built in",
)

# K chosen so split-K is active and the race is robust on gfx1100:
# BLOCK_KN_SIZE=256 -> Z=4 for K=1024 (measured: Z<=2 repeatable, Z>=4 not).
K, N, GROUP = 1024, 4096, 128
REPEATS = 20


def _build_layer():
    torch.manual_seed(1234)
    q_int4_kn = torch.randint(0, 16, (K, N), dtype=torch.int32)
    scales_gn = (torch.randn(K // GROUP, N) * 0.01 + 0.02).to(torch.bfloat16)
    qweight = pack_quantized_values_into_int32(q_int4_kn, WEIGHT_TYPE, packed_dim=0)
    no_loader = lambda *a, **k: None  # noqa: E731

    class DummyLayer(torch.nn.Module):
        pass

    layer = DummyLayer()
    layer.register_parameter(
        "qweight",
        PackedvLLMParameter(data=qweight, weight_loader=no_loader, input_dim=0,
                            output_dim=1, packed_dim=0, packed_factor=8))
    layer.register_parameter(
        "scales",
        GroupQuantScaleParameter(data=scales_gn, weight_loader=no_loader,
                                 input_dim=0, output_dim=1))
    return layer.to(device)


def _kernel_and_input():
    layer = _build_layer()
    cfg = MPLinearLayerConfig(
        full_weight_shape=(K, N), partition_weight_shape=(K, N),
        weight_type=WEIGHT_TYPE, act_type=torch.bfloat16,
        group_size=GROUP, zero_points=False, has_g_idx=False)
    kernel = RDNA3W4A16LinearKernel(cfg, w_q_param_name="qweight",
                                    w_s_param_name="scales",
                                    w_zp_param_name=None, w_gidx_param_name=None)
    kernel.process_weights_after_loading(layer)
    torch.manual_seed(4321)
    x = torch.randn(1, K, device=device, dtype=torch.bfloat16)
    return kernel, layer, x


@gfx1100_only
def test_rdna3_w4a16_deterministic_splitk_bit_repeatable(dist_init):
    """The deterministic split-K path must be bit-repeatable on fixed inputs."""
    kernel, layer, x = _kernel_and_input()
    w_q, w_s, w_zp, w_g_idx = kernel._get_weight_params(layer)
    outs = [
        torch.ops._rocm_C.gptq_gemm_rdna3_deterministic(
            x, w_q, w_zp, w_s, w_g_idx, False)
        for _ in range(REPEATS)
    ]
    for o in outs[1:]:
        assert torch.equal(o, outs[0]), (
            "deterministic RDNA3 W4A16 split-K path produced differing "
            "outputs for identical inputs")


@gfx1100_only
@pytest.mark.xfail(
    reason="known: CAS-atomic bf16 split-K accumulation is order-dependent "
    "(vllm#50603 root cause); tracked for the legacy epilogue",
    strict=False,
)
def test_rdna3_w4a16_legacy_atomic_splitk_known_nondeterministic(dist_init):
    """Documents the legacy epilogue's nondeterminism (xfail on gfx1100)."""
    kernel, layer, x = _kernel_and_input()
    w_q, w_s, w_zp, w_g_idx = kernel._get_weight_params(layer)
    outs = [
        torch.ops._rocm_C.gptq_gemm_rdna3(x, w_q, w_zp, w_s, w_g_idx, False)
        for _ in range(REPEATS)
    ]
    for o in outs[1:]:
        assert torch.equal(o, outs[0])
