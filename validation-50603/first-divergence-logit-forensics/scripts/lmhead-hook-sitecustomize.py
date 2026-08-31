"""Forensics hook (gated), injected via PYTHONPATH prepend (shadows the
Debian /usr/lib/python3.12/sitecustomize.py because PYTHONPATH precedes the
stdlib in sys.path).

When FORENSICS_LMHEAD_HOOK=1, installs a lazy import hook: when (and only
when) a process actually imports vllm.model_executor.layers.logits_processor,
LogitsProcessor._get_logits is wrapped to record per call the LM-head input
hidden-states sha256 and the output logits sha256. The eager-import pitfall
(cpuinfo's helper subprocess parses stdout as JSON) is why this is lazy:
processes that never import vLLM are untouched. Identity wrapper; no vLLM
source file is modified; inert without the env var.
"""
import builtins
import torch
import hashlib
import json
import os

_TARGET = "vllm.model_executor.layers.logits_processor"

if os.environ.get("FORENSICS_LMHEAD_HOOK") == "1":
    _OUT = os.environ["FORENSICS_LMHEAD_OUT"]
    _odir = os.path.dirname(_OUT)
    if _odir:
        os.makedirs(_odir, exist_ok=True)
    _F = open(_OUT, "a", buffering=1)
    _N = [0]

    def _sha(t):
        # uint8 byte view: works for bf16 (numpy-unsupported) and fp32 alike
        b = t.detach().contiguous().cpu().view(torch.uint8)
        return hashlib.sha256(b.numpy().tobytes()).hexdigest()

    def _patch(mod):
        LP = mod.LogitsProcessor
        if getattr(LP._get_logits, "_forensics_wrapped", False):
            return
        _orig = LP._get_logits

        def _wrapped(self, hidden_states, lm_head, embedding_bias=None):
            h_meta = (tuple(hidden_states.shape), str(hidden_states.dtype),
                      _sha(hidden_states))
            out = _orig(self, hidden_states, lm_head, embedding_bias)
            _N[0] += 1
            _F.write(json.dumps({
                "call": _N[0],
                "hidden_shape": list(h_meta[0]), "hidden_dtype": h_meta[1],
                "hidden_sha256": h_meta[2],
                "logits_shape": list(out.shape), "logits_dtype": str(out.dtype),
                "logits_sha256": _sha(out),
            }) + "\n")
            return out

        _wrapped._forensics_wrapped = True
        LP._get_logits = _wrapped

    _orig_import = builtins.__import__

    def _import_hook(name, *args, **kwargs):
        mod = _orig_import(name, *args, **kwargs)
        if name == _TARGET or name.startswith(_TARGET + "."):
            import sys
            m = sys.modules.get(_TARGET)
            if m is not None:
                _patch(m)
                builtins.__import__ = _orig_import  # one-shot uninstall
        return mod

    builtins.__import__ = _import_hook
