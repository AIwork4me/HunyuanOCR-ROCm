"""Phase 6: offline argmax replay, fresh process, no vLLM import.
Loads saved fp32 divergence-position rows and evaluates determinism."""
import json, sys, hashlib
import torch

probe, forensics_json, out = sys.argv[1], sys.argv[2], sys.argv[3]
d = json.load(open(forensics_json))
res = {"repeats": 20, "platform": torch.__version__, "pairs": []}
for p in d["pairs"]:
    g, D = p["run_A"], p["D"]
    t = torch.load(f"{probe}/gen{g}_fp32.pt", weights_only=True)[D]
    am = [int(t.argmax()) for _ in range(res["repeats"])]           # CPU xN
    am_cpu_stable = len(set(am)) == 1
    vals, ids = torch.topk(t, 2)
    if torch.cuda.is_available():
        tg = t.cuda()
        amg = [int(tg.argmax()) for _ in range(res["repeats"])]     # GPU xN
        am_gpu_stable = len(set(amg)) == 1
        gpu_equals_cpu = amg[0] == am[0]
    else:
        am_gpu_stable, gpu_equals_cpu = None, None
    res["pairs"].append({
        "pair": f"{p['run_A']}v{p['run_B']}", "D": D,
        "saved_sha": hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest(),
        "stored_sha": p["stageB_logits"][f"sha_{'A' if False else 'A'}"],
        "engine_token_A": p["token_A"],
        "cpu_argmax": am[0], "cpu_stable": am_cpu_stable,
        "gpu_argmax": (amg[0] if am_gpu_stable is not None else None),
        "gpu_stable": am_gpu_stable, "gpu_equals_cpu": gpu_equals_cpu,
        "cpu_argmax_matches_engine": am[0] == p["token_A"],
        "margin": float(vals[0] - vals[1]),
    })
ok = all(x["cpu_argmax_matches_engine"] and x["cpu_stable"] for x in res["pairs"])
res["verdict"] = "OFFLINE ARGMAX DETERMINISTIC AND MATCHES vLLM" if ok else "MISMATCH"
json.dump(res, open(out, "w"), indent=1)
print(res["verdict"])
for x in res["pairs"][:6]:
    print(f"  pair {x['pair']} D={x['D']}: cpu={x['cpu_argmax']} engine={x['engine_token_A']} "
          f"stable={x['cpu_stable']} gpu_stable={x['gpu_stable']} margin={x['margin']:.4g}")
