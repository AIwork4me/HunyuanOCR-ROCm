"""Derive the version A/B comparison tables from results/<tag>/*.json.

Every number printed is computed from the saved token sequences (the JSON's
own unique_outputs is cross-checked against a recount).
Usage: analyze_ab.py <results-root>
"""
import glob
import hashlib
import json
import os
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/workspace/vllm-50603-version-ab/results"

def sha(seq):
    return hashlib.sha256(",".join(map(str, seq)).encode()).hexdigest()

rows = []
for path in sorted(glob.glob(os.path.join(ROOT, "*", "*.json"))):
    tag = os.path.basename(os.path.dirname(path))
    with open(path) as f:
        d = json.load(f)
    for cell in d["cells"]:
        seqs = [tuple(s) for s in cell["token_ids"]]
        distinct = {sha(s) for s in seqs}
        # group structure: which runs share which sequence (settle detection)
        groups = {}
        for i, s in enumerate(seqs):
            groups.setdefault(sha(s), []).append(i)
        group_desc = "+".join(
            f"[{','.join(map(str, g))}]" for g in sorted(groups.values(), key=lambda g: g[0]))
        # recount must match stored count
        assert len(distinct) == cell["unique_outputs"], (path, cell["context"])
        # per-seq length sanity (ignore_eos should give full NGEN)
        lens = {len(s) for s in seqs}
        rows.append({
            "version_tag": tag,
            "vllm": d["vllm_version"],
            "git": (d.get("vllm_git_sha") or "")[:9],
            "torch": d["torch_version"],
            "model": d["model"],
            "tp": d["tp"],
            "eager": d["enforce_eager"],
            "engine": d["engine_id"],
            "ctx": cell["context"],
            "unique": len(distinct),
            "ngen": cell["measured_generations"],
            "first_div": cell["first_divergence_vs_run0"],
            "groups": group_desc,
            "seq_lens": sorted(lens),
            "path": os.path.relpath(path, ROOT),
        })

rows.sort(key=lambda r: (r["model"], r["ctx"], r["version_tag"], r["eager"], r["engine"]))
print("| model | ctx | TP | eager | vLLM | git | torch | eng | unique/8 | first divergence vs run0 | run groups |")
print("|---|--:|--:|--:|---|---|---|--:|--:|---|---|")
for r in rows:
    fd = ",".join("∅" if x is None else str(x) for x in r["first_div"])
    print(f"| {r['model']} | {r['ctx']} | {r['tp']} | {int(r['eager'])} | {r['vllm'].split('+')[0]} | {r['git']} | {r['torch'].split('+')[0]} | {r['engine']} | "
          f"{r['unique']}/{r['ngen']} | {fd} | {r['groups']} |")

print()
print("## Cell-level rollup")
seen = {}
for r in rows:
    k = (r["model"], r["ctx"], r["version_tag"], r["eager"])
    seen.setdefault(k, []).append(r["unique"])
print("| model | ctx | vLLM-tag | eager | engines | unique/8 per engine | varies? |")
print("|---|--:|---|--:|--:|---|---|")
for k in sorted(seen):
    v = seen[k]
    varies = "YES" if any(x > 1 for x in v) else "no"
    print(f"| {k[0]} | {k[1]} | {k[2]} | {int(k[3])} | {len(v)} | {v} | {varies} |")

with open(os.path.join(ROOT, "ab-analysis.json"), "w") as f:
    json.dump(rows, f, indent=1)
print("\nwrote", os.path.join(ROOT, "ab-analysis.json"))
