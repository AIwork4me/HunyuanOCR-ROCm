"""Analyze forensics probe evidence: first-divergence pairs, logits comparison,
near-tie statistics. Runs OUTSIDE vLLM in a fresh process (imports torch only).

Usage: analyze_forensics.py <probe-dir> <run.json> <out-dir>
"""
import hashlib
import itertools
import json
import os
import sys

import torch

VOCAB_SLICE_NOTE = "fp32 rows as seen by the identity probe = argmax input"


def sha(t):
    return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()


def main():
    probe_dir, run_json, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)
    run = json.load(open(run_json))
    seqs = run["token_ids"]
    G = len(seqs)

    steps = [json.loads(l) for l in open(os.path.join(probe_dir, "steps.jsonl"))]
    steps = [s for s in steps if s["gen"] < G]
    assert len(steps) == G * 64, f"expected {G*64} recorded steps, got {len(steps)}"
    # sanity: every recorded step's own argmax must equal the engine's token
    mismatched = [(s["gen"], s["step"], s["own_cpu_argmax"], seqs[s["gen"]][s["step"]])
                  for s in steps if s["own_cpu_argmax"] != seqs[s["gen"]][s["step"]]]
    print(f"[check] own argmax == engine token at all {len(steps)} steps: "
          f"{'YES' if not mismatched else 'NO ' + str(mismatched[:5])}")

    # load archived tensors lazily per generation
    cache = {}
    def row(g, step):
        if g not in cache:
            cache[g] = torch.load(os.path.join(probe_dir, f"gen{g}_fp32.pt"), weights_only=True)
        return cache[g][step]

    # ---- identity checks + first divergence for every pair ----
    pairs = []
    for a, b in itertools.combinations(range(G), 2):
        ta, tb = seqs[a], seqs[b]
        assert len(ta) == len(tb) == 64
        D = next((k for k in range(64) if ta[k] != tb[k]), None)
        if D is None:
            continue
        pairs.append((a, b, D))

    # aligned identical positions across all runs (same history) -> perturbation scale
    aligned_diffs = []
    for k in range(64):
        cols = [seqs[g][k] for g in range(G)]
        if len(set(cols)) != 1:
            break  # histories identical only up to first global divergence
        base = row(0, k)
        for g in range(1, G):
            d = (row(g, k) - base).abs()
            aligned_diffs.append(float(d.max()))
    print(f"[scale] max cross-run |logit diff| at {len(aligned_diffs)} aligned "
          f"identical-history positions: {max(aligned_diffs) if aligned_diffs else 'n/a'}")

    forensics = []
    for a, b, D in pairs:
        ra, rb = row(a, D), row(b, D)
        wa, wb = seqs[a][D], seqs[b][D]
        hist_identical = seqs[a][:D] == seqs[b][:D]
        equal = torch.equal(ra, rb)
        diff = (ra - rb).abs()
        maxd = float(diff.max()); imax = int(diff.argmax())
        mean_d = float(diff.mean())
        sa, sb = sha(ra), sha(rb)
        v10a, i10a = torch.topk(ra, 10)
        v10b, i10b = torch.topk(rb, 10)
        rank_of_wa_in_b = int((rb > rb[wa]).sum().item())
        rank_of_wb_in_a = int((ra > ra[wb]).sum().item())
        offline = {
            "cpu_argmax_A": int(ra.argmax()), "cpu_argmax_B": int(rb.argmax()),
            "match_engine_A": int(ra.argmax()) == wa, "match_engine_B": int(rb.argmax()) == wb,
        }
        forensics.append({
            "run_A": a, "run_B": b, "D": D, "token_A": wa, "token_B": wb,
            "history_identical_through_Dm1": hist_identical,
            "stageB_logits": {
                "sha_A": sa, "sha_B": sb, "torch_equal": equal,
                "max_abs_diff": maxd, "argmax_of_diff": imax, "mean_abs_diff": mean_d,
                "allclose_1e-8": bool(torch.allclose(ra, rb, atol=1e-8, rtol=1e-8)),
                "allclose_1e-6": bool(torch.allclose(ra, rb, atol=1e-6, rtol=1e-6)),
                "allclose_1e-4": bool(torch.allclose(ra, rb, atol=1e-4, rtol=1e-4)),
                "allclose_1e-2": bool(torch.allclose(ra, rb, atol=1e-2, rtol=1e-2)),
            },
            "top10_A": [[int(i), float(v)] for i, v in zip(i10a, v10a)],
            "top10_B": [[int(i), float(v)] for i, v in zip(i10b, v10b)],
            "logit_tokenA_in_A": float(ra[wa]), "logit_tokenA_in_B": float(rb[wa]),
            "logit_tokenB_in_A": float(ra[wb]), "logit_tokenB_in_B": float(rb[wb]),
            "margin_A": float(v10a[0] - v10a[1]), "margin_B": float(v10b[0] - v10b[1]),
            "competing_gap_A": float(ra[wa] - ra[wb]), "competing_gap_B": float(rb[wa] - rb[wb]),
            "rank_of_winnerA_in_B": rank_of_wa_in_b, "rank_of_winnerB_in_A": rank_of_wb_in_a,
            "offline_argmax": offline,
        })

    with open(os.path.join(out_dir, "first-divergence-forensics.json"), "w") as f:
        json.dump({"pairs": forensics,
                   "aligned_max_diff_max": max(aligned_diffs) if aligned_diffs else None,
                   "aligned_max_diff_median": sorted(aligned_diffs)[len(aligned_diffs)//2] if aligned_diffs else None,
                   "aligned_n": len(aligned_diffs),
                   "argmax_engine_match_all_steps": not mismatched}, f, indent=1)

    # ---- near-tie analysis (Phase 8) ----
    margins = [p["margin_A"] for p in forensics] + [p["margin_B"] for p in forensics]
    def pct_below(ms, t):
        return 100.0 * sum(m < t for m in ms) / len(ms) if ms else None
    lines = [
        "# Near-tie analysis — generated from first-divergence-forensics.json",
        "",
        f"- divergent pairs: {len(forensics)} (of {G} runs, {G*(G-1)//2} possible pairs)",
        f"- first-divergence steps: {[p['D'] for p in forensics]}",
        f"- margins at divergence (both runs pooled, n={len(margins)}): "
        f"min={min(margins):.6g} median={sorted(margins)[len(margins)//2]:.6g} max={max(margins):.6g}",
        f"- margin < 1e-2: {pct_below(margins,1e-2):.1f}%  <1e-3: {pct_below(margins,1e-3):.1f}%  "
        f"<1e-4: {pct_below(margins,1e-4):.1f}%  <1e-5: {pct_below(margins,1e-5):.1f}%",
        f"- reference scale: max cross-run |logit diff| over {len(aligned_diffs)} aligned "
        f"identical-history positions = {max(aligned_diffs) if aligned_diffs else float('nan'):.6g} "
        f"(median {sorted(aligned_diffs)[len(aligned_diffs)//2] if aligned_diffs else float('nan'):.6g})",
        "",
        "| pair | D | tokA | tokB | stageB equal | max abs diff | margin A | margin B | offline argmax matches |",
        "|---|--:|--:|--:|---|--:|--:|--:|---|",
    ]
    for p in forensics:
        e = "YES" if p["stageB_logits"]["torch_equal"] else "no"
        oa = ("A:" + ("✓" if p["offline_argmax"]["match_engine_A"] else "✗") +
              " B:" + ("✓" if p["offline_argmax"]["match_engine_B"] else "✗"))
        lines.append(f"| {p['run_A']}v{p['run_B']} | {p['D']} | {p['token_A']} | {p['token_B']} | {e} "
                     f"| {p['stageB_logits']['max_abs_diff']:.6g} | {p['margin_A']:.6g} | {p['margin_B']:.6g} | {oa} |")
    with open(os.path.join(out_dir, "near-tie-analysis.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[done] {len(forensics)} divergent pairs analyzed -> {out_dir}")


if __name__ == "__main__":
    main()
