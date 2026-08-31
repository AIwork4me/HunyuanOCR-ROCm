import json, hashlib
import torch

calls = [json.loads(l) for l in open('forensics/results/probe2/lmhead.jsonl')]
dec = [c for c in calls if c['hidden_shape'] == [1, 6656]]
meas = dec[8:]
assert len(meas) == 512
def at(g, s): return meas[g * 64 + s]

steps = [json.loads(l) for l in open('forensics/results/probe2/steps.jsonl')]
gens = {0: torch.load('forensics/results/probe2/gen0_fp32.pt', weights_only=True),
        5: torch.load('forensics/results/probe2/gen5_fp32.pt', weights_only=True)}
ok = bad = 0
for st in steps:
    if st['gen'] in (0, 5) and st['step'] % 16 == 0:
        row = gens[st['gen']][st['step']]
        bf = row.to(torch.bfloat16).contiguous().view(torch.uint8).numpy().tobytes()
        h = hashlib.sha256(bf).hexdigest()
        if h == at(st['gen'], st['step'])['logits_sha256']: ok += 1
        else: bad += 1
print("alignment spot-check:", ok, "match,", bad, "mismatch")

run = json.load(open('forensics/results/probe2/run.json'))
seqs = run['token_ids']
print("unique:", run['unique_outputs'])
print()
print("step | token sets | hidden shas distinct | logits shas distinct")
for s in range(10):
    toks = {seqs[g][s] for g in range(8)}
    hs = {at(g, s)['hidden_sha256'] for g in range(8)}
    ls = {at(g, s)['logits_sha256'] for g in range(8)}
    print(f"{s:4d} | {len(toks)} distinct | {len(hs):2d} | {len(ls):2d}")
print()
for g in range(1, 8):
    D = next((k for k in range(64) if seqs[0][k] != seqs[g][k]), None)
    if D is None:
        print(f"gen0 v gen{g}: identical to gen0"); continue
    h_eq = at(0, D)['hidden_sha256'] == at(g, D)['hidden_sha256']
    l_eq = at(0, D)['logits_sha256'] == at(g, D)['logits_sha256']
    print(f"gen0 v gen{g}: D={D} tok {seqs[0][D]}v{seqs[g][D]} | hidden@D equal: {h_eq} | logits@D equal: {l_eq}")
