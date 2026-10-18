import torch

MULT = ((1 << 34) + 6) // 7
SHIFT = 34
assert MULT == (1 << 34 + 7 - 1) // 7  # placeholder sanity, actual value below
MULT = ((1 << 34) + 7 - 1) // 7
print("MULT", MULT, "SHIFT", SHIFT, "MULT*7", MULT*7)

bad = 0
bad_examples = []
CH = 1 << 28
for base in range(0, 1 << 32, CH):
    v = torch.arange(base, base + CH, dtype=torch.int64)
    q = (v * MULT) >> SHIFT
    ref = v // 7
    diff = (q != ref).sum().item()
    bad += diff
    if diff:
        idx = (q != ref).nonzero()[:5].flatten().tolist()
        bad_examples += [(base + i, int((base+i)*MULT >> SHIFT), (base+i)//7) for i in idx]
print("total mismatches over all 2^32 uint32 values:", bad)
print("examples (value, computed q, true q):", bad_examples[:10])
import json
print(json.dumps({"mismatches": bad, "examples": bad_examples[:10]}))
