import torch, json
SHIFT = 34
MULT = ((1 << SHIFT) + 7 - 1) // 7
print("MULT", MULT, "SHIFT", SHIFT)
bad = 0
examples = []
CH = 1 << 28
for base in range(0, 1 << 32, CH):
    v = torch.arange(base, base + CH, dtype=torch.int64)
    q = (v * MULT) >> SHIFT
    ref = v // 7
    m = q != ref
    diff = int(m.sum())
    bad += diff
    if diff:
        idx = m.nonzero()[:5].flatten().tolist()
        examples += [(base + i, int((base + i) * MULT >> SHIFT), (base + i) // 7) for i in idx]
print(json.dumps({"mismatches_over_all_uint32": bad, "examples": examples[:10]}))
