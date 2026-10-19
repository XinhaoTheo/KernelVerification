import itertools, json
import numpy as np

SEED = 150405
rng = np.random.Generator(np.random.PCG64(SEED))
masks = rng.choice(np.arange(1, 1024), size=8, replace=False).astype(np.int32)
offsets = rng.integers(0, 2, size=8, dtype=np.int32)

seeds = np.arange(1024, dtype=np.int32)

# Reproduce kernel keep-bit arithmetic exactly (int32 logic, same mask/fold/offset)
S = seeds[:, None].astype(np.int64)  # (1024,1)
M = masks[None, :].astype(np.int64)  # (1,8)
folded = S & M
for shift in (8, 4, 2, 1):
    folded = folded ^ (folded >> shift)
keep = ((folded & 1) ^ offsets[None, :].astype(np.int64)).astype(np.int8)  # (1024,8)

# Exhaustive 4-way joint histogram audit: 70 4-tuples x 16 patterns
results = []
worst = None
for tup in itertools.combinations(range(8), 4):
    code = (keep[:, tup[0]] + 2*keep[:, tup[1]] + 4*keep[:, tup[2]] + 8*keep[:, tup[3]])
    counts = np.bincount(code, minlength=16)
    assert counts.sum() == 1024
    for b in range(16):
        c = int(counts[b])
        if not (63 <= c <= 65):
            results.append({"tuple": list(tup), "pattern": b, "count": c})
            if worst is None or abs(c - 64) > abs(worst["count"] - 64):
                worst = {"tuple": list(tup), "pattern": b, "count": c}

total_checks = 70 * 16
violations = len(results)

out = {
    "masks": masks.tolist(),
    "offsets": offsets.tolist(),
    "total_4tuple_pattern_checks": total_checks,
    "violations_outside_63_65": violations,
    "first_violations": results[:10],
    "worst_violation": worst,
    "all_counts_in_63_65": violations == 0,
    "c1_hypothesis_violation_exists": violations > 0,
}
print(json.dumps(out))
