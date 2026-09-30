import itertools, json
import numpy as np

SEED = 150405
rng = np.random.Generator(np.random.PCG64(SEED))
masks = rng.choice(np.arange(1, 1024), size=8, replace=False).astype(np.int32)
offsets = rng.integers(0, 2, size=8, dtype=np.int32)

# Build the 8x10 GF(2) matrix of mask parity vectors (bit b of mask j selects seed bit b)
# For seed s, parity(s & m) = XOR over bits b where m has bit b set of s_b
A = ((masks[:, None] >> np.arange(10)[None, :]) & 1).astype(np.int8)  # (8,10)

def gf2_rank(A):
    M = A.copy()
    rows, cols = M.shape
    rank = 0
    for c in range(cols):
        pivot = None
        for r in range(rank, rows):
            if M[r, c]:
                pivot = r
                break
        if pivot is None:
            continue
        M[[rank, pivot]] = M[[pivot, rank]]
        for r in range(rows):
            if r != rank and M[r, c]:
                M[r] ^= M[rank]
        rank += 1
    return rank

rank = gf2_rank(A)

# Exhaustive subset-XOR search: any nonempty subset of channels whose mask parity vectors XOR to 0
dependent_subsets = []
for size in range(1, 9):
    for sub in itertools.combinations(range(8), size):
        v = np.zeros(10, dtype=np.int8)
        for j in sub:
            v ^= A[j]
        if not v.any():
            dependent_subsets.append(list(sub))

# Also check subset XOR constant via offsets: XOR of keep-bits = (XOR of parities) ^ (XOR of offsets)
# A subset's keep-bit XOR is a balanced function iff the mask vectors are independent;
# it is constant iff mask XOR = 0 (then constant = XOR of offsets).
constant_subsets = [s for s in dependent_subsets]
const_vals = []
for s in dependent_subsets:
    const_vals.append(int(np.bitwise_xor.reduce(offsets[s])))

# If a dependent 4-subset exists, its 16 pattern counts collapse (8 patterns get 0, 8 get 128)
seeds = np.arange(1024, dtype=np.int64)
S = seeds[:, None]
M = masks[None, :].astype(np.int64)
folded = S & M
for shift in (8, 4, 2, 1):
    folded = folded ^ (folded >> shift)
keep = ((folded & 1) ^ offsets[None, :].astype(np.int64)).astype(np.int8)

collapse_evidence = []
for sub in dependent_subsets:
    if len(sub) == 4:
        code = keep[:, sub[0]] + 2*keep[:, sub[1]] + 4*keep[:, sub[2]] + 8*keep[:, sub[3]]
        counts = np.bincount(code, minlength=16).tolist()
        collapse_evidence.append({"subset": sub, "pattern_counts": counts})

out = {
    "masks": masks.tolist(),
    "offsets": offsets.tolist(),
    "gf2_matrix_rank": rank,
    "full_independence": rank == 8,
    "dependent_subsets_x0": dependent_subsets,
    "constant_keep_xor_subsets": [ {"subset": s, "constant": v} for s, v in zip(constant_subsets, const_vals) ],
    "collapsed_4way_counts": collapse_evidence,
    "c2_hypothesis_dependence_exists": len(dependent_subsets) > 0,
}
print(json.dumps(out))
