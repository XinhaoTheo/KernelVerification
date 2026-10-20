
import json, itertools, sys
import numpy as np
sys.path.insert(0, "/root/evidence_cases/case_e05")
import kernel as K

inputs = K.make_inputs()
seeds, x, masks, offsets = [t.detach().cpu().numpy() for t in inputs]
Y = K.run(*inputs).detach().cpu().numpy()
x_ = x
B = ((Y != 0).astype(np.int32))
allowed = np.logical_or(Y == 0, Y == 2*x[None, :]).all()
# structural
struct = dict(shape=list(Y.shape), dtype=str(Y.dtype), finite=bool(np.isfinite(Y).all()),
              allowed_values=bool(allowed))
# exhaustive order-4 histogram over all 70 4-subsets x 16 patterns
worst = 0.0; worst_subset=None; worst_count=None
counts_all = []
for quad in itertools.combinations(range(8), 4):
    code = (B[:,quad[0]] + 2*B[:,quad[1]] + 4*B[:,quad[2]] + 8*B[:,quad[3]])
    counts = np.bincount(code, minlength=16)
    counts_all.append(counts)
    dev = np.max(np.abs(counts/1024 - 1/16))
    if dev > worst:
        worst = dev; worst_subset = quad; worst_count = counts.tolist()
counts_all = np.array(counts_all)
min_count = int(counts_all.min()); max_count = int(counts_all.max())
violating = int((np.abs(counts_all - 64) > 1).sum())
# verify keep depends only on seed bits {0,1,2,4,8}: simulate fold and check vs B
def fold(seed, mask, off):
    f = seed & mask
    f ^= f >> 8; f ^= f >> 4; f ^= f >> 2; f ^= f >> 1
    return (f & 1) ^ off
sim = np.stack([fold(seeds.astype(np.int64), int(masks[j]), int(offsets[j])) for j in range(8)], axis=1)
fold_matches_output = bool((sim == B).all())
# distinct keep-vectors count
distinct_vectors = int(len(np.unique(sim, axis=0)))
result = dict(structure=struct, fold_sim_matches_output=fold_matches_output,
              distinct_keep_vectors=distinct_vectors,
              total_patterns_checked=int(counts_all.size),
              patterns_violating_tolerance=violating,
              min_pattern_count=min_count, max_pattern_count=max_count,
              max_abs_prob_deviation=float(worst),
              worst_subset=list(worst_subset), worst_subset_counts=worst_count,
              contract_holds=bool(violating == 0))
print(json.dumps(result))
