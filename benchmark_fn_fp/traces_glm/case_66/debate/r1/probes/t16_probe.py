
import json, itertools, sys
import numpy as np
sys.path.insert(0, "/root/evidence_cases/case_e05")
import kernel as K

inputs = K.make_inputs()
seeds, x, masks, offsets = [t.detach().cpu().numpy() for t in inputs]
Y = K.run(*inputs).detach().cpu().numpy()
B = ((Y != 0).astype(np.int32))
struct = dict(shape=list(Y.shape), dtype=str(Y.dtype),
              finite=bool(np.isfinite(Y).all()),
              allowed_values=bool(np.logical_or(Y == 0, Y == 2*x[None, :]).all()))
quads = list(itertools.combinations(range(8), 4))
counts_all = []
for quad in quads:
    code = B[:,quad[0]] + 2*B[:,quad[1]] + 4*B[:,quad[2]] + 8*B[:,quad[3]]
    counts_all.append(np.bincount(code, minlength=16))
counts_all = np.array(counts_all)
worst_dev = float(np.max(np.abs(counts_all/1024 - 1/16)))
worst_flat = int(np.argmax(np.abs(counts_all - 64)))
violating = int((np.abs(counts_all - 64) > 1).sum())
# fold simulation: verify keep depends only on seed bits {0,1,2,4,8}
def fold(seed, mask, off):
    f = seed & mask
    f ^= f >> 8; f ^= f >> 4; f ^= f >> 2; f ^= f >> 1
    return (f & 1) ^ off
sim = np.stack([fold(seeds.astype(np.int64), int(masks[j]), int(offsets[j])) for j in range(8)], axis=1)
fold_matches_output = bool((sim == B).all())
distinct_vectors = int(len(np.unique(sim, axis=0)))
result = dict(structure=struct, fold_sim_matches_output=fold_matches_output,
              distinct_keep_vectors=distinct_vectors,
              total_patterns_checked=int(counts_all.size),
              patterns_violating_tolerance=violating,
              min_pattern_count=int(counts_all.min()),
              max_pattern_count=int(counts_all.max()),
              max_abs_prob_deviation=worst_dev,
              worst_subset=list(quads[worst_flat // 16]),
              worst_subset_pattern=int(worst_flat % 16),
              worst_subset_count=int(counts_all.flatten()[worst_flat]),
              contract_holds=bool(violating == 0))
print(json.dumps(result))
