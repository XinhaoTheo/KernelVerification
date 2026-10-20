
import json, itertools, sys
import numpy as np
sys.path.insert(0, "/root/evidence_cases/case_e05")
import kernel as K

seeds, x, masks, offsets = K.make_inputs_numpy()
inputs = K.make_inputs()
Y = K.run(*inputs).detach().cpu().numpy()
B = (Y != 0).astype(np.int32)
quads = list(itertools.combinations(range(8), 4))
all_counts = np.array([np.bincount(
    sum(B[:, q[i]] << i for i in range(4)), minlength=16) for q in quads])
div32 = bool((all_counts % 32 == 0).all())
dev = np.abs(all_counts - 64)
n_bad_subsets = int((dev > 1).any(axis=1).sum())
worst_flat = int(np.argmax(dev))
pos = [0,1,2,4,8]
M = np.array([[(int(m) >> p) & 1 for p in pos] for m in masks])
def gf2_rank(rows):
    rows=[r.copy() for r in rows]; rank=0
    for c in range(5):
        piv=next((i for i in range(rank,len(rows)) if rows[i][c]), None)
        if piv is None: continue
        rows[rank],rows[piv]=rows[piv],rows[rank]
        for i in range(len(rows)):
            if i!=rank and rows[i][c]: rows[i]^=rows[rank]
        rank+=1
    return rank
deficient = [list(q) for q in quads if gf2_rank([M[j] for j in q]) < 4]
# check affine dependence: for each deficient subset, whether the linear
# dependency among keep forms involves a nonzero constant (which makes all
# 16 patterns achievable)
result = dict(all_counts_multiple_of_32=div32,
              num_subsets_violating_contract=n_bad_subsets,
              min_count=int(all_counts.min()), max_count=int(all_counts.max()),
              worst_subset=list(quads[worst_flat // 16]),
              worst_pattern=int(worst_flat % 16),
              worst_count=int(all_counts.flatten()[worst_flat]),
              num_rank_deficient_4subsets=len(deficient),
              necessarily_violates_order4=bool(n_bad_subsets > 0),
              all_70_subsets_all_16_patterns_equal_64=bool((all_counts == 64).all()))
print(json.dumps(result))
