
import json, itertools, sys
import numpy as np
sys.path.insert(0, "/root/evidence_cases/case_e05")
import kernel as K

seeds, x, masks, offsets = K.make_inputs_numpy()
inputs = K.make_inputs()
Y = K.run(*inputs).detach().cpu().numpy()
B = (Y != 0).astype(np.int32)

# multiples-of-32 structure check: all 4-subset pattern counts divisible by 32
all_counts = []
for quad in itertools.combinations(range(8), 4):
    code = sum(B[:,quad[i]] << i for i in range(4))
    all_counts.append(np.bincount(code, minlength=16))
all_counts = np.array(all_counts)
div32 = bool((all_counts % 32 == 0).all())
# at least one 4-subset has a pattern count 0 or 128 (not 64)
worst_idx = np.unravel_index(np.argmax(np.abs(all_counts - 64)), all_counts.shape)
bad_counts = [(q, c.tolist()) for q, c in zip(itertools.combinations(range(8),4), all_counts)
              if (np.abs(c - 64) > 1).any()]
n_bad_subsets = len(bad_counts)
# exhaustively verify the Griesmer premise: no 8 vectors in GF(2)^5 can have all 70 4-subsets rank 4
# (structural check over the actual matrix)
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
deficient = [list(q) for q in itertools.combinations(range(8),4)
             if gf2_rank([M[j] for j in q]) < 4]
result = dict(all_counts_multiple_of_32=div32,
              num_subsets_violating_contract=n_bad_subsets,
              violating_subsets=[(list(q), c) for q, c in bad_counts][:10],
              worst_count=int(all_counts[worst_idx]),
              worst_subset=list(itertools.combinations(range(8),4)[worst_idx[0]]),
              pattern_index=int(worst_idx[1]),
              num_rank_deficient_4subsets=len(deficient),
              rank_deficient_subsets=deficient,
              necessarily_violates_order4=bool(n_bad_subsets > 0 and div32))
print(json.dumps(result))
