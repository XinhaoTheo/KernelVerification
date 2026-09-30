
import json, itertools, sys
import numpy as np
sys.path.insert(0, "/root/evidence_cases/case_e05")
import kernel as K

seeds, x, masks, offsets = K.make_inputs_numpy()
seeds = seeds.astype(np.int64)
pos = [0,1,2,4,8]
M = np.array([[(int(m) >> p) & 1 for p in pos] for m in masks])  # 8x5 restricted matrix

def gf2_rank(rows):
    rows = [r.copy() for r in rows]; rank = 0
    for c in range(5):
        piv = None
        for i in range(rank, len(rows)):
            if rows[i][c]: piv = i; break
        if piv is None: continue
        rows[rank], rows[piv] = rows[piv], rows[rank]
        for i in range(len(rows)):
            if i != rank and rows[i][c]:
                rows[i] ^= rows[rank]
        rank += 1
    return rank

full_rank = gf2_rank(M)
sub_ranks = {}
min_rank = 5; rank3_subsets = []
for quad in itertools.combinations(range(8), 4):
    r = gf2_rank([M[j] for j in quad])
    sub_ranks[quad] = r
    if r < min_rank: min_rank = r
    if r < 4: rank3_subsets.append((list(quad), r))

# confirm with actual output histogram for every rank<4 subset
inputs = K.make_inputs()
Y = K.run(*inputs).detach().cpu().numpy()
B = (Y != 0).astype(np.int32)
bad_hist = {}
for quad, r in rank3_subsets:
    code = sum(B[:,quad[i]] << i for i in range(4))
    counts = np.bincount(code, minlength=16).tolist()
    bad_hist[str(quad)] = dict(rank=r, counts=counts)
# also verify: seeds differing only in dropped bits give identical keep vector
def fold(seed, mask, off):
    f = seed & mask
    f ^= f >> 8; f ^= f >> 4; f ^= f >> 2; f ^= f >> 1
    return (f & 1) ^ off
sim = np.stack([fold(seeds, int(masks[j]), int(offsets[j])) for j in range(8)], axis=1)
rank_of_sim_space = gf2_rank(sim[:32])  # keep vectors of seeds 0..31 span the space
result = dict(restricted_positions=pos, mask_rows=M.tolist(), masks=masks.tolist(),
              offsets=offsets.tolist(), full_matrix_rank=full_rank,
              min_4subset_rank=min_rank,
              num_4subsets_below_rank4=len(rank3_subsets),
              subsets_with_rank_lt4=[q for q,_ in rank3_subsets],
              hist_of_deficient_subsets=bad_hist,
              any_count_outside_63_65=bool(any(
                  (max(c) > 65 or min(c) < 63) for c in
                  [d["counts"] for d in bad_hist.values()]) if bad_hist else False),
              keep_vector_count_of_first_32_seeds=int(len(np.unique(sim[:32], axis=0))))
print(json.dumps(result))
