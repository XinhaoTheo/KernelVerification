import sys, json
sys.path.insert(0, "/root/evidence_cases/case_e07")
import numpy as np
from kernel import make_inputs_numpy

features, anchors = make_inputs_numpy()
a = anchors.astype(np.float32)
abs_a = np.abs(a)
ties = []
for v in range(2):
    vals = abs_a[v].tolist()
    for i in range(4):
        for j in range(i+1, 4):
            if vals[i] == vals[j]:
                ties.append({"view": v, "col_i": i, "col_j": j, "abs": vals[i]})
# simulate kernel slot assignment per view
slots = {}
for v in range(2):
    mags = abs_a[v]
    sv = [int((mags < mags[k]).sum()) for k in range(4)]
    slots[v] = sv
result = {
    "metric": "exact float32 |anchor| tie check per view plus slot-collision simulation on fixed make_inputs() anchors (seed 171249)",
    "anchors": a.tolist(),
    "abs_anchors": abs_a.tolist(),
    "exact_tie_pairs": ties,
    "slots_per_view": slots,
    "slot_collision_any_view": any(len(set(sv)) < 4 for sv in slots.values()),
    "distinct_abs_values_per_view": [len(set(abs_a[v].tolist())) for v in range(2)],
}
print(json.dumps(result))