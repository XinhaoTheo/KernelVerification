import sys, json, itertools
import numpy as np, torch
sys.path.insert(0, "/root/evidence_cases/case_e04")
import kernel as K

cpu = K.make_inputs_numpy()
gpu = K.make_inputs()
P0, P1 = cpu[2][0], cpu[2][1]
comp = P0[P1]  # order after (0,1): next_order[i]=order[P0[i]]=P0[i]; careful below
inv = np.array_equal(comp[comp], np.arange(K.N))

def reference(word):
    ref = cpu[0].astype(np.float64).copy()
    for op in word:
        ref = ref[cpu[2][op]].copy() if op < 2 else ref + cpu[1].astype(np.float64)
    return ref

words = [(0,1,2),(1,0,2),(0,1,1,2),(1,0,0,2),(0,1,0,2),(0,2,2),(1,2,2),(0,1,2,2)]
records = []
for w in words:
    actual = K.run_sequence(*gpu, w).detach().cpu().numpy()
    err = float(np.max(np.abs(actual.astype(np.float64) - reference(w))))
    records.append({"operations": list(w), "max_absolute_error": err, "passed": err <= 1e-5})
print(json.dumps({"P0": P0.tolist(), "P1": P1.tolist(), "P0_after_P1_is_involution": bool(inv),
                  "records": records, "any_fail": any(not r["passed"] for r in records)}))