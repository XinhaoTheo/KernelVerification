import sys, json, itertools
import numpy as np, torch
sys.path.insert(0, "/root/evidence_cases/case_e04")
import kernel as K

cpu = K.make_inputs_numpy()
gpu = K.make_inputs()

def reference(word):
    ref = cpu[0].astype(np.float64).copy()
    for op in word:
        ref = ref[cpu[2][op]].copy() if op < 2 else ref + cpu[1].astype(np.float64)
    return ref

words = [tuple(w) for L in range(5) for w in itertools.product((0,1,2), repeat=L)]
assert len(words) == 121
fails, worst = [], 0.0
for w in words:
    actual = K.run_sequence(*gpu, w).detach().cpu().numpy()
    err = float(np.max(np.abs(actual.astype(np.float64) - reference(w))))
    ok = actual.shape == (K.N, K.D) and actual.dtype == np.float32 and np.isfinite(actual).all() and err <= 1e-5
    worst = max(worst, err)
    if not ok:
        fails.append({"operations": list(w), "max_absolute_error": err})
print(json.dumps({"num_words": len(words), "num_fail": len(fails), "worst_error": worst,
                  "first_fails": fails[:10]}))