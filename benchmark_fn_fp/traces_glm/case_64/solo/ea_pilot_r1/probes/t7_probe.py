import itertools, sys, json
sys.path.insert(0, "/root/evidence_cases/case_e03")
import numpy as np, torch
import kernel as K

cpu = K.make_inputs_numpy()
gpu = K.make_inputs("cuda")
C, Delta, P = cpu[0].astype(np.float64), cpu[1].astype(np.float64), cpu[2]

words = [tuple(w) for L in range(5) for w in itertools.product((0,1,2), repeat=L)]
assert len(words) == 121
fails = []
maxerr = 0.0
for w in words:
    ref = C.copy()
    for op in w:
        if op < 2: ref = ref[P[op]].copy()
        else: ref = ref + Delta
    out = K.run_sequence(*gpu, w).detach().cpu().numpy()
    err = float(np.max(np.abs(out.astype(np.float64) - ref)))
    ok = out.shape == (8,32) and out.dtype == np.float32 and np.isfinite(out).all() and err <= 1e-5
    maxerr = max(maxerr, err)
    if not ok:
        fails.append({"word": list(w), "err": err})

print(json.dumps({"num_words": len(words), "num_failures": len(fails),
                  "max_abs_error_over_all": maxerr,
                  "first_failures": fails[:5],
                  "claim_confirmed": len(fails) > 0}))