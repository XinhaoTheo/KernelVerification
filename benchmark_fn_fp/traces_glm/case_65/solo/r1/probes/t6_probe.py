
import itertools, json, numpy as np, torch
import sys
sys.path.insert(0, "/root/evidence_cases/case_e04")
import kernel

cpu = kernel.make_inputs_numpy()
gpu = kernel.make_inputs("cuda")
initial, increment, perms = cpu
fails = []
max_err = 0.0
words = [tuple(w) for L in range(5) for w in itertools.product((0,1,2), repeat=L)]
for w in words:
    ref = initial.astype(np.float64).copy()
    for op in w:
        ref = ref[perms[op]].copy() if op < 2 else ref + increment.astype(np.float64)
    out = kernel.run_sequence(*gpu, w).detach().cpu().numpy()
    e = float(np.max(np.abs(out.astype(np.float64) - ref)))
    max_err = max(max_err, e)
    ok = out.shape == (8,32) and out.dtype == np.float32 and np.isfinite(out).all() and e <= 1e-5
    if not ok:
        fails.append({"operations": list(w), "error": e})
print(json.dumps({"num_words": len(words), "num_failures": len(fails), "max_error_overall": max_err, "failures": fails[:10]}))
