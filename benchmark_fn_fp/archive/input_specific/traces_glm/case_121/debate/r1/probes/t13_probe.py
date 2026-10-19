import sys, json, itertools
sys.path.insert(0, "/root/evidence_cases/case_e03")
import numpy as np, torch
from kernel import make_inputs_numpy, make_inputs, run_sequence

cpu = make_inputs_numpy()
gpu = make_inputs(device="cuda")
C, Delta, P = cpu
results = []
for L in range(5):
    for word in itertools.product((0,1,2), repeat=L):
        ref = C[0].astype(np.float64).copy()
        for op in word:
            ref = ref[P[op]].copy() if op < 2 else ref + Delta.astype(np.float64)
        actual = run_sequence(*gpu, word).detach().cpu().numpy()
        err = float(np.max(np.abs(actual.astype(np.float64) - ref)))
        if err > 1e-5:
            results.append({"word":list(word),"err":err})
mixed = [r for r in results if 0 in r["word"] and 1 in r["word"] and 2 in r["word"] and r["word"].index(2) > max(r["word"].index(0), r["word"].index(1))]
print(json.dumps({"total_words":121,"failing_count":len(results),
 "mixed_before_append_failing":len(mixed),
 "max_err_overall":max((r["err"] for r in results), default=0.0),
 "failing_words_sample":results[:12],
 "all_failing_words_mixed_or_multi2":all(
   (0 in r["word"] and 1 in r["word"]) or r["word"].count(2) >= 2 for r in results)}))