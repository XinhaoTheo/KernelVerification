import sys, json
sys.path.insert(0, "/root/evidence_cases/case_e03")
import numpy as np, torch
from kernel import make_inputs_numpy, make_inputs, run_sequence

cpu = make_inputs_numpy()
gpu = make_inputs(device="cuda")
C, Delta, P = cpu

# reference for (0,1,2): C[P0], then [P1], then + Delta
ref = C[0].astype(np.float64).copy()
for op in (0,1,2):
    ref = ref[P[op]].copy() if op < 2 else ref + Delta.astype(np.float64)
actual = run_sequence(*gpu, (0,1,2)).detach().cpu().numpy()
err = float(np.max(np.abs(actual.astype(np.float64) - ref)))
# also print seeded permutations and whether P1∘P0 composition (order) is an involution
order = P[1][P[0]]  # order[i] = physical slot of logical row i after (0,1)
inv = bool(np.array_equal(order[order], np.arange(8)))
print(json.dumps({"word":[0,1,2],"max_abs_error":err,"tolerance":1e-5,"fails":err>1e-5,
 "P0":P[0].tolist(),"P1":P[1].tolist(),"order_after_01":order.tolist(),"order_is_involution":inv}))