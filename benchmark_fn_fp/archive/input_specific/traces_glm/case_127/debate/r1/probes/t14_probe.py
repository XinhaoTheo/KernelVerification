import sys, json
sys.path.insert(0, "/root/evidence_cases/case_e03")
import numpy as np, torch
from kernel import make_inputs_numpy, make_inputs, run_sequence

initial, delta, P = make_inputs_numpy()
gpu = make_inputs(device="cuda")

ref = initial.astype(np.float64).copy()
for op in (0, 1, 2):
    ref = ref[P[op]].copy() if op < 2 else ref + delta.astype(np.float64)
actual = run_sequence(*gpu, (0, 1, 2)).detach().cpu().numpy()
err = float(np.max(np.abs(actual.astype(np.float64) - ref)))
order = P[1][P[0]]
inv = bool(np.array_equal(order[order], np.arange(8)))
print(json.dumps({"word": [0, 1, 2], "max_abs_error": err, "tolerance": 1e-5,
                  "fails": err > 1e-5, "P0": P[0].tolist(), "P1": P[1].tolist(),
                  "order_after_01": order.tolist(), "order_is_involution": inv}))