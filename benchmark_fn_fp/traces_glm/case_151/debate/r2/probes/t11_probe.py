import sys, json
import numpy as np

Q = 3329
vals = np.arange(0, 3329, dtype=np.int64)
# Build all reachable coefficient values: every residue can appear as a Montgomery coefficient? Check both the full square range and actual banks.
def multiply(v, c):
    product = v * c  # int64, max 3328*3328 ~ 1.1e7
    correction = (product * 3327) & 65535
    reduced = (product + correction * Q) >> 16
    return reduced - Q if reduced >= Q else reduced

# Exhaustive over full reachable operand grid (value in [0,3328], coefficient in [0,3328])
V = vals[:, None]
C = vals[None, :]
product = V * C
correction = (product * 3327) & 65535
reduced = (product + correction * Q) >> 16
out = np.where(reduced >= Q, reduced - Q, reduced)
expected = (V * C * pow(1 << 16, -1, Q)) % Q  # value * coeff in Montgomery form -> value*omega mod Q
wrong_residue = int((out != expected).sum())
out_of_range = int(((out < 0) | (out > 3328)).sum())
reduced_max = int(reduced.max())
out = out.astype(np.int64)

print(json.dumps({"metric": "exhaustive count of wrong residues and out-of-range results over all 3329x3329 reachable operand pairs",
 "reason": "tol-0 exactness; _multiply must return (value*coefficient*R^-1) mod Q = value*omega mod Q in [0,3328] for every pair",
 "wrong_residue_count": wrong_residue, "out_of_range_count": out_of_range,
 "pairs_checked": 3329 * 3329, "max_reduced_before_subtract": reduced_max}))