import numpy as np, json

SEED = 193602
def make_inputs_numpy():
    rng = np.random.Generator(np.random.PCG64(SEED))
    low = rng.uniform(0.25, 1.75, (4, 8)).astype(np.float32)
    interior = np.concatenate((np.full((4, 1), 2.0**30, np.float32),
                               np.full((4, 1), -(2.0**30), np.float32), low), axis=1)
    order = rng.permutation(10)
    x = np.empty((4, 12), dtype=np.float32)
    x[:, 0] = np.float32(2.0**80)
    x[:, 1:-1] = interior[:, order]
    x[:, -1] = np.float32(-(2.0**80))
    return (x, order, low)

x, order, low = make_inputs_numpy()
values = x.astype(np.float64)
expected = np.zeros(4, dtype=np.float64)
for column in range(12):  # replicate initial_probe reference exactly
    expected = expected + values[:, column]
exact = values.sum(axis=1)
print(json.dumps({
    "metric": "recomputed initial_probe float64 sequential reference vs recorded [0,0,0,0]",
    "recomputed_reference": expected.tolist(),
    "exact_real_sum": exact.tolist(),
    "sum_of_low_values": low.astype(np.float64).sum(axis=1).tolist(),
    "recorded_reference": [0.0, 0.0, 0.0, 0.0],
    "mismatch_with_recorded": bool(not np.allclose(expected, 0.0)),
    "permutation_order": order.tolist()
}))