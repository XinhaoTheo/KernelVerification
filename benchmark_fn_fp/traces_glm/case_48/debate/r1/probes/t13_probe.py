import numpy as np, json

# reproduce make_inputs_numpy exactly
rng = np.random.Generator(np.random.PCG64(501973))
coefficients = rng.normal(0.0, 1.0, (8, 49)).astype(np.float32)
anchor = np.float32(1.015625)
powers = float(anchor) ** np.arange(1, 49)
gen_expr = (-np.sum(coefficients[:, 1:].astype(np.float64) * powers[None, :], axis=1) + 0.003)
coefficients[:, 0] = gen_expr.astype(np.float32)
points = (float(anchor) + rng.normal(0.0, 0.00004, 8)).astype(np.float32)

c64 = coefficients.astype(np.float64)
p64 = points.astype(np.float64)
ref_stored = np.array([sum(c64[i,j]*p64[i]**j for j in range(49)) for i in range(8)])
# reference using unrounded generating expression for c[:,0] (invalid variant)
c_gen = c64.copy(); c_gen[:,0] = gen_expr
ref_gen = np.array([sum(c_gen[i,j]*p64[i]**j for j in range(49)) for i in range(8)])

# intermediate Horner partial-sum magnitudes (fp64, from k=48 down)
max_partial = []
for i in range(8):
    r = c64[i,48]; mx = abs(r)
    for kk in range(47,-1,-1):
        r = r*p64[i] + c64[i,kk]; mx = max(mx, abs(r))
    max_partial.append(mx)

norm_ref = np.linalg.norm(ref_stored)
floor = 0.001*np.sqrt(8)
print(json.dumps({
  "reference_stored_fp64": ref_stored.tolist(),
  "norm_ref": float(norm_ref),
  "floor": float(floor),
  "denominator": float(max(norm_ref, floor)),
  "binding_abs_budget": float(0.0002*max(norm_ref, floor)),
  "abs_budget_assumed_in_claim_1.6e-6": 1.6e-6,
  "ref_shift_if_gen_expr_used": (ref_stored - ref_gen).tolist(),
  "max_abs_shift_gen_vs_stored": float(np.max(np.abs(ref_stored-ref_gen))),
  "max_horner_partial_abs_per_row": max_partial,
  "floor_is_binding": bool(norm_ref < floor),
}))