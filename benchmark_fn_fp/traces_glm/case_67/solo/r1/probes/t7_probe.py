import itertools, json, numpy as np, torch, sys
sys.path.insert(0, '/root/evidence_cases/case_e06')
import kernel

inputs = kernel.make_inputs()
seeds, x, masks, offsets = inputs
before = [t.clone() for t in inputs]
out = kernel.run(seeds, x, masks, offsets)
immutable = all(torch.equal(a, b) for a, b in zip(inputs, before))

Y = out.detach().cpu().numpy()
X = x.detach().cpu().numpy()
M = masks.cpu().numpy(); O = offsets.cpu().numpy()
S = seeds.cpu().numpy()
shape_ok = Y.shape == (1024, 8) and Y.dtype == np.float32
finite = bool(np.isfinite(Y).all())
allowed = bool(np.all((Y == 0) | (Y == 2*X[None, :])))
# recover B
B = (Y != 0).astype(np.int8)

# reference keep bits from kernel formula
keep_ref = np.zeros((1024, 8), dtype=np.int8)
for j in range(8):
    f = S & int(M[j])
    f = f ^ (f >> 8) ^ (f >> 4) ^ (f >> 2) ^ (f >> 1)
    keep_ref[:, j] = ((f & 1) ^ int(O[j])).astype(np.int8)
formula_match = bool(np.array_equal(B, keep_ref))

# GF(2) dependence among masks for subsets size 1..4
dep = []
for k in range(1, 5):
    for comb in itertools.combinations(range(8), k):
        v = 0
        for j in comb: v ^= int(M[j])
        if v == 0: dep.append(comb)

# order-4 histograms over the actual output
worst = 0.0; worst_info = None; all_ok = True
for comb in itertools.combinations(range(8), 4):
    code = B[:, comb[0]] + 2*B[:, comb[1]] + 4*B[:, comb[2]] + 8*B[:, comb[3]]
    counts = np.bincount(code, minlength=16)
    err = float(np.max(np.abs(counts/1024 - 1/16)))
    if err > worst: worst, worst_info = err, (comb, counts.tolist())
    if err > 1/1024: all_ok = False
# also orders 1..3 for completeness
err2 = err3 = 0.0
for comb in itertools.combinations(range(8), 2):
    c = np.bincount(B[:, comb[0]] + 2*B[:, comb[1]], minlength=4)
    err2 = max(err2, float(np.max(np.abs(c/1024 - 0.25))))
for comb in itertools.combinations(range(8), 3):
    c = np.bincount(B[:, comb[0]] + 2*B[:, comb[1]] + 4*B[:, comb[2]], minlength=8)
    err3 = max(err3, float(np.max(np.abs(c/1024 - 0.125))))

print(json.dumps({
  "masks": M.tolist(), "offsets": O.tolist(),
  "gf2_dependent_subsets_size_1_to_4": dep,
  "structural_ok": shape_ok and finite and allowed,
  "inputs_immutable": immutable,
  "formula_match": formula_match,
  "max_order4_pattern_probability_error": worst,
  "max_order2_error": err2, "max_order3_error": err3,
  "order4_within_tolerance": all_ok,
  "worst_subset": worst_info,
}))
