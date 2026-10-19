import json, sys
sys.path.insert(0, "/root/cases/case_141")
import torch, kernel

# Build residue-stratified high values plus the uint32 max.
vals = []
vals.append(4294967295)  # uint32 max
# high values congruent 0..6 mod 7
for r in range(7):
    v = 4294967295 - ((4294967295 - r) % 7)  # largest value ≡ r mod 7
    vals.append(v)
    vals.append(v - 7)
    vals.append(3435973837 + ((r - 3435973837) % 7))
    vals.append(3435973836 + ((r - 3435973836) % 7))
# some middle benign values
vals += [0, 1, 6, 7, 100, 2**32 - 7, 2**32 - 13]
vals = sorted(set(vals))
idx = torch.tensor(vals, dtype=torch.uint32, device="cuda")
out = kernel.run(idx)
ref = kernel.reference(idx)

out_i = out.to(torch.int64)
ref_i = ref.to(torch.int64)
v64 = idx.to(torch.int64)
true_q = v64 // 7

row_mismatch = (out_i[:, 0] != ref_i[:, 0])
diff = (out_i[:, 0] - ref_i[:, 0])[row_mismatch].tolist()
bad_vals = v64[row_mismatch].tolist()
kern_q = out_i[:, 0].tolist()
trueq = true_q.tolist()

result = {
    "n_tested": len(vals),
    "row_mismatch_count": int(row_mismatch.sum()),
    "mismatch_values": bad_vals,
    "kernel_minus_true_quotient": diff,
    "example_uint32max": {
        "value": 4294967295,
        "kernel_row": kern_q[vals.index(4294967295)],
        "true_row": 4294967295 // 7,
    },
}
print(json.dumps(result))