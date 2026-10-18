import torch, json, itertools, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_154/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
dev = "cuda" if torch.cuda.is_available() else "cpu"

def check(num, den, tag):
    n0, d0 = num.clone(), den.clone()
    out = k.run(num, den)
    ref = k.reference(num, den).to(torch.float64)
    err = (out.to(torch.float64) - ref).abs()
    tol = 1e-7 + 5e-7 * ref.abs()
    ratio = (err / tol)
    return {"tag": tag, "n": num.shape[0], "max_ratio": ratio.max().item(),
            "max_abs_err": err.max().item(), "finite_out": bool(torch.isfinite(out).all()),
            "inputs_preserved": bool(torch.equal(n0, num) and torch.equal(d0, den))}

results = []
# corner values incl. extrema, subnormals, signed zeros
vals = [0.0, -0.0, 5.96e-8, -5.96e-8, 6.1e-5, -6.1e-5, 1.0, -1.0, 1.001, 2048.0, 65504.0, -65504.0, 5e-4, 0.1, 3.0, 1234.5]
den_vals = [v for v in vals if v != 0.0] + [1e-4, -1e-4, 6e-5, 65504.0, -65504.0, 0.5, -0.5, 2.0]
num_rows, den_rows = [], []
rng = torch.Generator().manual_seed(7)
for a in vals:
    for b in vals:
        num_rows.append((a, b))
for c in den_vals:
    for d in den_vals:
        if not (c == 0.0 and d == 0.0):
            den_rows.append((c, d))
num = torch.tensor(num_rows, dtype=torch.float16, device=dev)
den = torch.tensor(den_rows[: len(num_rows)] if len(den_rows) >= len(num_rows) else den_rows, dtype=torch.float16, device=dev)
if len(num_rows) < len(den_rows):
    num = num.repeat(2, 1)[: len(den_rows)]
    den = den.repeat(2, 1)[: num.shape[0]]
    num = num[: den.shape[0]]
results.append(check(num, den, "grid_corners"))

# cancellation: numerator components chosen so a*c ~ -b*d, plus random pairs of all FP16 magnitudes
n2 = torch.empty((65536, 2), dtype=torch.float16)
n2[:, 0] = torch.randn(65536, generator=torch.Generator().manual_seed(1))
n2[:, 1] = -(n2[:, 0].to(torch.float32) * 1.0).to(torch.float16)  # near-cancellation pairs
d2 = torch.randn((65536, 2), generator=torch.Generator().manual_seed(2)).to(torch.float16).to(dev)
d2[(d2 == 0).all(dim=1), 0] = 1
n2 = n2.to(dev)
results.append(check(n2, d2, "near_cancellation_randn"))

# extreme magnitudes: large nums / tiny dens and vice versa
n3 = torch.full((8192, 2), 65504.0, dtype=torch.float16, device=dev); n3[:, 1] = -65504.0
d3 = torch.full((8192, 2), 5.96e-8, dtype=torch.float16, device=dev); d3[:, 1] = 5.96e-8
d3[::2, 0] = 6.1e-5
results.append(check(n3, d3, "extremes"))
n4 = torch.randn((8192, 2)).to(torch.float16).to(dev)
d4 = torch.full((8192, 2), 65504.0, dtype=torch.float16, device=dev); d4[:, 1] = 6.0
results.append(check(n4, d4, "big_denominator"))

print(json.dumps({"metric": "max_ratio_of_error_to_per_component_tolerance", "results": results}))
