import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_29/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
# e4m3 representable finite values (positive); grid: normals e in [-6,8], 8 values per binade; max = 448
rep = set()
for e in range(-6, 9):
    step = 2.0**(e-3)
    for i in range(8):
        v = i * step  # values within [2^e, 2^(e+1))
        if e == -6:
            # binade [2^-6, 2^-5): mantissas 1.0..1.875 -> v = 2^-6 + i*2^-9
            v = 2.0**-6 + i * 2.0**-9
        rep.add(v)
# subnormals
for i in range(1, 8):
    rep.add(i * 2.0**-9)
rep.add(0.0)
def representable(v):
    a = abs(v)
    return a in rep
inputs = [449.0, 460.0, 464.0, 465.0, 500.0, 1000.0, 10000.0, -464.0, -1000.0]
x = torch.tensor(inputs, dtype=torch.float32, device="cuda")
out = k.fp8_roundtrip(x)
res = {float(v): float(o) for v, o in zip(x.tolist(), out.tolist())}
violations = {v: o for v, o in res.items() if abs(v) > 448.0 and not representable(o)}
nearest_rep = 448.0  # no finite e4m3 value exists above 448
wrong_nearest = {v: o for v, o in violations.items() if o != (nearest_rep if v > 0 else -nearest_rep)}
print(json.dumps({
    "metric": "representability on the e4m3 grid + nearest-representable match",
    "results": res,
    "e4m3_max_finite": 448.0,
    "n_inputs_above_448": sum(1 for v in inputs if abs(v) > 448),
    "n_nonrepresentable_outputs": len(violations),
    "nonrepresentable_outputs": violations,
    "n_wrong_nearest": len(wrong_nearest),
    "wrong_nearest_outputs": wrong_nearest
}))