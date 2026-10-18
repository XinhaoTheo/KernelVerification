import torch, sys, json
sys.path.insert(0, "/root/cases/case_163")
import kernel

dev = "cuda"
results = {}
fail = False
details = []

def check(name, values, ref):
    global fail
    out = kernel.run(values)
    ref = ref.to(dev).to(torch.float32).unsqueeze(-1)
    ref = torch.stack([kernel.reference(values)[...,0], kernel.reference(values)[...,1]], dim=-1)
    exact = ref
    err = (out.double() - exact.double()).abs()
    tol = 1e-4 + 1e-6 * exact.double().abs()
    bad = (err > tol) | ~torch.isfinite(out)
    mx = err.max().item()
    nbad = int(bad.sum())
    results[name] = {"max_err": mx, "violations": nbad, "shape": list(out.shape), "nonfinite": int((~torch.isfinite(out)).sum())}
    if nbad:
        fail = True
        idx = bad.nonzero()[:5]
        details.append({"name": name, "idx": idx.tolist(), "out": out[bad][:5].tolist(), "exact": exact[bad][:5].tolist()})
    # input preservation
    return values

g = torch.Generator(device="cpu").manual_seed(1234)
for n in (64, 128, 256):
    for b in (1, 7, 256):
        v = torch.randn((b, n), generator=g).to(torch.float16).to(dev)
        ref = kernel.reference(v)
        out = kernel.run(v)
        err = (out.double() - ref.double()).abs()
        tol = 1e-4 + 1e-6 * ref.double().abs()
        bad = (err > tol) | ~torch.isfinite(out)
        results[f"rand_n{n}_b{b}"] = {"max_err": err.max().item(), "violations": int(bad.sum())}
        if bad.sum():
            fail = True
            details.append({"case": f"rand_n{n}_b{b}", "idx": bad.nonzero()[:5].tolist(), "out": out[bad][:5].tolist(), "ref": ref[bad][:5].tolist()})

# extreme magnitudes
for n in (64, 128, 256):
    v = torch.full((3, n), 65504.0, dtype=torch.float16, device=dev)
    v[1] = -65504.0
    v[2] = torch.linspace(-65504, 65504, n, dtype=torch.float32).to(torch.float16)
    ref = kernel.reference(v); out = kernel.run(v)
    err = (out.double() - ref.double()).abs(); tol = 1e-4 + 1e-6 * ref.double().abs()
    bad = (err > tol) | ~torch.isfinite(out)
    results[f"extreme_n{n}"] = {"max_err": err.max().item(), "violations": int(bad.sum())}
    if bad.sum():
        fail = True
        details.append({"case": f"extreme_n{n}", "idx": bad.nonzero()[:5].tolist(), "out": out[bad][:5].tolist(), "ref": ref[bad][:5].tolist()})

# subnormals and signed zeros
v = torch.zeros((4, 128), dtype=torch.float16, device=dev)
v[0] = 6e-8
v[1] = -6e-8
v[2, 0] = -0.0
v[3] = torch.tensor([x * 6e-8 for x in range(128)], dtype=torch.float32).to(torch.float16)
ref = kernel.reference(v); out = kernel.run(v)
err = (out.double() - ref.double()).abs(); tol = 1e-4 + 1e-6 * ref.double().abs()
bad = (err > tol) | ~torch.isfinite(out)
results["subnormal_signedzero"] = {"max_err": err.max().item(), "violations": int(bad.sum())}
if bad.sum():
    fail = True
    details.append({"case": "subnormal", "out": out[bad][:5].tolist(), "ref": ref[bad][:5].tolist()})

# adversarial bit patterns (finite FP16 values)
vals = []
for i in range(0, 65536, 97):
    vals.append(i)
h = torch.tensor(vals, dtype=torch.int16).view(torch.float16).to(dev)
if not torch.isfinite(h).all():
    pass
else:
    v = h[: 128 * 4].reshape(4, 128)
    ref = kernel.reference(v); out = kernel.run(v)
    err = (out.double() - ref.double()).abs(); tol = 1e-4 + 1e-6 * ref.double().abs()
    bad = (err > tol) | ~torch.isfinite(out)
    results["bitpattern"] = {"max_err": err.max().item(), "violations": int(bad.sum())}
    if bad.sum():
        fail = True
        details.append({"case": "bitpattern", "out": out[bad][:5].tolist(), "ref": ref[bad][:5].tolist()})

# random uniform magnitudes across full FP16 range
v = (torch.rand((256, 256), generator=g) * 2 - 1).mul(65504).to(torch.float16).to(dev)
ref = kernel.reference(v); out = kernel.run(v)
err = (out.double() - ref.double()).abs(); tol = 1e-4 + 1e-6 * ref.double().abs()
bad = (err > tol) | ~torch.isfinite(out)
results["fullrange_n256_b256"] = {"max_err": err.max().item(), "violations": int(bad.sum())}
if bad.sum():
    fail = True
    details.append({"case": "fullrange", "idx": bad.nonzero()[:5].tolist(), "out": out[bad][:5].tolist(), "ref": ref[bad][:5].tolist()})

# input preservation + output properties
v = torch.randn((8, 128)).to(torch.float16).to(dev)
vc = v.clone()
out = kernel.run(v)
preserved = torch.equal(v, vc)
fresh = out.data_ptr() != v.data_ptr()
prop_ok = out.dtype == torch.float32 and out.is_contiguous() and list(out.shape) == [8, 65, 2] and out.device == v.device
results["input_preserved"] = preserved
results["output_props"] = prop_ok

print(json.dumps({"metric": "per-component abs error vs FP64 reference vs tol 1e-4+1e-6|exact|",
                  "results": results, "details": details, "fail": fail or not (preserved and prop_ok)}))
