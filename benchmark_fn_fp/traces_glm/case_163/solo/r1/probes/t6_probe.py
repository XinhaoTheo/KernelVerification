import torch, json, sys
sys.path.insert(0, "/root/cases/case_163")
import kernel as K

dev = "cuda"
results = {"cases": [], "fail": None}
tol_a, tol_r = 1e-4, 1e-6

def check(name, values):
    try:
        orig = values.clone()
        out = K.run(values)
        exact = K.reference(values)
    except Exception as e:
        results["fail"] = (name, repr(e)); return False
    ok = True
    info = {}
    if out.dtype != torch.float32 or not out.is_contiguous():
        info["bad_container"] = str((out.dtype, out.is_contiguous())); ok = False
    if list(out.shape) != [values.shape[0], values.shape[1]//2+1, 2]:
        info["bad_shape"] = str(out.shape); ok = False
    if not torch.equal(values, orig):
        info["input_mutated"] = True; ok = False
    if not torch.isfinite(out).all():
        info["nonfinite"] = True; ok = False
    err = (out.to(torch.float64) - exact).abs()
    lim = tol_a + tol_r * exact.abs()
    bad = int((err > lim).sum())
    if bad:
        idx = (err > lim).nonzero()[:5]
        info["violations"] = bad
        info["examples"] = [(idx.tolist(), err[tuple(idx.t())].tolist(), exact[tuple(idx.t())].tolist())]
        ok = False
    results["cases"].append({"name": name, "ok": ok, "max_err": float(err.max()), **info})
    return ok

torch.manual_seed(0)
all_ok = True
# random inputs across N and batch sizes
for n in (64, 128, 256):
    for b in (1, 3, 7, 256):
        vals = torch.randn((b, n), dtype=torch.float16)
        all_ok &= check(f"randn_B{b}_N{n}", vals.cuda())
# make_inputs example
v, = K.make_inputs(); all_ok &= check("make_inputs", v)
# extreme cases
for n in (64, 128, 256):
    v = torch.full((5, n), 65504.0, dtype=torch.float16).cuda(); all_ok &= check(f"allmax_N{n}", v)
    v = torch.full((5, n), -65504.0, dtype=torch.float16).cuda(); all_ok &= check(f"allmin_N{n}", v)
    v = torch.full((5, n), 0.0, dtype=torch.float16).cuda(); all_ok &= check(f"zeros_N{n}", v)
    v = torch.zeros((5, n), dtype=torch.float16); v[:, ::3] = -0.0; all_ok &= check(f"negzeros_N{n}", v.cuda())
    v = torch.full((5, n), 5.96e-8, dtype=torch.float16).cuda(); all_ok &= check(f"subnormal_N{n}", v)
    # alternating extreme pattern
    v = torch.empty((4, n), dtype=torch.float16)
    v[:, 0::2] = 65504.0; v[:, 1::2] = -65504.0; all_ok &= check(f"alt_extreme_N{n}", v.cuda())
    # single impulse
    v = torch.zeros((3, n), dtype=torch.float16); v[1, 5] = 1.0; all_ok &= check(f"impulse_N{n}", v.cuda())
    # random full-range fp16
    v = (torch.randn((8, n)) * 20000).to(torch.float16).cuda(); all_ok &= check(f"bigrand_N{n}", v)
    # constant row with nonzero offset into a larger storage
    base = torch.randn((9, n), dtype=torch.float16).cuda()
    all_ok &= check(f"offset_N{n}", base[2:2+4])
    # storage offset via storage window
    flat = torch.randn(4*n + 16, dtype=torch.float16).cuda()
    w = flat[16:16+4*n].view(4, n)
    all_ok &= check(f"storage_offset_N{n}", w)
# all batch rows identical
v = torch.randn((1, 128), dtype=torch.float16).cuda().expand(256, 128).contiguous()
all_ok &= check("identical_rows", v)
# quantized small values, broad range
v = torch.arange(256, dtype=torch.float32).unsqueeze(0).repeat(8,1).to(torch.float16).cuda()
all_ok &= check("ramp_N256", v)

results["all_ok"] = all_ok
print(json.dumps(results))
