import sys, json, importlib.util, torch, triton
spec = importlib.util.spec_from_file_location("kmod", "/root/pilot_cases/case_87/kernel.py")
kmod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kmod)

result = {"triton_version": triton.__version__}
a, b = kmod.make_inputs("cuda")
try:
    out_default = kmod.run(a, b)  # uses enable_fp_fusion=False
    torch.cuda.synchronize()
    result["launch_with_kwarg_false"] = "ok"
except Exception as e:
    result["launch_with_kwarg_false"] = "error: " + repr(e)
    print(json.dumps(result)); sys.exit(0)

out2 = torch.empty_like(b)
try:
    kmod._kernel[(1,)](a, b, out2, a.shape[0], a.shape[1], enable_fp_fusion=True)
    torch.cuda.synchronize()
    result["launch_with_kwarg_true"] = "ok"
    result["bitwise_equal_fusion_on_off"] = bool(torch.equal(out_default, out2))
    result["max_abs_diff_fusion_on_off"] = float((out_default.double() - out2.double()).abs().max().item())
except Exception as e:
    result["launch_with_kwarg_true"] = "error: " + repr(e)

af, bf = a.double(), b.double()
ref = torch.empty_like(bf)
h = torch.zeros(bf.shape[1], dtype=torch.float64, device=bf.device)
for t in range(bf.shape[0]):
    h = af[t] * h + bf[t]
    ref[t] = h
diff = out_default.double() - ref
N = out_default.numel()
E = diff.norm() / max(ref.norm(), 0.001 * N ** 0.5)
result["E_with_kwarg_false"] = float(E)
result["all_finite"] = bool(torch.isfinite(out_default).all().item())
print(json.dumps(result))