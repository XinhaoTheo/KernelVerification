
import json, torch, importlib.util, traceback
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_14/kernel.py")
k = importlib.util.module_from_spec(spec)
out = {}
try:
    spec.loader.exec_module(k)
    torch.manual_seed(0)
    x = torch.randn(8, 256, device="cuda", dtype=torch.float32)
    got = k.quant_dequant_int8(x)

    absmax = x.abs().amax(dim=1, keepdim=True).clamp_min(1e-10)
    scale = absmax / 127.0
    q = torch.clamp(torch.round(x / scale), -127, 127)
    ref = q * scale

    err = (got - ref).abs()
    # implied step per row = smallest nonzero |out| value
    def implied(t):
        res = []
        for r in range(t.shape[0]):
            v = t[r].abs()
            nz = v[v > 0]
            res.append(float(nz.min()) if nz.numel() else 0.0)
        return res
    out = {
        "metric": "max_abs_err_vs_per_row_quantized_reference",
        "shape": list(x.shape),
        "dtype": str(got.dtype),
        "row_absmax": [round(float(v), 5) for v in absmax.flatten()],
        "ref_step_per_row": [round(float(v), 6) for v in scale.flatten()],
        "kernel_implied_step_per_row": [round(v, 7) for v in implied(got)],
        "ref_implied_step_per_row": [round(v, 7) for v in implied(ref)],
        "max_abs_err": float(err.max()),
        "mean_abs_err": float(err.mean()),
        "ref_max_quant_err_vs_x": float((ref - x).abs().max()),
        "kernel_max_quant_err_vs_x": float((got - x).abs().max()),
        "frac_elems_differ": float((err > 1e-6).float().mean()),
        "allclose_rtol0_atol_refstep": bool(torch.allclose(got, ref, rtol=0, atol=float(scale.max()))),
    }
except Exception as e:
    out = {"error": repr(e), "tb": traceback.format_exc()[-1500:]}
print(json.dumps(out))
