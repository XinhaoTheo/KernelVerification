
import json, torch, importlib.util, traceback
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_14/kernel.py")
k = importlib.util.module_from_spec(spec)
out = {}
try:
    spec.loader.exec_module(k)
    torch.manual_seed(2)
    base = torch.randn(4, 256, device="cuda", dtype=torch.float32)
    base = base / base.abs().amax(dim=1, keepdim=True)   # each row absmax = 1
    mults = torch.tensor([[1e-3], [1e-2], [1.0], [1e3]], device="cuda")
    x = base * mults
    got = k.quant_dequant_int8(x)

    absmax = x.abs().amax(dim=1, keepdim=True).clamp_min(1e-10)
    scale = absmax / 127.0
    ref = torch.clamp(torch.round(x / scale), -127, 127) * scale

    def relerr(a, b):
        return [round(float((a[r]-b[r]).norm() / b[r].norm()), 6) for r in range(a.shape[0])]
    out = {
        "metric": "zero_code_fraction_and_per_row_relative_error",
        "row_absmax": [float(v) for v in absmax.flatten()],
        "kernel_zero_frac_per_row": [round(float((got[r] == 0).float().mean()), 4) for r in range(4)],
        "ref_zero_frac_per_row": [round(float((ref[r] == 0).float().mean()), 4) for r in range(4)],
        "kernel_relerr_vs_x_per_row": relerr(got, x),
        "ref_relerr_vs_x_per_row": relerr(ref, x),
        "kernel_relerr_vs_ref_per_row": relerr(got, ref),
        "kernel_row_absmax_out": [float(v) for v in got.abs().amax(dim=1)],
        "max_abs_err_vs_ref": float((got - ref).abs().max()),
    }
except Exception as e:
    out = {"error": repr(e), "tb": traceback.format_exc()[-1500:]}
print(json.dumps(out))
