
import json, torch, importlib.util, traceback
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_14/kernel.py")
k = importlib.util.module_from_spec(spec)
out = {}
try:
    spec.loader.exec_module(k)
    torch.manual_seed(1)
    x = torch.randn(4, 128, device="cuda", dtype=torch.float32) * 5.0
    x[0, 7] = 50.0
    x[1, 3] = -30.0
    got = k.quant_dequant_int8(x)

    absmax = x.abs().amax(dim=1, keepdim=True).clamp_min(1e-10)
    scale = absmax / 127.0
    ref = torch.clamp(torch.round(x / scale), -127, 127) * scale

    clipped = x.abs() > 4.0
    # saturated reconstruction magnitude should be exactly 4.0 (=127*4/127)
    sat = (got.abs() - 4.0).abs() < 1e-5
    out = {
        "metric": "saturation_count_and_abs_err_at_outliers",
        "n_elems_absx_gt_4": int(clipped.sum()),
        "n_saturated_to_4": int((clipped & sat).sum()),
        "outlier_x_50": float(x[0, 7]), "kernel_out_at_50": float(got[0, 7]), "ref_out_at_50": float(ref[0, 7]),
        "abs_err_at_50": float((got[0, 7] - ref[0, 7]).abs()),
        "outlier_x_m30": float(x[1, 3]), "kernel_out_at_m30": float(got[1, 3]), "ref_out_at_m30": float(ref[1, 3]),
        "max_abs_err_vs_ref": float((got - ref).abs().max()),
        "kernel_row_max_abs_out": [round(float(v), 5) for v in got.abs().amax(dim=1)],
        "row_absmax_x": [round(float(v), 5) for v in absmax.flatten()],
        "ref_row_max_abs_out": [round(float(v), 5) for v in ref.abs().amax(dim=1)],
        "max_shrink_below_x": float((x.abs() - got.abs()).max()),
    }
except Exception as e:
    out = {"error": repr(e), "tb": traceback.format_exc()[-1500:]}
print(json.dumps(out))
