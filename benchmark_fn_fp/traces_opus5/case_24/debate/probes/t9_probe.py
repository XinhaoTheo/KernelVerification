
import json, importlib.util, torch
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_25/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
dev="cuda"
torch.manual_seed(1)

cases = []
overshoot_count = 0
for amax in [3.0, 1.0, 2.5, 0.7, 5.0, 1.3, 0.1, 7.77, 12.5, 0.333]:
    x = (torch.rand(1024, device=dev)*2-1) * (amax*0.5)   # all |x| < amax
    x[0] = -amax                                          # absmax element is negative
    out = k.requantize(x)
    scale = float(x.abs().max()/127.0)
    q0 = float(out[0]/scale)
    ov = float(out.abs().max()) > float(x.abs().max()) + 0.0
    overshoot_count += int(ov)
    cases.append({"amax": amax, "scale": scale, "q_of_min_elem": q0,
                  "out_abs_max": float(out.abs().max()),
                  "x_abs_max": float(x.abs().max()),
                  "overshoot": ov,
                  "ratio_out_over_in": float(out.abs().max())/float(x.abs().max())})
res = {"num_cases": len(cases), "num_overshoot": overshoot_count, "cases": cases}
print(json.dumps(res))
