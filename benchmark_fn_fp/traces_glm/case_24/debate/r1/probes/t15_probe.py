import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("kernel", "/root/cases/case_25/kernel.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
# craft one block of 1024: absmax = 1.27 so scale = 0.01; values with x/scale in (-1,0) and (0,1)
neg_small = torch.full((512,), -0.005, device="cuda")   # x/scale = -0.5
pos_small = torch.full((512,), 0.005, device="cuda")    # x/scale = +0.5
x = torch.cat([neg_small, pos_small])
y = mod.requantize(x)
scale = 1.27 / 127.0
yref = torch.zeros_like(x)  # nearest of +/-0.5 is 0 (tie, but magnitude equal); use unambiguous cases too
# unambiguous cases: fraction magnitude < 0.5
x2 = torch.cat([torch.full((512,), -0.004, device="cuda"), torch.full((512,), 0.004, device="cuda")])
y2 = mod.requantize(x2)
print(json.dumps({
 "metric": "per-element output for small |x/scale| < 0.5 (nearest level is 0)",
 "reason": "floor maps small negatives to -scale while nearest gives 0; sign asymmetry",
 "neg_out_ambiguous(-0.5 levels)": y[:512].mean().item(),
 "pos_out_ambiguous(+0.5 levels)": y[512:].mean().item(),
 "neg_out_unambiguous(-0.4 levels)": y2[:512].mean().item(),
 "pos_out_unambiguous(+0.4 levels)": y2[512:].mean().item(),
 "scale": scale,
 "neg_magnified": bool(abs(y2[:512].mean().item()) > 0.005 * scale * 0.1 + 0.001),
 "pos_zero": bool(abs(y2[512:].mean().item()) < 1e-9)}))