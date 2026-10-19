import torch, json, importlib.util, sys
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_28/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

torch.manual_seed(0)
res = {}

def relerr(x, y):
    return ((y - x).norm(dim=1) / x.norm(dim=1)).cpu()

# Case A: plain gaussian activations (no outliers)
xA = torch.randn(8, 512, device='cuda')
yA = m.quant_dequant(xA)
res['A_gauss_max_relerr'] = float(relerr(xA, yA).max())

# Case B: heavy-tailed: a few channels 1-2 orders of magnitude above bulk
xB = torch.randn(8, 512, device='cuda')
for r in range(8):
    xB[r, 3] = 60.0
    xB[r, 17] = -45.0
    xB[r, 100] = 30.0
res['B_outlier_relerr_per_row'] = [round(v,4) for v in relerr(xB, yB := m.quant_dequant(xB)).tolist()]
res['B_max_relerr'] = float(relerr(xB, yB).max())
res['B_row0_outlier_in'] = [float(xB[0,3]), float(xB[0,17]), float(xB[0,100])]
res['B_row0_outlier_out'] = [float(yB[0,3]), float(yB[0,17]), float(yB[0,100])]

# Case C: single outlier only (10x bulk)
xC = torch.randn(8, 512, device='cuda')
xC[:, 5] = 20.0
yC = m.quant_dequant(xC)
res['C_max_relerr'] = float(relerr(xC, yC).max())
res['C_row0_out5'] = float(yC[0,5])

# Reference: correct absmax calibration
def ref(x):
    s = (x.abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1e-30)
    return torch.clamp(torch.floor(x/s + 0.5), -127, 127) * s
res['B_ref_absmax_max_relerr'] = float(relerr(xB, ref(xB)).max())
res['C_ref_absmax_max_relerr'] = float(relerr(xC, ref(xC)).max())
res['A_ref_absmax_max_relerr'] = float(relerr(xA, ref(xA)).max())
res['limit'] = 0.05
print(json.dumps(res, indent=1))
