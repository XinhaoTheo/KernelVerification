import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_28/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

torch.manual_seed(0)
res = {}

def relerr(x, y):
    return ((y - x).norm(dim=1) / x.norm(dim=1)).cpu()

def ref(x):
    s = (x.abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1e-30)
    return torch.clamp(torch.floor(x/s + 0.5), -127, 127) * s

# A: plain gaussian, no outliers
xA = torch.randn(8, 512, device='cuda')
yA = m.quant_dequant(xA)
res['A_kernel_max_relerr'] = float(relerr(xA, yA).max())
res['A_ref_max_relerr'] = float(relerr(xA, ref(xA)).max())

# B: heavy-tailed, 3 outlier channels 1-2 orders above bulk
xB = torch.randn(8, 512, device='cuda')
xB[:, 3] = 60.0
xB[:, 17] = -45.0
xB[:, 100] = 30.0
yB = m.quant_dequant(xB)
eB = relerr(xB, yB)
res['B_kernel_relerr_rows'] = [round(v,4) for v in eB.tolist()]
res['B_kernel_max_relerr'] = float(eB.max())
res['B_ref_max_relerr'] = float(relerr(xB, ref(xB)).max())
res['B_row0_in'] = [float(xB[0,3]), float(xB[0,17]), float(xB[0,100])]
res['B_row0_out'] = [float(yB[0,3]), float(yB[0,17]), float(yB[0,100])]

# C: single outlier 20x bulk
xC = torch.randn(8, 512, device='cuda')
xC[:, 5] = 20.0
yC = m.quant_dequant(xC)
eC = relerr(xC, yC)
res['C_kernel_max_relerr'] = float(eC.max())
res['C_ref_max_relerr'] = float(relerr(xC, ref(xC)).max())
res['C_row0_in5'] = float(xC[0,5]); res['C_row0_out5'] = float(yC[0,5])

res['limit'] = 0.05
print(json.dumps(res, indent=1))
