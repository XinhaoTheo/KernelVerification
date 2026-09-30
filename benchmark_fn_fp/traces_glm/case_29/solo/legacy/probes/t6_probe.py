import torch, json
from kernel import fp8_roundtrip

def e4m3_grid():
    vals = [0.0]
    for e in range(-9, 9):  # subnormals e=-9 step 2^-9; normals from 2^-6 to 2^8
        step = 2.0**e
        for m in range(8):
            v = (1 + m/8) * step if e >= -6 else (m/8) * 2.0**(-6)
            vals.append(v)
    vals.append(448.0)
    s = sorted(set(vals))
    return torch.tensor(s + [-v for v in s if v > 0], dtype=torch.float32)

grid = e4m3_grid()

def ref(x):
    # nearest representable e4m3 value
    idx = torch.searchsorted(grid, x.contiguous().view(-1))
    idx = idx.clamp(1, grid.numel()-1)
    lo, hi = grid[idx-1], grid[idx]
    pick_hi = (hi - x.view(-1)) < (x.view(-1) - lo)
    return torch.where(pick_hi, hi, lo).view_as(x)

x = torch.tensor([0.001, 0.0009, 2.0**-9, 2.0**-7, 0.01, 500.0, 448.0, 449.5, 2.0**10, 3.7, -3.7, 7.6], dtype=torch.float32, device='cuda')
out = fp8_roundtrip(x)
r = ref(x)
res = {
    "inputs": x.tolist(),
    "kernel_out": out.tolist(),
    "ref_out": r.tolist(),
    "max_abs_err_kernel_vs_ref": float((out-r).abs().max()),
}
x2 = torch.randn(1<<20, device='cuda') * 0.05
out2 = fp8_roundtrip(x2); r2 = ref(x2)
res["rand_max_abs_err"] = float((out2-r2).abs().max())
res["rand_max_rel_err"] = float(((out2-r2).abs()/(r2.abs()+1e-30)).max())
print(json.dumps(res))