
import numpy as np, torch, json, sys
sys.path.insert(0, '/root/cases/case_154')
import kernel as K

rng = np.random.default_rng(1)

def fp16_bits(n):
    bits = rng.integers(0, 0x7C00, size=n, dtype=np.uint32)
    signs = (rng.integers(0, 2, size=n, dtype=np.uint32) << 15)
    v = (signs | bits).astype(np.uint16)
    return torch.from_numpy(v.view(np.float16).astype(np.float32).copy())

n = 300000
a = fp16_bits(n); b = fp16_bits(n); c = fp16_bits(n); d = fp16_bits(n)
A, B, C, D = a.double(), b.double(), c.double(), d.double()
legal = ~((C == 0) & (D == 0))
num_i = B*C - A*D
prod_max = torch.maximum((B*C).abs(), (A*D).abs())
canc = num_i.abs() / prod_max.clamp_min(1e-300)

sel1 = legal & (num_i != 0) & (prod_max > 1)
idx1 = torch.nonzero(sel1 & (canc < 2**-15)).flatten()
if len(idx1) > 0:
    idx1 = idx1[canc[idx1].argsort()[:4000]]
else:
    idx1 = torch.zeros(0, dtype=torch.long)

sel2 = legal & (num_i == 0) & (prod_max > 1000)
idx2 = torch.nonzero(sel2).flatten()[:2000]

extreme = torch.tensor([0.0, -0.0, 6.0464e-8, -6.0464e-8, 2**-24, -2**-24, 2**-14, -2**-14,
                        1.0, -1.0, 1024.0, -1024.0, 65504.0, -65504.0, 65504.0/4, -65504.0/4,
                        5.9605e-7, -5.9605e-7], dtype=torch.float64)
m = 100000
ai = rng.integers(0, len(extreme), m); bi = rng.integers(0, len(extreme), m)
ci = rng.integers(0, len(extreme), m); di = rng.integers(0, len(extreme), m)
ea, eb, ec, ed = extreme[ai], extreme[bi], extreme[ci], extreme[di]
legal3 = ~((ec == 0) & (ed == 0))
num_i3 = eb*ec - ea*ed
canc3 = num_i3.abs() / torch.maximum((eb*ec).abs(), (ea*ed).abs()).clamp_min(1e-300)
idx3keep = legal3 & (canc3 < 0.5)

parts = []
for src, ix in (( (a,b,c,d), idx1), ((a,b,c,d), idx2)):
    if len(ix):
        parts.append(torch.stack([src[0][ix], src[1][ix], src[2][ix], src[3][ix]], dim=1))
parts.append(torch.stack([ea, eb, ec, ed], dim=1).float())
rows = torch.cat(parts, dim=0).double()
num = rows[:, :2]; den = rows[:, 2:]
legal_all = ~((den[:,0]==0)&(den[:,1]==0))
rows = rows[legal_all]
if len(rows) > 100000:
    rows = rows[:100000]
num = rows[:, :2].to(torch.float16).cuda()
den = rows[:, 2:].to(torch.float16).cuda()

out = K.run(num, den)
ref = K.reference(num.cpu(), den.cpu()).cuda()
tol = 1e-7 + 5e-7 * ref.abs()
err_ratio = (out - ref).abs() / tol
imag_ratio = err_ratio[:, 1]
real_ratio = err_ratio[:, 0]

best = int(imag_ratio.argmax().item())
result = {
    "claim": "c2 imag-sum cancellation",
    "n_rows": int(len(rows)),
    "max_imag_err_ratio": float(imag_ratio.max().item()),
    "max_real_err_ratio": float(real_ratio.max().item()),
    "n_imag_violations": int((imag_ratio > 1).sum().item()),
    "n_real_violations": int((real_ratio > 1).sum().item()),
    "worst_row_inputs_num": num.cpu()[best].tolist(),
    "worst_row_inputs_den": den.cpu()[best].tolist(),
    "worst_row_out": out.cpu()[best].tolist(),
    "worst_row_ref": ref.cpu()[best].tolist(),
    "worst_row_imag_err_ratio": float(imag_ratio[best].item()),
    "worst_row_tol": float(tol.cpu()[best, 1].item()),
    "max_imag_abs_err": float((out[:,1]-ref[:,1]).abs().max().item()),
    "metric": "per-component |out-ref| / (1e-7+5e-7|ref|), max over rows",
}
print(json.dumps(result))
