
import numpy as np, torch, json, sys, os
sys.path.insert(0, '/root/cases/case_154')
import kernel as K

rng = np.random.default_rng(0)

def fp16_bits(n):
    # uniform over finite fp16 bit patterns (incl. zeros, subnormals, extremes)
    bits = rng.integers(0, 0x7C00, size=n, dtype=np.uint32)
    signs = (rng.integers(0, 2, size=n, dtype=np.uint32) << 15)
    v = (signs | bits).astype(np.uint16)
    return torch.from_numpy(v.view(np.float16).astype(np.float32).copy())

n = 300000
a = fp16_bits(n); b = fp16_bits(n); c = fp16_bits(n); d = fp16_bits(n)
A, B, C, D = a.double(), b.double(), c.double(), d.double()
legal = ~((C == 0) & (D == 0))
num_r = A*C + B*D
prod_max = torch.maximum((A*C).abs(), (B*D).abs())
canc = num_r.abs() / prod_max.clamp_min(1e-300)

# 1) rows with strongest real-numerator cancellation (nonzero exact sum)
sel1 = legal & (num_r != 0) & (prod_max > 1)
idx1 = torch.nonzero(sel1 & (canc < 2**-15)).flatten()
if len(idx1) > 0:
    order = canc[idx1].argsort()
    idx1 = idx1[order[:4000]]
else:
    idx1 = torch.zeros(0, dtype=torch.long)

# 2) rows with exact zero real numerator and large products (cancellation)
sel2 = legal & (num_r == 0) & (prod_max > 1000)
idx2 = torch.nonzero(sel2).flatten()[:2000]

# 3) extreme-value cross product
extreme = torch.tensor([0.0, -0.0, 6.0464e-8, -6.0464e-8, 2**-24, -2**-24, 2**-14, -2**-14,
                        1.0, -1.0, 1024.0, -1024.0, 65504.0, -65504.0, 65504.0/4, -65504.0/4,
                        5.9605e-7, -5.9605e-7], dtype=torch.float64)
m = 100000
ai = rng.integers(0, len(extreme), m); bi = rng.integers(0, len(extreme), m)
ci = rng.integers(0, len(extreme), m); di = rng.integers(0, len(extreme), m)
ea, eb, ec, ed = extreme[ai], extreme[bi], extreme[ci], extreme[di]
legal3 = ~((ec == 0) & (ed == 0))
num_r3 = ea*ec + eb*ed
canc3 = num_r3.abs() / torch.maximum((ea*ec).abs(), (eb*ed).abs()).clamp_min(1e-300)
idx3keep = legal3 & (canc3 < 0.5)  # any partial cancellation
# take all, they are only 100k

# assemble candidate rows
parts = []
for src, ix in (( (a,b,c,d), idx1), ((a,b,c,d), idx2)):
    if len(ix):
        parts.append(torch.stack([src[0][ix], src[1][ix], src[2][ix], src[3][ix]], dim=1))
parts.append(torch.stack([ea, eb, ec, ed], dim=1).float())
rows = torch.cat(parts, dim=0).double()
# dedupe legal rows and cap size
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
real_ratio = err_ratio[:, 0]
imag_ratio = err_ratio[:, 1]

best = int(real_ratio.argmax().item())
result = {
    "claim": "c1 real-sum cancellation",
    "n_rows": int(len(rows)),
    "max_real_err_ratio": float(real_ratio.max().item()),
    "max_imag_err_ratio": float(imag_ratio.max().item()),
    "n_real_violations": int((real_ratio > 1).sum().item()),
    "n_imag_violations": int((imag_ratio > 1).sum().item()),
    "worst_row_inputs_num": num.cpu()[best].tolist(),
    "worst_row_inputs_den": den.cpu()[best].tolist(),
    "worst_row_out": out.cpu()[best].tolist(),
    "worst_row_ref": ref.cpu()[best].tolist(),
    "worst_row_real_err_ratio": float(real_ratio[best].item()),
    "worst_row_tol": float(tol.cpu()[best, 0].item()),
    "max_real_abs_err": float((out[:,0]-ref[:,0]).abs().max().item()),
    "metric": "per-component |out-ref| / (1e-7+5e-7|ref|), max over rows",
}
print(json.dumps(result))
