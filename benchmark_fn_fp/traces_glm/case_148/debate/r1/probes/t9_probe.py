import torch, importlib.util, mpmath as mp
mp.mp.dps = 60
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_148/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

def ref_bits(x_f64):
    # correctly-rounded fp16 (RNE) of exact sigmoid(x) computed in high precision
    v = mp.mpf(x_f64)
    s = 1/(1+mp.exp(-v))
    if s <= 0: return 0x0000 if s == 0 else None
    # decompose
    e = 0
    t = s
    while t < mp.mpf(2)**-14 and t > 0:  # subnormal
        break
    # determine exponent
    if s >= mp.mpf(2)**-14:
        e = -1
        while not (mp.mpf(2)**e <= s < mp.mpf(2)**(e+1)): e += 1
        frac = s / mp.mpf(2)**e - 1  # in [0,1)
        step = mp.mpf(2)**-10
    else:
        e = -14
        frac = s / mp.mpf(2)**-14  # in [0,1)
        step = mp.mpf(2)**-24
    q = frac / step  # exact-ish quotient in high precision
    qi = mp.floor(q)
    r = q - qi
    if r > mp.mpf('0.5'): qi += 1
    elif r == mp.mpf('0.5'):
        qi += qi % 2  # ties to even
    if e == -14:
        val = qi * step
        if qi >= mp.mpf(2)**11: return 0x3C00  # shouldn't happen for sigmoid<1
        bits = (qi.int_() if hasattr(qi,'int_') else int(qi)) & 0x3FF
        return 0x400 | bits if val >= mp.mpf(2)**-14 else bits
    m = int(qi)
    if m >= 2048: m = 1024; e += 1
    return ((e + 15) << 10) | (m & 0x3FF)

def ref_tensor(x_cpu_f16):
    xs = x_cpu_f16.to(torch.float64).tolist()
    out = [ref_bits(v) for v in xs]
    return torch.tensor(out, dtype=torch.int32)

xs = torch.arange(0, 65536, dtype=torch.int32)
xf = xs.to(torch.uint16).numpy().view  # placeholder
import numpy as np
bits = xs.numpy().astype(np.uint16)
xf16 = torch.from_numpy(bits.view(np.float16).copy())
# exclude NaN/Inf: exponent all ones
exp_all = (bits.astype(np.int64) & 0x7C00) == 0x7C00
legal = ~exp_all
xf16 = xf16[torch.from_numpy(legal)]
N = xf16.numel()
print("legal count:", N, file=__import__("sys").stderr)

dev = "cuda"
res = {"n_legal": int(N), "mismatches": 0, "examples": []}
B = 131072
mism = 0
ex = []
# split into chunks <=131072; N ~ 61696 < 131072 so one shot
xc = xf16.to(dev)
out = k.run(xc)
ob = out.view(torch.int16).to(torch.int64).to("cpu").numpy() & 0xFFFF
rb = ref_tensor(xf16.to("cpu")).numpy().astype(np.int64)
bad = ob != rb
mism = int(bad.sum())
idx = np.nonzero(bad)[0][:10]
for i in idx:
    ex.append({"in_hex": int(bits[torch.from_numpy(legal).numpy().nonzero()[0][i]] if False else int(xf16[i].view(torch.int16)) & 0xFFFF),
               "kernel_bits": int(ob[i]), "ref_bits": int(rb[i]),
               "kernel_val": float(out[i].cpu()), "x": float(xc[i].cpu())})
res["mismatches"] = mism
res["examples"] = ex
import json
print(json.dumps(res))
