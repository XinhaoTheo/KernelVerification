import torch, importlib.util, numpy as np, struct, json
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_148/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

# extreme tail encodings: all FP16 values >= 17.0 and <= -11.0, plus zeros, subnormals
bits = np.arange(65536, dtype=np.uint16)
legal = (bits & np.uint16(0x7C00)) != np.uint16(0x7C00)
lb = bits[legal]
xf = lb.view(np.float16)
sel = (xf.astype(np.float32) >= 17.0) | (xf.astype(np.float32) <= -11.0) | (xf == 0)
xs = xf[sel]
x64 = xs.astype(np.float64)
# overflow-safe FP64 sigmoid reference
a = np.abs(x64)
e = np.exp(-a)
s64 = np.where(x64 >= 0, 1.0/(1.0+e), e/(1.0+e))
ref16 = s64.astype(np.float16)  # RNE
refbits = ref16.view(np.uint16).astype(np.int64)

xt = torch.from_numpy(xs.copy()).cuda()
out = k.run(xt)
ob = out.view(torch.int16).to(torch.int64).cpu().numpy() & 0xFFFF
bad = np.nonzero(ob != refbits)[0]

def f16hex(v): return format(np.float16(v).view(np.uint16) if isinstance(v,(int,float)) else v, "04x")

res = {"n_tested": int(len(xs)),
       "mismatches": int(len(bad)),
       "examples": [{"x": float(x64[i]), "kernel_bits": hex(int(ob[i])), "ref_bits": hex(int(refbits[i]))} for i in bad[:20]],
       "zero_gives_0x3800": bool(np.all(ob[(xs==0)] == 0x3800)),
       "large_pos_gives_zero": bool(np.all(ob[xs.astype(np.float32) >= 17.5] == 0x0000)),
       "large_neg_gives_0x3B80": bool(np.all(ob[xs.astype(np.float32) <= -11.1] == 0x3B80)),
       "midrange_tail_ok": bool(np.all(ob[(np.abs(xs.astype(np.float32))>=11.0)&(np.abs(xs.astype(np.float32))<17.0)] ==
                                        refbits[(np.abs(xs.astype(np.float32))>=11.0)&(np.abs(xs.astype(np.float32))<17.0)]))}
print(json.dumps(res))
