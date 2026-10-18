import torch, importlib.util, json
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_148/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

def bits(t): return (t.view(torch.int16).to(torch.int64).to("cpu").numpy() & 0xFFFF).tolist()

vals = [0.0, -0.0, 65504.0, -65504.0, 17.5, 18.0, 11.09, -11.09, -11.1, 12.0, -12.0,
        5.96e-8, -5.96e-8, 2.98e-8, -2.98e-8, 1e-4, -1e-4, 20.0, -20.0, 100.0, -100.0]
x = torch.tensor(vals, dtype=torch.float16, device="cuda")
# reference per contract
import struct
def ref_one(v):
    import math
    r = 1.0/(1.0+math.exp(-v))  # fp64 sigmoid
    # correctly rounded once to fp16 via struct (RNE)
    return struct.unpack("<e", struct.pack("<e", r))[0]
out = k.run(x)
kb = bits(out)
rb = [int(struct.unpack("<H", struct.pack("<e", 1.0/(1.0+__import__("math").exp(-float(xi)))))[0]) for xi in x.cpu()]
# careful: rb computed above
rb2 = []
for xi in x.cpu().tolist():
    r = 1.0/(1.0+__import__("math").exp(-xi))
    rb2.append(int(struct.unpack("<H", struct.pack("<e", r))[0]))
res = {"inputs": x.cpu().tolist(), "kernel_bits": kb, "ref_bits": rb2,
       "mismatch": [i for i in range(len(vals)) if kb[i]!=rb2[i]]}
# extra checks: zero inputs must give 0x3800; x>17.5 must give 0x0000
res["zero_gives_half"] = kb[0]==0x3800 and kb[1]==0x3800
res["large_pos_zero_bits"] = all(kb[i]==0x0000 for i in [2,6,17,20])
res["large_neg"] = {int(x.cpu().tolist()[i]): kb[i] for i in [3,18,21]}
print(json.dumps(res))
