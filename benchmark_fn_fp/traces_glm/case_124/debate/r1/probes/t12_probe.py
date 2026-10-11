import sys, json
sys.path.insert(0, '/root/cases/case_124')
import torch
import kernel as K

dev = 'cuda'

def contract(a, m, r, z, s):
    P = a*m
    H = (P + (1 << 30)) // (1 << 31)
    R = H
    Q = min(127, max(-128, R + z))
    return (Q - z) * s

m = (1 << 31) - 1
r = 0
z = 0
s = 1.0
a = (1 << 31) - 1
P = a*m
H = (P + (1 << 30)) // (1 << 31)
vals = [a, a-1, (1<<30), -(1<<31), -((1<<31)-1)]
acc = torch.tensor([vals], dtype=torch.int32, device=dev)
mult = torch.tensor([m], dtype=torch.int32, device=dev)
shift = torch.tensor([r], dtype=torch.int32, device=dev)
zero = torch.tensor([z], dtype=torch.int32, device=dev)
scale = torch.tensor([s], dtype=torch.float32, device=dev)
out = K.run(acc, mult, shift, zero, scale).cpu()
exp = [contract(v, m, r, z, s) for v in vals]
got = out[0].tolist()
bad = [(v, e, g, H) for v, e, g in zip(vals, exp, got) if e != g]
print(json.dumps({'claim': 'c3', 'a_max_int31_H': H, 'mismatches': bad,
                  'expected': exp, 'got': got}))