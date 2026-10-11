import sys, json
sys.path.insert(0, '/root/cases/case_124')
import torch
import kernel as K

dev = 'cuda'

def contract(a, m, r, z, s):
    P = a*m
    H = (P + (1 << 30)) // (1 << 31)
    if r:
        mag = abs(H)
        Rmag = (mag + (1 << (r-1))) // (1 << r)
        R = -Rmag if H < 0 else Rmag
    else:
        R = H
    Q = min(127, max(-128, R + z))
    return (Q - z) * s

r = 1
m = (1 << 30) + 3
z = 0
s = 1.0
# negative a with H exactly odd (half-tie for /2, away-from-zero should round down i.e. more negative)
neg_tie = []
a = -1
while len(neg_tie) < 128:
    H = (a*m + (1<<30)) // (1<<31)
    if H < 0 and (H % 2 == 1 or H % 2 == -1):
        neg_tie.append(a)
    a -= 1

acc = torch.tensor([neg_tie], dtype=torch.int32, device=dev)
mult = torch.tensor([m], dtype=torch.int32, device=dev)
shift = torch.tensor([r], dtype=torch.int32, device=dev)
zero = torch.tensor([z], dtype=torch.int32, device=dev)
scale = torch.tensor([s], dtype=torch.float32, device=dev)
out = K.run(acc, mult, shift, zero, scale).cpu()
exp = [contract(a, m, r, z, s) for a in neg_tie]
got = out[0].tolist()
bad = [(a, e, g) for a, e, g in zip(neg_tie, exp, got) if e != g]
# also positive-side odd H for contrast
pos_tie = []
a = 1
while len(pos_tie) < 128:
    H = (a*m + (1<<30)) // (1<<31)
    if H % 2 == 1:
        pos_tie.append(a)
    a += 1
accp = torch.tensor([pos_tie], dtype=torch.int32, device=dev)
outp = K.run(accp, mult, shift, zero, scale).cpu()
expp = [contract(a, m, r, z, s) for a in pos_tie]
badp = [(a, e, g) for a, e, g in zip(pos_tie, expp, outp[0].tolist()) if e != g]

print(json.dumps({'claim': 'c2', 'r': r, 'm': m,
                  'n_neg_tie': len(neg_tie), 'neg_mismatches': len(bad),
                  'neg_first_example_a_expected_got': bad[0] if bad else None,
                  'n_pos_tie': len(pos_tie), 'pos_mismatches': len(badp),
                  'pos_first_example': badp[0] if badp else None}))