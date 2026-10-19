import sys, json
sys.path.insert(0, '/root/cases/case_124')
import torch
import kernel as K

dev = 'cuda'

def contract(a, m, r, z, s):
    P = a*m
    H = (P + (1 << 30)) // (1 << 31)  # floor, ties toward +inf (exact for negatives too since Python floor)
    if r:
        mag = abs(H)
        Rmag = (mag + (1 << (r-1))) // (1 << r)
        R = -Rmag if H < 0 else Rmag
    else:
        R = H
    Q = min(127, max(-128, R + z))
    return (Q - z) * s

results = {}
mismatches = []
for r in (1, 2, 3):
    m = (1 << 30) + 3
    z = 0
    s = 1.0
    # scan a values, find exact tie residues H mod 2^r == 2^(r-1)
    tie_vals = []
    for a in range(-5000, 5001):
        H = (a*m + (1<<30)) // (1<<31)
        if H % (1 << r) == (1 << (r-1)):
            tie_vals.append(a)
        if len(tie_vals) >= 64:
            break
    if not tie_vals:
        results[f'r{r}'] = 'no tie values found'
        continue
    acc = torch.tensor([tie_vals], dtype=torch.int32, device=dev)
    mult = torch.tensor([m], dtype=torch.int32, device=dev)
    shift = torch.tensor([r], dtype=torch.int32, device=dev)
    zero = torch.tensor([z], dtype=torch.int32, device=dev)
    scale = torch.tensor([s], dtype=torch.float32, device=dev)
    out = K.run(acc, mult, shift, zero, scale).cpu()
    exp = [contract(a, m, r, z, s) for a in tie_vals]
    got = out[0].tolist()
    bad = [(a, e, g) for a, e, g in zip(tie_vals, exp, got) if e != g]
    mismatches.extend((r, *b) for b in bad)
    results[f'r{r}'] = {'n_tie': len(tie_vals), 'n_mismatch': len(bad),
                       'first_example': bad[0] if bad else None}

print(json.dumps({'claim': 'c1', 'results': results,
                  'any_mismatch': len(mismatches) > 0,
                  'example_r_a_expected_got': mismatches[:3]}))