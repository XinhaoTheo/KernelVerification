import torch, sys
sys.path.insert(0, '/root/cases/case_168')
from kernel import run

# Independently compute the AES S-box per FIPS 197 (not using kernel's reference).
def gf_mul(a, b):
    p = 0
    for _ in range(8):
        if b & 1: p ^= a
        hi = a & 0x80
        a = (a << 1) & 0xFF
        if hi: a ^= 0x1b
        b >>= 1
    return p

def gf_inv(a):
    if a == 0: return 0
    for b in range(256):
        if gf_mul(a, b) == 1: return b

sbox = []
for x in range(256):
    v = gf_inv(x)
    y = v ^ 0x63
    for shift in range(1, 5):
        y ^= ((v << shift) | (v >> (8 - shift))) & 0xFF
    sbox.append(y)

expected = torch.tensor(sbox, dtype=torch.uint8, device='cuda')
out = run(expected.clone())  # sweep all 256 values in one call
out_cpu = out.cpu().tolist()
mismatches = [(x, out_cpu[x], sbox[x]) for x in range(256) if out_cpu[x] != sbox[x]]
print({"n_mismatch": len(mismatches), "samples": mismatches[:10], "check0": out_cpu[0], "check53": out_cpu[0x53]})
import json
print(json.dumps({"metric": "exact byte equality vs independent FIPS197 S-box over 256 values",
                  "n_mismatch": len(mismatches), "mismatch_samples": mismatches[:10],
                  "known_pairs_ok": out_cpu[0]==0x63 and out_cpu[0x53]==0xed}))