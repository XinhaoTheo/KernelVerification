import json, torch, sys
sys.path.insert(0, "/root/cases/case_141")
import kernel as K

# small and mid-range full-domain sampling; if 32-bit intermediate corruption existed, these would fail
torch.manual_seed(0)
vals = [0, 1, 6, 7, 13, 100, 1000, 12345, 1000000, 999999999, 2147483647, 3435973830]
rand = torch.randint(0, 3435973836, (4093,), dtype=torch.int64)
t = torch.tensor(vals, dtype=torch.int64).to(torch.uint32).cuda()
r = rand.to(torch.uint32).cuda()
allt = torch.cat([t, r])
out = K.run(allt).cpu().long()
x = allt.cpu().long()
row, col = out[:, 0].long(), out[:, 1].long()
ok = (row == x // 7) & (col == x % 7)
result = {
    "n": int(allt.numel()),
    "n_mismatch": int((~ok).sum()),
    "col_in_0_6": bool(((col >= 0) & (col <= 6)).all()),
    "reconstruction_ok": bool((7*row + col == x).all()),
    "max_value_tested": int(x.max()),
    "first_mismatch_value": int(x[~ok][0]) if (~ok).any() else None,
}
print(json.dumps(result))