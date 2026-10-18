
import json, sys, torch
sys.path.insert(0, "/root/cases/case_151")
import kernel as K

# 1) verify last-stage start offset numerically against _make_coefficients layout
coeffs128 = K._make_coefficients(128)
LOG_N = 7
last_start = 5 * ((1 << (2 * (LOG_N // 2))) - 1) // 3
fused_len = len(coeffs128) - 128 // 2  # last-stage bank length is width//2=64
offset_ok = last_start == fused_len
# also check each fused stage start
starts = []
half = 1
total = 0
for stage in range(LOG_N // 2):
    s = 5 * (half - 1) // 3
    starts.append([stage, half, s, total])
    total += 5 * half
    half *= 4
stage_offsets_ok = all(s == t for _, _, s, t in starts)

# 2) run N=128 vs reference (this exercises the odd last stage)
g = torch.Generator().manual_seed(1)
rows = []
# unit rows
for j in range(128):
    e = torch.zeros(1, 128, dtype=torch.int32); e[0, j] = 1
    rows.append(e)
rows.append(torch.randint(0, 3329, (32, 128), generator=g, dtype=torch.int32))
rows.append(torch.full((1, 128), 3328, dtype=torch.int32))
x = torch.cat([r if r.shape[0] > 1 else r for r in rows], 0)
xc = x.to("cuda")
out = K.run(xc)
ref = K.reference(x).to("cuda")
ok = torch.equal(out, ref)
mism = (out != ref).sum().item()
# find first mismatching frequency index if any
idx = (out != ref).nonzero()
first_bad = idx[:3].tolist()

print(json.dumps({
  "last_start_offset": last_start,
  "fused_bank_len": fused_len,
  "offset_ok": offset_ok,
  "stage_offsets_ok": stage_offsets_ok,
  "N128_all_rows_ok": ok,
  "N128_mismatch_count": mism,
  "first_bad_positions": first_bad,
}))
