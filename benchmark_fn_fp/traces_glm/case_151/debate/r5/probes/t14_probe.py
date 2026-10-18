import json, sys, torch
sys.path.insert(0, "/root/cases/case_151")
import kernel as K

# 1) bank offset checks (host-side)
coeffs128 = K._make_coefficients(128)
LOG_N = 7
last_start = 5 * ((1 << (2 * (LOG_N // 2))) - 1) // 3
fused_len = len(coeffs128) - 128 // 2
offset_ok = last_start == fused_len
half = 1; total = 0; stage_offsets_ok = True
for stage in range(LOG_N // 2):
    if 5 * (half - 1) // 3 != total: stage_offsets_ok = False
    total += 5 * half; half *= 4

# 2) N=128 run vs reference, B<=128
g = torch.Generator().manual_seed(1)
rows = []
for j in range(128):
    e = torch.zeros(1, 128, dtype=torch.int32); e[0, j] = 1
    rows.append(e)
rows.append(torch.randint(0, 3329, (128 - len(rows) if False else 32, 128), generator=g, dtype=torch.int32))
x = torch.cat(rows, 0)  # 160 -> too many; split into chunks
def test(xc_cpu):
    xc = xc_cpu.to("cuda")
    out = K.run(xc)
    ref = K.reference(xc_cpu).to("cuda")
    return torch.equal(out, ref), (out != ref).sum().item(), out, ref

ok_all = True; mism_total = 0; sample_out = None; sample_ref = None; sample_in = None
chunks = [x[:100], x[100:]]
outs = []
for ch in chunks:
    ok, m, out, ref = test(ch)
    ok_all &= ok; mism_total += m
    if sample_out is None:
        sample_out = out; sample_ref = ref; sample_in = ch

# 3) diagnostics: what does the kernel compute for a constant row (row 0 of first chunk is unit row;
# use last chunk's random rows + one constant row)
cdiag = torch.full((1, 128), 1234, dtype=torch.int32)
ok_c, m_c, out_c, ref_c = test(cdiag)
const_expected = 1234 * 128 % 3329
out_const = out_c[0, :4].tolist()
# unit row 0 (x[j=0]=1): DFT should be all ones (omega^0=1)
unit0 = torch.zeros(1, 128, dtype=torch.int32); unit0[0, 0] = 1
ok_u, m_u, out_u, ref_u = test(unit0)
out_unit0 = out_u[0, :8].tolist()
# check whether output is bit-reversed or otherwise permuted reference
perm_check = False
for ch_out, ch_ref in []:
    pass
o = sample_out[0].tolist(); r = sample_ref[0].tolist()
is_bitrev = o[0] == r[0]
if o[0] == r[0]:
    # test if o equals bit-reversal permutation of r
    def brev(k):
        b = 0
        for i in range(7):
            b |= ((k >> i) & 1) << (6 - i)
        return b
    is_bitrev = all(o[brev(k)] == r[k] for k in range(128))
else:
    is_bitrev = False
# check if difference is a multiple of 3329 per element
diffs = [(int(a) - int(b)) % 3329 for a, b in zip(o, r)]
mult_of_q = all(d % 3329 == 0 for d in [ (int(a)-int(b)) for a,b in zip(o,r) ])

print(json.dumps({
  "offset_ok": offset_ok, "stage_offsets_ok": stage_offsets_ok,
  "last_start": last_start, "fused_len": fused_len,
  "N128_all_ok": ok_all, "N128_total_mismatches": mism_total,
  "constant_row_ok": ok_c, "constant_row_mismatches": m_c,
  "constant_expected": const_expected, "constant_out_sample": out_const,
  "unit_row0_expected": 1, "unit_row0_out_sample": out_unit0,
  "unit_row0_ok": ok_u,
  "bitrev_perm_match": is_bitrev,
  "diffs_multiple_of_3329": mult_of_q,
}))
