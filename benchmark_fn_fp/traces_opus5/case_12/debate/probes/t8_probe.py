
import json, sys, torch
sys.path.insert(0, "/root/cases/case_13")
from kernel import gqa_gather

res = {}
torch.manual_seed(0)
for (nkv, dim, nq) in [(2,64,4), (4,128,32), (1,64,8), (8,64,8), (2,32,2)]:
    kv = torch.randn(nkv, dim, device="cuda", dtype=torch.float32)
    out = gqa_gather(kv, nq)
    n_rep = nq // nkv
    ref_consec = torch.repeat_interleave(kv, n_rep, dim=0)   # contract
    ref_inter  = kv.repeat(n_rep, 1)                          # modulo mapping
    row_eq_consec = (out == ref_consec).all(dim=1)
    row_eq_inter  = (out == ref_inter).all(dim=1)
    # which kv row did each output row actually come from
    src = []
    for q in range(nq):
        m = [k for k in range(nkv) if torch.equal(out[q], kv[k])]
        src.append(m[0] if m else -1)
    first_bad = int((~row_eq_consec).nonzero()[0].item()) if (~row_eq_consec).any() else -1
    res[f"nkv{nkv}_dim{dim}_nq{nq}"] = {
        "n_rep": n_rep,
        "matches_consecutive_contract": bool(row_eq_consec.all()),
        "matches_interleaved_modulo": bool(row_eq_inter.all()),
        "num_mismatched_rows_vs_contract": int((~row_eq_consec).sum().item()),
        "first_mismatch_row": first_bad,
        "actual_src_kv_row_per_q": src,
        "expected_src_kv_row_per_q": [q // n_rep for q in range(nq)],
        "out_shape": list(out.shape),
        "dtype_ok": out.dtype == kv.dtype,
    }
print(json.dumps(res))
