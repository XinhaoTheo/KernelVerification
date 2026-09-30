import json, torch, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/evidence_cases/case_e10/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
dev = "cuda"
w, v, g = k.make_inputs(dev)
w0, v0 = w.clone(), v.clone()
h = k.run_sequence(w, v, g)  # retain all six, observe after completion
H = torch.stack([t.float() for t in h])
alias = all(t.data_ptr() == w.data_ptr() for t in h)
rw, rv = w0.double().clone(), v0.double().clone()
refs = []
for t in range(6):
    rv = 0.75*rv + 0.25*g[t].double()
    rw = rw - (1/32)*rv
    refs.append(rw.clone())
Href = torch.stack(refs)
rel = float(torch.linalg.norm(H.double()-Href)/max(float(torch.linalg.norm(Href)),0.1))
rows_equal_w6 = all(torch.equal(H[t], w) for t in range(6))
gr = torch.from_numpy(k.make_inputs_numpy()[2]).to(dev)
out = {
 "alias_all_rows_share_weights_storage": bool(alias),
 "history_relative_error": rel,
 "tolerance": 0.025,
 "rows_equal_final_weights": bool(rows_equal_w6),
 "gradients_unchanged": bool(torch.equal(g, gr)),
 "final_w_rel_err": float(torch.linalg.norm(w.double()-rw)/max(float(torch.linalg.norm(rw)),0.1)),
 "final_v_rel_err": float(torch.linalg.norm(v.double()-rv)/max(float(torch.linalg.norm(rv)),0.1)),
 "all_finite": bool(torch.isfinite(H).all()) and bool(torch.isfinite(w).all()) and bool(torch.isfinite(v).all()),
 "shapes": [list(H.shape), list(w.shape), list(v.shape)],
 "H_row0_norm_diff_from_target": float(torch.linalg.norm(H[0].double()-Href[0])),
}
print(json.dumps(out))