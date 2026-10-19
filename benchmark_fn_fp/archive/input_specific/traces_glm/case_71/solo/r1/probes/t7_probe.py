import json, torch, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/evidence_cases/case_e10/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
dev = "cuda"
w, v, g = k.make_inputs(dev)
w0, v0 = w.clone(), v.clone()
h = k.run_sequence(w, v, g)  # retain all six, observe after completion
H = torch.stack([t.float() for t in h])
alias = all(t.data_ptr() == w.data_ptr() for t in h)
# float64 reference
rw, rv = w0.double().clone(), v0.double().clone()
refs = []
for t in range(6):
    rv = 0.75*rv + 0.25*g[t].double()
    rw = rw - (1/32)*rv
    refs.append(rw.clone())
Href = torch.stack(refs)
Htarget32 = Href.float()
rel = float(torch.linalg.norm(H.double()-Href)/max(float(torch.linalg.norm(Href)),0.1))
rows_equal_w6 = all(torch.equal(H[t], w.cpu()) for t in range(6))
gunchanged = torch.equal(g, k.make_inputs(dev)[2]) if False else None
# gradient unchanged check: compare against numpy regeneration
import numpy as np
gr = torch.from_numpy(k.make_inputs_numpy()[2]).to(dev)
out = {
 "alias_all_rows_share_weights_storage": alias,
 "history_relative_error": rel,
 "tolerance": 0.025,
 "rows_equal_final_weights": bool(rows_equal_w6),
 "rows_share_values": bool(len(set(t.data_ptr() for t in h))==1),
 "H_row0_minus_target_row0_norm": float(torch.linalg.norm(H[0].double()-Href[0])),
 "gradients_unchanged": bool(torch.equal(g, gr)),
 "final_w_rel_err": float(torch.linalg.norm(w.double()-rw)/max(float(torch.linalg.norm(rw)),0.1)),
 "final_v_rel_err": float(torch.linalg.norm(v.double()-rv)/max(float(torch.linalg.norm(rv)),0.1)),
}
print(json.dumps(out))