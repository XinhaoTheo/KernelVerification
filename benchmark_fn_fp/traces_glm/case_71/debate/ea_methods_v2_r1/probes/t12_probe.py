import json, torch, importlib.util
spec = importlib.util.spec_from_file_location("kernel", "/root/evidence_cases/case_e10/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
device = "cuda" if torch.cuda.is_available() else "cpu"
w, v, G = k.make_inputs(device)
# float64 reference w[t+1]
w64 = w.double().clone(); v64 = v.double().clone()
target = []
for g in G:
    v64 = 0.75*v64 + 0.25*g.double()
    w64 = w64 - 0.03125*v64
    target.append(w64.clone())
target = torch.stack(target)
# kernel: retained history observed only after full sequence (as contract requires)
hist = k.run_sequence(w, v, G)  # retain, observe after
torch.cuda.synchronize() if device=="cuda" else None
H = torch.stack(hist)
aliasing = all(h.data_ptr() == hist[0].data_ptr() for h in hist)
num = torch.linalg.vector_norm(H.double()-target)
den = max(torch.linalg.vector_norm(target).item(), 0.1)
rel = (num/den).item()
rows_equal_w6 = all(torch.allclose(h.double(), target[5]) for h in hist)
out = {"history_relative_error": rel, "tolerance": 0.025,
       "exceeds_tolerance": rel > 0.025,
       "all_rows_equal_w6_at_observation": rows_equal_w6,
       "history_tensors_share_storage": aliasing,
       "target_norm": torch.linalg.vector_norm(target).item()}
print(json.dumps(out))