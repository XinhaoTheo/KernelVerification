import json, torch, importlib.util
spec = importlib.util.spec_from_file_location("kernel", "/root/evidence_cases/case_e10/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
device = "cuda" if torch.cuda.is_available() else "cpu"
w, v, G = k.make_inputs(device)
G0 = G.clone()
# float64 reference
w64 = w.double().clone(); v64 = v.double().clone()
for g in G:
    v64 = 0.75*v64 + 0.25*g.double()
    w64 = w64 - 0.03125*v64
k.run(w, v, G)  # consume all six steps
if device == "cuda": torch.cuda.synchronize()
def rel(a, b):
    return (torch.linalg.vector_norm(a.double()-b.double())/max(torch.linalg.vector_norm(b.double()).item(), 0.1)).item()
werr = rel(w, w64); verr = rel(v, v64)
g_same = torch.equal(G.view(torch.uint8), G0.view(torch.uint8))
finite = bool(torch.isfinite(w).all() and torch.isfinite(v).all() and w.dtype == torch.float32 and v.dtype == torch.float32 and w.shape == (128,) and v.shape == (128,))
out = {"final_weights_relative_error": werr, "final_velocity_relative_error": verr,
       "state_tolerance": 1e-5, "state_ok": werr <= 1e-5 and verr <= 1e-5,
       "G_byte_identical": g_same, "finite_correct_shapes": finite}
print(json.dumps(out))