
import torch, importlib.util, json
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_12/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

torch.manual_seed(0)
eps = 1e-6
res = {}

# ordinary row
X = torch.randn(8, 512, device="cuda", dtype=torch.float32)
Y = m.rms_norm_forward(X, eps)
ms = (X*X).sum(-1, keepdim=True)/X.shape[1]
ref = X * (1.0/torch.sqrt(ms+eps))
res["normal_max_abs_err"] = (Y-ref).abs().max().item()
res["normal_max_rel_err"] = ((Y-ref).abs()/(ref.abs()+1e-12)).max().item()

# near-zero row
Z = torch.zeros(3, 512, device="cuda", dtype=torch.float32)
Z[1] = 1e-8
Z[2] = 1e-4
Yz = m.rms_norm_forward(Z, eps)
msz = (Z*Z).sum(-1, keepdim=True)/Z.shape[1]
refz = Z * (1.0/torch.sqrt(msz+eps))
res["nearzero_kernel"] = Yz[:,0].tolist()
res["nearzero_ref"] = refz[:,0].tolist()
res["nearzero_max_abs_err"] = (Yz-refz).abs().max().item()
res["nearzero_rstd_kernel"] = (1.0/torch.sqrt(msz)+0).squeeze().tolist()
res["expected_ref_rstd"] = (1.0/torch.sqrt(msz+eps)).squeeze().tolist()
res["kernel_impl_rstd_formula_pred"] = (1.0/(torch.sqrt(msz)+eps)).squeeze().tolist()

# check kernel matches the "eps outside" formula exactly on normal rows
alt = X * (1.0/(torch.sqrt(ms)+eps))
res["matches_eps_outside_max_abs_err"] = (Y-alt).abs().max().item()
print(json.dumps(res, indent=1))
