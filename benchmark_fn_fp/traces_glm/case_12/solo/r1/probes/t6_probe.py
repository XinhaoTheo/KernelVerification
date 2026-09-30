import torch, json, sys
sys.path.insert(0, "/root/cases/case_12")
from kernel import rms_norm_forward

torch.manual_seed(0)
X = torch.randn(64, 128, device="cuda", dtype=torch.float32)
eps = 1e-6
Y = rms_norm_forward(X, eps)
ms = (X*X).sum(-1, keepdim=True) / X.shape[1]
ref = X / torch.sqrt(ms + eps)
rel = ((Y - ref).abs() / ref.abs()).max().item()
abs_err = (Y - ref).abs().max().item()

# also eps-sensitive case: row with small norm
X2 = torch.full((1, 128), 1e-4, device="cuda")
Y2 = rms_norm_forward(X2, 1e-5)
ms2 = (X2*X2).sum(-1, keepdim=True)/128
ref2 = X2 / torch.sqrt(ms2 + 1e-5)
rel2 = ((Y2-ref2).abs()/ref2.abs()).max().item()

print(json.dumps({"max_abs_err": abs_err, "max_rel_err": rel, "small_row_rel_err": rel2,
                  "kernel_impl": "1/(sqrt(ms)+eps)", "contract": "1/sqrt(ms+eps)"}))