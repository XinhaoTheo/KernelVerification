import torch, json, sys
sys.path.insert(0, '/root/cases/case_19')
from kernel import cross_entropy_with_predictions
n_rows, n_cols = 4, 1000
torch.manual_seed(0)
logits = torch.randn(n_rows, n_cols, device='cuda', dtype=torch.float32)
target = torch.randint(0, n_cols, (n_rows,), device='cuda')
loss, predicted = cross_entropy_with_predictions(logits, target)
ref = torch.nn.functional.cross_entropy(logits, target, reduction='none')
ratio = (ref / loss).tolist()
out = {"n_rows": n_rows, "kernel_loss": loss.tolist(), "ref_per_row": ref.tolist(),
       "ref_over_kernel_ratio": ratio,
       "max_rel_err": float(((loss - ref).abs() / ref.abs()).max()),
       "scaling_bug": bool(((loss * n_rows) - ref).abs().max() < 1e-4 * ref.abs().max())}
print(json.dumps(out))