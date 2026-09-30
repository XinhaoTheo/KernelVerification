import sys, json, torch
sys.path.insert(0, "/root/cases/case_19")
import kernel as K

dev = "cuda"
torch.manual_seed(1)
n_rows, n_cols = 3, 1000
logits = torch.randn(n_rows, n_cols, device=dev)
targets = torch.randint(0, n_cols, (n_rows,), device=dev)
loss, pred = K.cross_entropy_with_predictions(logits.clone(), targets)
ref_loss = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
ref_pred = logits.argmax(dim=1)
out = {
    "loss_kernel": loss.tolist(),
    "loss_ref": ref_loss.tolist(),
    "ratio_kernel_over_ref": (loss / ref_loss).tolist(),
    "loss_max_abs_err": float((loss - ref_loss).abs().max()),
    "n_non_ignore": n_rows,
    "pred_match": bool((pred == ref_pred).all()),
    "metric": "per-row loss vs -log softmax(logits[i])[target[i]] (no normalization)",
}
print(json.dumps(out, indent=2))
