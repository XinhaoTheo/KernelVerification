
import sys, json, torch
sys.path.insert(0, "/root/cases/case_19")
import kernel as K

dev = "cuda"
n_cols = 70000  # spans blocks: BLOCK_SIZE = min(32768, next_pow2(70000)) = 32768
BLOCK = min(K.MAX_FUSED_SIZE, 32768)
n_rows = 4
torch.manual_seed(0)
logits = torch.randn(n_rows, n_cols, device=dev, dtype=torch.float32)
targets = torch.randint(0, n_cols, (n_rows,), device=dev)

# tie the row max across blocks: idx0 in block0 (small), idx1 in block1 (large)
idx0, idx1 = 10, 40000  # both == row max
results = []
for r in range(n_rows):
    logits[r] = -5.0
    logits[r, idx0] = 3.0
    logits[r, idx1] = 3.0

logits_in = logits.clone()
loss, pred = K.cross_entropy_with_predictions(logits_in, targets)

# reference
ref_pred = []
for r in range(n_rows):
    ref_pred.append(int(torch.argmin(logits[r])))  # not used
# proper reference: lowest argmax index
ref_pred = [int((logits[r] == logits[r].max()).nonzero()[0].item()) for r in range(n_rows)]
ref_loss = torch.nn.functional.cross_entropy(logits, targets, reduction="none")

out = {
    "BLOCK_SIZE": BLOCK,
    "n_cols": n_cols,
    "n_blocks": (n_cols + BLOCK - 1)//BLOCK,
    "kernel_pred": pred.tolist(),
    "ref_lowest_tie_pred": ref_pred,
    "tie_indices": [idx0, idx1],
    "loss_kernel": loss.tolist(),
    "loss_ref": ref_loss.tolist(),
    "loss_max_abs_err": float((loss - ref_loss).abs().max()),
    "metric": "exact predicted index vs contract's lowest-tied-index rule",
}
print(json.dumps(out, indent=2))
