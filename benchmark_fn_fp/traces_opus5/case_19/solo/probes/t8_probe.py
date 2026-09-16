
import torch, sys, importlib.util, json
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_19/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

torch.manual_seed(0)
n_rows, n_cols = 4, 40000   # > 32768 -> multiple blocks
logits = torch.randn(n_rows, n_cols, device="cuda", dtype=torch.float32) * 0.1
# place exact ties at a low index (block 0) and a high index (block 1)
low_idx, high_idx = 100, 35000
logits[:, low_idx] = 5.0
logits[:, high_idx] = 5.0
target = torch.tensor([0,1,2,3], device="cuda", dtype=torch.int64)

lg = logits.clone()
loss, pred = k.cross_entropy_with_predictions(lg, target)
ref_loss = torch.nn.functional.cross_entropy(logits, target, reduction="none")
# reference argmax lowest index
mx = logits.max(dim=1, keepdim=True).values
ref_pred = (logits == mx).float().argmax(dim=1)

# single-block control
n_cols2 = 1024
l2 = torch.randn(2, n_cols2, device="cuda")*0.1
l2[:, 10] = 3.0; l2[:, 900] = 3.0
t2 = torch.tensor([0,1], device="cuda", dtype=torch.int64)
_, pred2 = k.cross_entropy_with_predictions(l2.clone(), t2)

print(json.dumps({
 "block_size_multi": min(32768, 1<<(n_cols-1).bit_length()),
 "pred_multiblock": pred.tolist(),
 "ref_pred_lowest": ref_pred.tolist(),
 "low_idx": low_idx, "high_idx": high_idx,
 "loss_kernel": loss.tolist(),
 "loss_ref_per_row": ref_loss.tolist(),
 "loss_ratio": (loss/ref_loss).tolist(),
 "pred_singleblock": pred2.tolist(),
}, indent=1))
