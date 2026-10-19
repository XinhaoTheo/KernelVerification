import torch, json, sys
sys.path.insert(0, '/root/cases/case_19')
from kernel import cross_entropy_with_predictions
n_rows, n_cols = 2, 40000  # > 32768 so masked remainder blocks exist; also power-of-2 case
torch.manual_seed(0)
logits = torch.randn(n_rows, n_cols, device='cuda', dtype=torch.float32)
logits[0] = float('-inf')  # all -inf row
target = torch.tensor([0, 5], device='cuda')
loss, predicted = cross_entropy_with_predictions(logits, target)
out = {"n_cols": n_cols, "predicted_all_neg_inf_row": int(predicted[0]),
       "loss_all_neg_inf_row": float(loss[0]),
       "predicted_normal_row": int(predicted[1]),
       "out_of_range": bool(predicted[0] >= n_cols)}
print(json.dumps(out))