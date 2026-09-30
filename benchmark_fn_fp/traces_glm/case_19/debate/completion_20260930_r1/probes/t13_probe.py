import torch, json, sys
sys.path.insert(0, '/root/cases/case_19')
from kernel import cross_entropy_with_predictions
n_rows, n_cols = 2, 40000
torch.manual_seed(0)
logits = torch.randn(n_rows, n_cols, device='cuda', dtype=torch.float32)
# row 0: max at index 0, tied copy at 40000-block boundary region (index 35000, in 2nd block)
logits[0] = -10.0
logits[0, 0] = 5.0
logits[0, 35000] = 5.0  # tie in second block (BLOCK_SIZE=32768)
# row 1: max in first block only (control)
logits[1] = torch.randn(n_cols, device='cuda')
target = torch.tensor([0, 5], device='cuda')
loss, predicted = cross_entropy_with_predictions(logits, target)
ref_pred0 = int(torch.argmax(logits[0]))  # torch argmax returns lowest index on ties
out = {"predicted_row0": int(predicted[0]), "expected_lowest_tied_row0": 0,
       "ref_argmax_row0": ref_pred0, "n_cols": n_cols,
       "tie_violation": int(predicted[0]) != 0, "predicted_row1": int(predicted[1])}
print(json.dumps(out))