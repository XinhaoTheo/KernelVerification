import torch, sys, json
sys.path.insert(0, "/root/cases/case_26")
from kernel import topk_mask

torch.manual_seed(2)
K = 3
N = 8
# Large-magnitude float32 scores around 1e10.
# ULP at 1e10 ~ 1e-3; residual bracket ~ range/2^40 ~ 1e10/1e12 ~ 1e-2.
# Gap between k-th (3rd) and (k+1)-th (4th) value: 4e-3  -> ULP < gap < residual.
base = torch.tensor([1.0e10, 0.9e10, 0.8e10, 0.8e10 - 0.004, 0.7e10, 0.6e10, 0.5e10, 0.4e10], device="cuda", dtype=torch.float32)
gap = float(base[2] - base[3])
scores = base.unsqueeze(0)
out = topk_mask(scores, K)
s = scores[0]
sorted_desc, _ = torch.sort(s, descending=True)
kth = sorted_desc[K-1].item()
kept_mask = out[0] != 0
kept_vals = s[kept_mask].tolist()
kept_count = int(kept_mask.sum().item())
# violation: kept entry strictly below the k-th largest value (and not equal to it)
bad = [v for v in kept_vals if v < kth]
residual_est = 2e10 / 2**40
result = {
  "claim": "c3 residual bisection bracket admits non-tied sub-cutoff entries",
  "k": K, "n_cols": N,
  "input_row": s.tolist(),
  "kth_largest_value": kth,
  "kept_count": kept_count,
  "kept_values": kept_vals,
  "kept_exceeds_k": kept_count > K,
  "bad_entries_below_kth_not_tied": bad,
  "num_bad": len(bad),
  "cutoff_gap": gap,
  "residual_bracket_est": residual_est,
  "gap_less_than_residual": gap < residual_est,
  "out_row": out[0].tolist(),
}
print(json.dumps(result))