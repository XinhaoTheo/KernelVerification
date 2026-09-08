
import json, importlib.util, torch
spec = importlib.util.spec_from_file_location("kern", "/root/cases/case_26/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

res = {}
def analyze(name, row_list, k):
    x = torch.tensor([row_list], dtype=torch.float32, device="cuda")
    out = m.topk_mask(x, k)
    o = out[0].tolist(); r = x[0].tolist()
    # kept = output value equals input value (inputs contain no zeros)
    kept = [i for i in range(len(r)) if o[i] == r[i] and r[i] != 0.0]
    finite = torch.tensor([v for v in r if v == v], dtype=torch.float64)
    v = torch.topk(torch.tensor(r, dtype=torch.float64), k).values[-1].item()
    viol = [r[i] for i in kept if r[i] < v]
    res[name] = {
        "n": len(r), "k": k, "kth_largest": v,
        "kept_count": len(kept),
        "kept_values": [r[i] for i in kept],
        "n_kept_strictly_below_kth": len(viol),
        "out_row": o,
    }

# (a) pre-masked logits with -inf
analyze("neg_inf_row", [3.0, 1.0, float("-inf"), float("-inf"), 5.0, 2.0, float("-inf"), 0.5], 2)
# (b) +inf present with multiplicity 1 < k
analyze("pos_inf_row", [float("inf"), 3.0, 1.0, 5.0, 2.0, 0.5, 4.0, 6.0], 3)
# control: all finite
analyze("finite_control", [3.0, 1.0, 7.0, 5.0, 2.0, 0.5, 4.0, 6.0], 3)
print(json.dumps(res))
