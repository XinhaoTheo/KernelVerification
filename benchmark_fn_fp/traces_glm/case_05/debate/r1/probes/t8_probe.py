import torch, json, sys
sys.path.insert(0, "/root/cases/case_05")
from kernel import count_tied_at_boundary

dev = "cuda" if torch.cuda.is_available() else "cpu"
pivot = 0.0
def run(vals):
    N = 8
    s = torch.full((N,), pivot, dtype=torch.float32, device=dev)
    s[:len(vals)] = torch.tensor(vals, dtype=torch.float32, device=dev)
    return count_tied_at_boundary(s, pivot)

out = {}
out["exact_tie"] = run([0.0005, 0.0005, 0.0005])
out["near_tie_case"] = run([0.0002, 0.0009])  # gap 0.0007 < EPS=1e-3
out["clearly_distinct"] = run([0.0002, 0.005])
out["empty_above"] = run([])
out["no_above_below"] = run([-1.0, -2.0])
print(json.dumps({"counts": out, "eps": 1e-3, "pivot": 0.0, "device": dev}))