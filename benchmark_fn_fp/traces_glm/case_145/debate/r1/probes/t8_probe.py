import sys, json, torch
sys.path.insert(0, "/root/cases/case_145")
from kernel import run, reference

dev = "cuda"
def row(vals):
    t = torch.tensor(vals, dtype=torch.float32).to(torch.float16).to(dev)
    return t

# Row 0: [-0.0, +0.0, -1.0, -2.0] -> contract: max is 0, first zero (index 0) wins
# Row 1: [+0.0, -0.0, -1.0] -> first zero index 0
# Row 2: [-0.0, -0.0, -1.0] -> index 0 (kernel fine here, control)
x = torch.tensor([[float("-0.0"), 0.0, -1.0, -2.0],
                  [0.0, float("-0.0"), -1.0, -2.0],
                  [float("-0.0"), float("-0.0"), -1.0, -2.0]], dtype=torch.float32).to(torch.float16).to(dev)
out = run(x)
ref = reference(x)
res = {
  "rows": [[str(v) for v in r] for r in x.tolist()],
  "kernel_out": out.tolist(),
  "reference_out": ref.tolist(),
  "match": bool((out == ref).all().item()),
}
print(json.dumps(res))