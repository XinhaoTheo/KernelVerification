import numpy as np, torch, json, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/numerical_cases/case_q/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
q, c, e = k.make_inputs()
out = k.run(q, c, e)
# reference: float64 exact distances, smallest-index tie-break
d = ((c.double() - q.double())**2).sum(dim=1)
ref_winner = int(torch.argmin(d).item())
# kernel winner: find embedding row matching output
kernel_winner = int(torch.argmin((e - out.unsqueeze(0)).norm(dim=1)).item())
qdist = sorted(d.tolist())[:3]
print(json.dumps({"ref_winner": ref_winner, "kernel_winner": kernel_winner,
  "match": ref_winner == kernel_winner,
  "ref_dist_top3": qdist, "dist_spread": float(d.max()-d.min())}))