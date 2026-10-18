import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_154/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
g = torch.Generator(device="cpu").manual_seed(123)
import itertools
vals = [65504.0, -65504.0, 6.0e-8, -6.0e-8, 0.0, -0.0, 1.0, 65504.0, 5.96e-8]
import random
random.seed(7)
rows = list(itertools.product(vals, repeat=4))
rows = [r for r in rows if not (r[2] == 0 and r[3] == 0)]
random.shuffle(rows); rows = rows[:200000]
rows += [(65504, -65504, 65504, 65504), (65504, 0, 6e-8, 6e-8), (65504, 65504, 65504, 0), (0, 65504, 0, 6e-8)]
num = torch.tensor([[r[0], r[1]] for r in rows], dtype=torch.float16, device="cuda")
den = torch.tensor([[r[2], r[3]] for r in rows], dtype=torch.float16, device="cuda")
out = k.run(num, den)
nonfinite = ~torch.isfinite(out)
print(json.dumps({"n_rows": len(rows), "n_nonfinite": int(nonfinite.sum()),
                 "nonfinite_rows": torch.nonzero(nonfinite).flatten().tolist()[:20]}))