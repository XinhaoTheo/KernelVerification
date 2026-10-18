
import torch, sys, json
sys.path.insert(0, "/root/cases/case_146")
from kernel import run

# FP16 values adjacent to boundaries 2.41421356 and 0.66, plus dense sweep in |x| in [0.5, 5]
target = torch.tensor(
    [0.6298828125,0.640625,0.650390625,0.6552734375,0.66015625,0.6650390625,0.6708984375,
     2.330078125,2.39453125,2.40625,2.41015625,2.4140625,2.41796875,2.42578125,2.44140625,2.47265625,
     2.5,2.75,3.0,3.5,4.0,5.0,
     0.5,0.55,0.6,0.66,0.7,0.75,0.8,0.9,1.0,1.1,1.25,1.5,1.75,2.0,2.25],
    dtype=torch.float64).to(torch.float16)
target = torch.cat([target, -target, target*0.0])  # duplicates fine
target = target.unique()
vals = target.cuda().contiguous()
print("boundary-focused values:", vals.numel(), file=sys.stderr)

out = run(vals)
ref = torch.atan(vals.to(torch.float64))
err = (out.to(torch.float64) - ref).abs()
fails = err > 1e-6
result = {
    "metric": "max abs error at FP16 values adjacent to reduction boundaries 2.414213562373095 and 0.66",
    "tolerance": 1e-6,
    "n_tested": vals.numel(),
    "max_abs_err": err.max().item(),
    "n_exceeding_tol": int(fails.sum().item()),
    "worst_inputs": [float(vals[i].item()) for i in torch.nonzero(fails).flatten()[:10].tolist()],
    "worst_input": float(vals[int(err.argmax().item())].item()),
}
print(json.dumps(result))
