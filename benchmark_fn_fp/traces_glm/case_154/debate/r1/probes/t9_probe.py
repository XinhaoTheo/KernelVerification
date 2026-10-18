import torch, json, sys
sys.path.insert(0, "/root/cases/case_154")
import kernel as K

dev = "cuda"
g = torch.Generator(device="cpu").manual_seed(1234)
worst = None
worst_info = None
fails = 0
trials = 0
for t in range(200):
    n = 4096
    # mixture: uniform-log magnitudes over full FP16 range incl. subnormals, skewed denominators
    def samp():
        e = torch.empty(n,2).uniform_(-20, 15, generator=g)  # ~1e-20..3e15 -> clamp to fp16 later
        x = torch.sign(torch.empty(n,2).uniform_(-1,1,generator=g)) * (2.0**e)
        return x.to(torch.float16)
    num = samp()
    den = samp()
    # ensure denominator rows nonzero
    z = (den == 0).all(dim=1)
    den[z,0] = 1.0
    # occasional adversarial skewed denominators: one component ~1e5x the other
    if t % 4 == 0:
        m = torch.randperm(n, generator=g)[: n//2]
        den[m,1] = (den[m,0].float()*1e-4).to(torch.float16)
    num = num.to(dev); den = den.to(dev)
    out = K.run(num, den)
    ref = K.reference(num, den)
    err = (out.double()-ref.double()).abs()
    tol = 1e-7 + 5e-7*ref.abs()
    f = err > tol
    trials += 1
    if bool(f.any()):
        fails += 1
    rel = (err / tol)
    mx = float(rel.max())
    if worst is None or mx > worst:
        worst = mx
        idx = torch.unravel_index(rel.argmax(), rel.shape)
        worst_info = {
            "worst_ratio": mx,
            "row": int(idx[0]), "comp": int(idx[1]),
            "num_row": num.cpu()[idx[0]].tolist(),
            "den_row": den.cpu()[idx[0]].tolist(),
            "out": out.cpu()[idx[0]].tolist(),
            "ref": ref.cpu()[idx[0]].tolist(),
            "err": float(err.cpu()[idx[0],idx[1]]),
            "tol": float(tol.cpu()[idx[0],idx[1]]),
        }
print(json.dumps({"trials": trials, "trials_with_fail": fails,
                  "worst_err_over_tol": worst, "worst_case": worst_info}))
