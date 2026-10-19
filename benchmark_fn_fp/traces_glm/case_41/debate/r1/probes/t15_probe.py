
import torch, json, math, sys, itertools
sys.path.insert(0, "/root/cases/case_114")
import kernel as K

dev = "cuda"
g = torch.Generator(device=dev).manual_seed(11)

def build(B, Q, HQ, HK, D, S, length, window, seed):
    gg = torch.Generator(device=dev).manual_seed(seed)
    P = 32  # enough pages, C=512/S>=ceil(length/S)
    C = 512 // S
    q = (torch.rand((B, Q, HQ, D), device=dev, generator=gg) * 2 - 1).half()
    k = (torch.rand((P, S, HK, D), device=dev, generator=gg) * 2 - 1).half()
    v = (torch.rand((P, S, HK, D), device=dev, generator=gg) * 2 - 1).half()
    # shuffled unique pages per batch (safe: no NaN, no alias concern)
    table = torch.zeros((B, C), device=dev, dtype=torch.int32)
    for b in range(B):
        table[b] = torch.randperm(P, generator=gg).to(dev)
        if C > P: table[b, P:] = 0
    lengths = torch.tensor([length] * B, dtype=torch.int32, device=dev)
    return q, k, v, table, lengths, window

worst = {"case": None, "ratio": 0.0}
failures = []
n_run = 0
for window in (1, 2, 3, 7, 8, 15, 16, 17, 31, 63, 64, 65, 127, 128, 129, 255, 256):
    for QT in (1, 2, 5, 16, 17, 33):
        for length in (QT, 33, 64, 65, 100, 200):
            if length < QT or length > 512: continue
            for group in (1, 2, 4):
                HQ, HK, D, S = group, 1, 64, 16
                B = 1
                try:
                    q, k, v, table, lengths, w = build(B, QT, HQ, HK, D, S, length, window, 1000 + window)
                    out = K.run(q, k, v, table, lengths, w)
                    torch.cuda.synchronize()
                    ref = K.reference(q, k, v, table, lengths, w)
                    abs_err = (out.double() - ref).abs()
                    tol = 0.003 + 0.003 * ref.abs()
                    r = float((abs_err / tol).max())
                    n_run += 1
                    if r > worst["ratio"]:
                        worst = {"case": {"window": window, "QT": QT, "length": length, "group": group}, "ratio": r}
                    if r > 1.0:
                        failures.append({"window": window, "QT": QT, "length": length, "group": group, "ratio": r})
                except Exception as e:
                    failures.append({"case": {"window": window, "QT": QT, "length": length, "group": group},
                                     "exception": str(e)[:120]})
print(json.dumps({"claim": "c3", "cases_run": n_run, "n_failures": len(failures),
                  "worst": worst, "failures": failures[:10]}))
