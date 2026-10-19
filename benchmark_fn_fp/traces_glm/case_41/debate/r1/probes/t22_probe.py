import torch, json, sys
sys.path.insert(0, "/root/cases/case_114")
import kernel as K

dev = "cuda"

def build(B, Q, HQ, HK, D, S, length, window, seed):
    gg = torch.Generator().manual_seed(seed)  # CPU generator, tensors moved to dev
    P, C = 32, 512 // S
    q = ((torch.rand((B, Q, HQ, D), generator=gg) * 2 - 1).to(dev)).half()
    k = ((torch.rand((P, S, HK, D), generator=gg) * 2 - 1).to(dev)).half()
    v = ((torch.rand((P, S, HK, D), generator=gg) * 2 - 1).to(dev)).half()
    table = torch.zeros((B, C), dtype=torch.int32)
    for b in range(B):
        perm = torch.randperm(P, generator=gg)
        table[b, :P] = perm.to(torch.int32)
    table = table.to(dev)
    lengths = torch.tensor([length] * B, dtype=torch.int32, device=dev)
    return q, k, v, table, lengths, window

worst = {"case": None, "ratio": 0.0}
failures = []
n_run = 0
for window in (1, 2, 3, 7, 8, 15, 16, 17, 31, 63, 64, 65, 127, 128, 129, 255, 256):
    for QT in (1, 2, 5, 16, 17, 33):
        for length in (QT, 33, 64, 65, 100, 200, 512):
            if length < QT or length > 512: continue
            for group in (1, 2, 4):
                HQ, HK, D, S = group, 1, 64, 16
                try:
                    q, k, v, table, lengths, w = build(1, QT, HQ, HK, D, S, length, window, 1000 + window)
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
                    failures.append({"window": window, "QT": QT, "length": length, "group": group, "exception": str(e)[:100]})
print(json.dumps({"claim": "c3", "cases_run": n_run, "n_failures": len(failures),
                  "worst": worst, "failures": failures[:10]}))