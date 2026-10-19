import importlib.util, torch, json
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_118/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
dev = "cuda"
g = torch.Generator(device="cpu").manual_seed(2)
def build(M, E, Q, per_counts, K=33, N=65):
    ids = torch.zeros((M,Q), dtype=torch.int32)
    row = 0; e = 0
    for cnt in per_counts:
        for _ in range(cnt):
            for q in range(Q):
                ids[row, (e+q) % E] = (e+q) % E
            row += 1
        e += 1
    # fill remaining tokens with last expert to keep distinctness simple: use Q=1
    assert row == M
    return ids
# 5 expert blocks: counts 16,16,16,16,5 -> 5 blocks (not multiple of GROUP_M=4), N=65 (3 col tiles)
M,K,N,E,Q = 69, 33, 65, 8, 1
ids = build(M, E, Q, [16,16,16,16,5,0,0,0]).to(dev)
x = ((torch.rand((M,K), generator=g)-.5)*0.9).half().to(dev)
w = ((torch.rand((E,K,N), generator=g)-.5)*0.9).half().to(dev)
rt = (torch.rand((M,Q), generator=g)*0.9).float().to(dev)
ref = k.reference(x, w, ids, rt); out = k.run(x, w, ids, rt)
err = (out.double()-ref).abs(); tol = 0.01 + 0.0002*ref.abs()
r5 = {"violations_5blocks": int((err>tol).sum()), "max_abs_err_5blocks": float(err.max()), "max_tol_5blocks": float(tol.max())}
# 7 expert blocks: counts 16,16,16,16,16,16,1
M2 = 97
ids2 = build(M2, E, Q, [16,16,16,16,16,16,1,0]).to(dev)
x2 = ((torch.rand((M2,K), generator=g)-.5)*0.9).half().to(dev)
rt2 = (torch.rand((M2,Q), generator=g)*0.9).float().to(dev)
ref2 = k.reference(x2, w, ids2, rt2); out2 = k.run(x2, w, ids2, rt2)
err2 = (out2.double()-ref2).abs(); tol2 = 0.01 + 0.0002*ref2.abs()
r7 = {"violations_7blocks": int((err2>tol2).sum()), "max_abs_err_7blocks": float(err2.max()), "max_tol_7blocks": float(tol2.max())}
print(json.dumps({"metric":"elementwise abs err vs 0.01+0.0002|ref| for 5 and 7 expert-block partial-group cases, N=65", **r5, **r7, "passed_both": bool((err<=tol).all() and (err2<=tol2).all())}))