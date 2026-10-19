import importlib.util, torch, json
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_118/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
dev = "cuda"
g = torch.Generator(device="cpu").manual_seed(2)
K, N, E = 33, 65, 8  # N=65 -> 3 column tiles (N>32)

def build(M, per_counts):
    # Q=1, token i routed to expert per_counts layout: sequential blocks of tokens per expert
    ids = torch.zeros((M,1), dtype=torch.int32)
    row = 0
    for e, cnt in enumerate(per_counts):
        for _ in range(cnt):
            ids[row, 0] = e
            row += 1
    assert row == M
    return ids

results = {}
for tag, counts in [("5blocks", [16,16,16,16,5,0,0,0]), ("7blocks", [16,16,16,16,16,16,1,0]), ("6blocks", [16,16,16,16,16,1,0,0])]:
    M = sum(counts)
    ids = build(M, counts).to(dev)
    x = ((torch.rand((M,K), generator=g)-.5)*0.9).half().to(dev)
    w = ((torch.rand((E,K,N), generator=g)-.5)*0.9).half().to(dev)
    rt = (torch.rand((M,1), generator=g)*0.9).float().to(dev)
    ref = k.reference(x, w, ids, rt)
    out = k.run(x, w, ids, rt)
    err = (out.double()-ref).abs()
    tol = 0.01 + 0.0002*ref.abs()
    results[tag] = {"M": M, "violations": int((err>tol).sum()),
                    "max_abs_err": float(err.max()), "max_tol": float(tol.max())}
print(json.dumps({"metric": "elementwise abs err vs 0.01+0.0002|ref| for 5/6/7 expert-block partial-group cases, N=65 (3 col tiles)", **results,
                  "passed_all": all(r["violations"]==0 for r in results.values())}))