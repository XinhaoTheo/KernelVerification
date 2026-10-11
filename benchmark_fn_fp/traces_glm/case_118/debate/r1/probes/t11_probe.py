import importlib.util, torch, json
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_118/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
dev = "cuda"
# Worst-case magnitudes: all |x|=|w|=0.5, K=N=129, E=4, Q=4, weights 0.25 each (sum=1)
M,K,N,E,Q = 4, 129, 129, 4, 4
x = torch.full((M,K), 0.5, dtype=torch.half, device=dev)
w = torch.full((E,K,N), 0.5, dtype=torch.half, device=dev)
ids = torch.arange(M, dtype=torch.int32, device=dev).unsqueeze(1).repeat(1,Q).contiguous()
ids = (ids + torch.arange(Q, dtype=torch.int32, device=dev).unsqueeze(0)) % E
ids = ids.contiguous()
rt = torch.full((M,Q), 0.25, dtype=torch.float32, device=dev)
ref = k.reference(x, w, ids, rt)
out = k.run(x, w, ids, rt)
err = (out.double()-ref).abs()
tol = 0.01 + 0.0002*ref.abs()
print(json.dumps({"metric":"max_abs_err vs 0.01+0.0002*|ref|, worst-case 0.5-magnitude FP16, K=N=129, Q=4",
 "max_abs_target": float(ref.abs().max()), "max_abs_err": float(err.max()),
 "tol_at_max_err": float(tol.flatten()[err.argmax()]), "violations": int((err>tol).sum()),
 "M":M,"K":K,"N":N,"Q":Q,"passed": bool((err<=tol).all())}))