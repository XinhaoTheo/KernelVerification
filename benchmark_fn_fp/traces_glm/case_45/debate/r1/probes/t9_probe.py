import importlib.util, torch, json
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_118/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
dev = "cuda"
g = torch.Generator(device="cpu").manual_seed(1)
# E=3, M=16, Q=1, all tokens routed to expert 0; experts 1,2 zero routes.
M,K,N,E,Q = 16, 33, 65, 3, 1
x = ((torch.rand((M,K), generator=g)-.5)*0.9).half().to(dev)
w = ((torch.rand((E,K,N), generator=g)-.5)*0.9).half().to(dev)
ids = torch.zeros((M,Q), dtype=torch.int32, device=dev)
rt = torch.full((M,Q), 0.9999, dtype=torch.float32, device=dev)
ref = k.reference(x, w, ids, rt)
out = k.run(x, w, ids, rt)
err = (out.double()-ref).abs()
tol = 0.01 + 0.0002*ref.abs()
print(json.dumps({"metric":"max_abs_err vs tol over all entries", "max_abs_err": float(err.max()),
 "tol_at_max": float(tol.flatten()[err.argmax()]), "violations": int((err>tol).sum()),
 "setup":"E=3,M=16,Q=1,all->expert0,experts1&2 zero routes", "passed": bool((err<=tol).all())}))