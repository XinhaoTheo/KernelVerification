import sys, torch, json
sys.path.insert(0, "/root/cases/case_140")
import kernel as k
torch.cuda.init()
dev = "cuda"
g = torch.Generator().manual_seed(1234)
vals = torch.randint(0, 1 << 32, (8192,), dtype=torch.int64, generator=g).to(torch.uint32).to(dev)
# include large-value half explicitly (where sign/narrowing errors would show)
big = torch.arange(1<<31, (1<<32), step=1000003, dtype=torch.int64)[:4096].to(torch.uint32).to(dev)
x = torch.cat([vals, big])
out = k.run(x)
ref = k.reference(x)
exact = torch.equal(out.to(torch.int64), ref.to(torch.int64))
inv = (out[:,0].to(torch.int64)*7 + out[:,1].to(torch.int64)) != x.to(torch.int64)
print(json.dumps({"claim":"c1","mismatch_count": int(inv.sum()), "max_index_tested": 4294967295,
 "exact_vs_reference": bool(exact), "sample_out_first8": out[:8].tolist(),
 "sample_ref_first8": ref[:8].tolist()}))