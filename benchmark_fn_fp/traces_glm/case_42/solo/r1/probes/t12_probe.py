import sys, torch, json
sys.path.insert(0, "/root/cases/case_115")
import kernel

dev="cuda"
q,k,v,tbl,ln,window = kernel.make_inputs(device=dev, seed=7)
snap = [t.clone() for t in (q,k,v,tbl,ln)]
out = kernel.run(q,k,v,tbl,ln,window)
mut = []
for name,a,b in zip(("q","k","v","table","lengths"),(q,k,v,tbl,ln),snap):
    mut.append([name, bool(torch.equal(a.view(torch.int16) if a.dtype==torch.float16 else a, b.view(torch.int16) if b.dtype==torch.float16 else b))])
print(json.dumps({"metric":"bit-for-bit equality of each input after run()","mutated":mut, "any_mutated": any(m[1]==False for m in mut), "output_finite": bool(torch.isfinite(out).all())}))
