import importlib.util, torch, json
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_155/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
dev="cuda"
def exact(nume,den):
    a,b=nume.to(torch.float64).unbind(1); c,d=den.to(torch.float64).unbind(1)
    s=c*c+d*d
    return torch.stack(((a*c+b*d)/s,(b*c-a*d)/s),dim=1)
sub = float(torch.tensor(5.96e-8, dtype=torch.float16))
cases=[
  [[65504.0,0.0],[sub,0.0]],
  [[65504.0,65504.0],[sub,0.0]],
  [[-65504.0,65504.0],[0.0,sub]],
  [[65504.0,65504.0],[sub,sub]],
  [[-65504.0,65504.0],[sub,-sub]],
  [[65504.0,-65504.0],[-sub,-sub]],
  [[0.0,65504.0],[sub,sub]],
]
res=[]
for nc,dc in cases:
    num=torch.tensor([nc],dtype=torch.float16).to(dev)
    den=torch.tensor([dc],dtype=torch.float16).to(dev)
    out=k.run(num,den).cpu().double()
    tgt=exact(num.cpu(),den.cpu())
    finite=bool(torch.isfinite(out).all())
    tol=1e-7+5e-7*tgt.abs(); err=(out-tgt).abs()
    res.append({"num":nc,"den":dc,"kernel":out.tolist()[0],"exact":tgt.tolist()[0],
                "finite":finite,"abs_err":err.tolist()[0],"tol":tol.tolist()[0],
                "err_over_tol":float((err/tol).max())})
print(json.dumps({"cases":res,"max_exact_quotient_magnitude":65504.0/5.96e-8,
                  "fp32_max":3.4028234663852886e38}))