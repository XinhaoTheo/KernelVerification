import importlib.util, torch, json
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_155/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
dev="cuda"
def exact(nume,den):
    a,b=nume.to(torch.float64).unbind(1); c,d=den.to(torch.float64).unbind(1)
    s=c*c+d*d
    return torch.stack(((a*c+b*d)/s,(b*c-a*d)/s),dim=1)
# exact FP16 inputs; ratio d/c inexact in FP32; a chosen so real numerator nearly cancels
b=65504.0
rows=[]
for cc,dd in [(3.0,1.0),(5.0,2.0),(7.0,3.0),(11.0,5.0),(13.0,6.0),(17.0,8.0),(1021.0,512.0),(2053.0,1024.0)]:
    aa = torch.tensor(-b*dd/cc, dtype=torch.float64).to(torch.float16).item()
    n=torch.tensor([[aa,b]],dtype=torch.float16).to(dev)
    dn=torch.tensor([[cc,dd]],dtype=torch.float16).to(dev)
    t=exact(n.cpu(),dn.cpu()); o=k.run(n,dn).cpu().double()
    e=(o-t).abs(); tl2=1e-7+5e-7*t.abs()
    rows.append({"c":cc,"d":dd,"num":[aa,b],"exact":t.tolist()[0],"kernel":o.tolist()[0],
                 "abs_err":e.tolist()[0],"tol":tl2.tolist()[0],
                 "err_over_tol":float((e/tl2).max())})
worst=max(rows,key=lambda r:r["err_over_tol"])
print(json.dumps({"worst":worst,"sweep":rows}))