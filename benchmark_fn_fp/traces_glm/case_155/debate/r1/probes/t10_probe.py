import importlib.util, torch, json
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_155/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
dev="cuda"
def exact(nume,den):
    a,b=nume.to(torch.float64).unbind(1); c,d=den.to(torch.float64).unbind(1)
    s=c*c+d*d
    return torch.stack(((a*c+b*d)/s,(b*c-a*d)/s),dim=1)
# exact FP16 inputs; ratio d/c = 1/3 inexact in FP32; b*ratio error amplified then divides near-cancelling sum
c=3.0; d=1.0; b=65504.0
a = torch.tensor(-b*d/c, dtype=torch.float64).to(torch.float16).item()  # quantized nearest
num=torch.tensor([[a,b]]).to(dev); den=torch.tensor([[c,d]]).to(dev)
tgt=exact(num.cpu(),den.cpu()); out=k.run(num,den).cpu().double()
tol=1e-7+5e-7*tgt.abs(); err=(out-tgt).abs()
# also sweep c over odd small values
rows=[]
for cc in [3.0,5.0,7.0,11.0,13.0,17.0,1021.0,2053.0]:
    aa=torch.tensor(-b*1.0/cc,dtype=torch.float64).to(torch.float16).item()
    n=torch.tensor([[aa,b]]).to(dev); dn=torch.tensor([[cc,1.0]]).to(dev)
    t=exact(n.cpu(),dn.cpu()); o=k.run(n,dn).cpu().double()
    e=(o-t).abs(); tl2=1e-7+5e-7*t.abs()
    rows.append({"c":cc,"num":[aa,b],"exact":t.tolist()[0],"kernel":o.tolist()[0],
                 "abs_err":e.tolist()[0],"tol":tl2.tolist()[0],"ratio":float(e.max()/tl2.max())})
print(json.dumps({"headline":{"num":[a,b],"den":[c,d],"exact":tgt.tolist()[0],
  "kernel":out.tolist()[0],"abs_err":err.tolist()[0],"tol":tol.tolist()[0],
  "err_over_tol":float((err/tol).max())},"sweep":rows}))