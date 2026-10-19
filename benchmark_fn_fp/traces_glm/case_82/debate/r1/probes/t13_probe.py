import importlib.util, torch, json
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_155/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
dev = "cuda"
def exact(nume, den):
    a,b = nume.to(torch.float64).unbind(1); c,d = den.to(torch.float64).unbind(1)
    s = c*c+d*d
    return torch.stack(((a*c+b*d)/s,(b*c-a*d)/s),dim=1)
worst=None; nviol=0; total=0
g = torch.Generator().manual_seed(0)
for trial in range(300):
    # |c| ~ |d| so ratio ~ 1 and a = -b*d/c is large (~ +-65504)
    c = (torch.rand(1,generator=g)*2-1).double()
    d = c*(1.0 - torch.rand(1,generator=g).double()*4e-4)  # ratio ~ 1, inexact
    b = torch.tensor(-65504.0).double() if trial%2==0 else torch.tensor(65504.0).double()
    a = -b*d/c  # near-exact cancellation of real numerator, both operands ~65504
    num = torch.tensor([[a.item(),b.item()]]).to(torch.float16).to(dev)
    den = torch.tensor([[c.item(),d.item()]]).to(torch.float16).to(dev)
    tgt = exact(num.cpu(), den.cpu())
    out = k.run(num, den).cpu().double()
    tol = 1e-7 + 5e-7*tgt.abs(); err=(out-tgt).abs()
    total += err.numel(); nviol += int((err>tol).sum())
    r = float((err/tol).max())
    if worst is None or r>worst[0]:
        worst=(r,num.cpu().tolist(),den.cpu().tolist(),out.tolist(),tgt.tolist(),err.tolist(),tol.tolist())
print(json.dumps({"max_err_over_tol":worst[0],"n_violations":nviol,"n_components":total,
  "example":{"num":worst[1],"den":worst[2],"kernel_out":worst[3],"exact":worst[4],
             "abs_err":worst[5],"tol":worst[6]}}))