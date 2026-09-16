import torch, json, importlib.util, sys
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_22/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
torch.manual_seed(0)
res = []
for K in [1,2,3,7,255,256,257,511,512,513,1000,1023,1024,4096,5000,100000,1<<20,(1<<20)+37]:
    a = torch.randn(K, device='cuda', dtype=torch.float32)
    b = torch.randn(K, device='cuda', dtype=torch.float32)
    ref = (a.double()*b.double()).sum().item()
    got = m.splitk_dot(a,b).item()
    # ones probe: exposes coverage directly (sum should equal K)
    ao = torch.ones(K, device='cuda'); bo = torch.ones(K, device='cuda')
    cov = m.splitk_dot(ao,bo).item()
    res.append(dict(K=K, ref=ref, got=got, abs_err=abs(got-ref),
                    rel=abs(got-ref)/max(1e-30,abs(ref)), coverage=cov, cov_err=cov-K))
print(json.dumps(res, indent=1))
