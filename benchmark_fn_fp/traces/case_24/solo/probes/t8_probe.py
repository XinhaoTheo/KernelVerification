
import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_24/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
torch.manual_seed(1)
R,N=6,300
a=torch.randn(R,N,device='cuda'); b=torch.randn(R,N,device='cuda')
a[0]=0.0            # zero norm a
b[1]=0.0            # zero norm b
a[2]=0.0; b[2]=0.0  # both zero
a[3]=1e-25          # tiny norm
b[3]=1e-25
o=k.cosine_similarity(a,b)
A=a.double();B=b.double()
den=(A.pow(2).sum(1).sqrt()*B.pow(2).sum(1).sqrt()).clamp_min(1e-8)
r=(A*B).sum(1)/den
print(json.dumps({"out":[float(x) for x in o],"ref":[float(x) for x in r],
 "finite":bool(torch.isfinite(o).all()),
 "maxabs":float((o.double()-r).abs().max())},indent=1))
