import torch, json, importlib.util
dev='cuda'
# full legal FP16 grid: all finite values strictly > -1 (both signs, subnormals, zeros)
bits=torch.arange(65536,dtype=torch.int32).to(torch.uint16).to(torch.int32)
x=torch.tensor(bits.view(torch.uint16).numpy().view('float16').copy(),dtype=torch.float16)
# x computed from raw bit patterns
import numpy as np
arr=bits.numpy().astype(np.uint16)
xf=np.frombuffer(arr.tobytes(),dtype=np.float16).copy()
x=torch.from_numpy(xf).float()  # filter in fp32 first
finite=torch.isfinite(x)
gt=x>(-1.0)
mask=finite&gt
x=x[mask].to(torch.float16)
n=x.numel()
assert 1<=n<=131072, n
xc=x.to(dev)
spec=importlib.util.spec_from_file_location('kernel','/root/cases/case_165/kernel.py')
k=importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
out=k.run(xc)
ref=torch.log1p(xc.to(torch.float64))
err=(out.double()-ref).abs()
i=int(err.argmax())
mx=float(err.max()); tol=1e-6
viol=int((err>tol).sum())
worst_x=float(xc[i])
print(json.dumps({
 'metric':'exhaustive max_abs_err vs FP64 log1p over full legal FP16 grid',
 'n_legal_values':n,'max_abs_err':mx,'tolerance':tol,'violations':viol,
 'worst_x':worst_x,'worst_ref':float(ref[i]),'worst_out':float(out[i]),
 'contains_violation':bool(viol>0)}))