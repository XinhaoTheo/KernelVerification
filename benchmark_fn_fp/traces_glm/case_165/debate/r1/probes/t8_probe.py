import torch, json
torch.manual_seed(0)
# large-magnitude legal FP16 values: exponent 15..? x near 65504, and a dense set of large values
dev='cuda'
vals=[]
# all positive FP16 values with exponent >= 10 (x >= 1024), plus 65504 explicitly
import struct
all_bits=[(s,e,m) for s in (0,) for e in range(10,32) for m in range(16,1024)]
cand=[]
for s in (0,1):
    for e in range(15,32):
        for m in range(0,1024):
            if e==31 and m==0: continue
            cand.append((s,e,m))
# simpler: build all positive finite fp16 large values via torch
t=torch.tensor([65504.0],dtype=torch.float16)
big=[]
for e in range(10,16):
    for m in range(0,1024):
        # normal fp16 encoding
        val=m*2.0**(e-25)+2.0**e
        big.append(val)
x=torch.tensor(big+[65504.0],dtype=torch.float16).to(dev)
n=x.numel()
assert n<=131072
import importlib.util
spec=importlib.util.spec_from_file_location('kernel','/root/cases/case_165/kernel.py')
k=importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
out=k.run(x)
ref=torch.log1p(x.to(torch.float64))
err=(out.double()-ref).abs()
i=int(err.argmax())
mx=float(err.max())
tol=1e-6
print(json.dumps({
 'metric':'max_abs_err vs FP64 log1p over large-x subset (x>=1024 + 65504)',
 'n':n,'max_abs_err':mx,'tolerance':tol,'violations':int((err>tol).sum()),
 'worst_x':float(x[i]),'worst_ref':float(ref[i]),'worst_out':float(out[i]),
 'violates':bool(mx>tol)}))