
import torch, json, importlib.util, sys
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_09/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

torch.manual_seed(0)
dev='cuda'
res={}

# Case A: explicit ties at the maximum
N=8
rows=[]
expected=[]
# row0: tie between idx 0 and 5 at max
r=torch.tensor([1.0,0.2,0.3,0.1,0.4,1.0,0.0,-1.0]); rows.append(r); expected.append(0)
# row1: tie between idx 2 and 3
r=torch.tensor([0.1,0.2,0.9,0.9,0.4,0.5,0.0,-1.0]); rows.append(r); expected.append(2)
# row2: all equal
r=torch.zeros(8); rows.append(r); expected.append(0)
# row3: unique max at 6
r=torch.tensor([0.1,0.2,0.3,0.4,0.5,0.6,0.9,0.7]); rows.append(r); expected.append(6)
# row4: tie between 1 and 7
r=torch.tensor([-1.0,2.0,0.3,0.4,0.5,0.6,0.9,2.0]); rows.append(r); expected.append(1)
scores=torch.stack(rows).to(dev)
out=m.sorted_topk_indices(scores,1).squeeze(1).cpu().tolist()
res['tied_cases']={'expected_lowest_idx':expected,'kernel':out}
res['tie_violations']=[i for i,(e,o) in enumerate(zip(expected,out)) if e!=o]

# Case B: random unique values, compare to torch.argmax
B,N=64,16
s=torch.randn(B,N,device=dev)
out2=m.sorted_topk_indices(s,1).squeeze(1)
ref2=s.argmax(dim=1)
res['random_unique_mismatch']=int((out2!=ref2).sum().item())

# Case C: quantized values -> many ties
s3=(torch.randint(0,3,(64,16),device=dev)).float()
out3=m.sorted_topk_indices(s3,1).squeeze(1)
mx=s3.max(dim=1,keepdim=True).values
ref3=(s3==mx).float().argmax(dim=1)   # lowest index attaining max
sel_vals=s3.gather(1,out3[:,None]).squeeze(1)
res['tied_rows_total']=int(((s3==mx).sum(1)>1).sum().item())
res['quantized_lowest_idx_mismatch']=int((out3!=ref3).sum().item())
res['quantized_value_mismatch']=int((sel_vals!=mx.squeeze(1)).sum().item())
res['examples']=[{'row':int(i),'scores':s3[i].cpu().tolist(),'kernel_idx':int(out3[i]),'expected_idx':int(ref3[i])} for i in (out3!=ref3).nonzero().squeeze(1)[:3].cpu().tolist()]
print(json.dumps(res,indent=1))
