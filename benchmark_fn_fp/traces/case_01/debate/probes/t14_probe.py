
import json, importlib.util, torch
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_01/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
dev='cuda'; torch.manual_seed(1)
res={}

B,V=4,256
draft = torch.rand(B,V,device=dev)*0.01+0.01
target = draft*0.5                      # every residual strictly negative
q = torch.empty(B,V,device=dev).exponential_(1.0); inv_q = 1.0/q
out = m.sample_recovered_tokens(target, draft, inv_q)
resid = target-draft
ar = torch.arange(B,device=dev)
clamped = torch.clamp(resid, min=0.0)
ref_score = clamped*inv_q
res['all_residuals_strictly_negative']=bool((resid<0).all())
res['clamped_ref_score_all_zero']=bool((ref_score==0).all())
res['ref_score_max']=float(ref_score.max()); res['ref_score_min']=float(ref_score.min())
res['kernel_idx']=out.tolist()
res['kernel_selected_residual']=[float(x) for x in resid[ar,out]]
res['kernel_picked_negative_residual']=bool((resid[ar,out]<0).all())
# is kernel's pick exactly argmax of unclamped score?
un = resid*inv_q
res['kernel_matches_unclamped_argmax']=bool((out==un.argmax(dim=1)).all())
res['kernel_idx_nonzero_count']=int((out!=0).sum())
# reference-undefined check: does ANY token carry positive residual probability?
res['count_tokens_with_positive_residual']=int((resid>0).sum())
res['contract_valid_answer_exists']=bool((resid>0).any())

# reachability from normalized probability rows: can all residuals be strictly negative?
viol=0; trials=0
for t in range(200):
    a=torch.softmax(torch.randn(16,1024,device=dev),dim=-1)
    b=torch.softmax(torch.randn(16,1024,device=dev),dim=-1)
    r=a-b
    viol+=int(((r<0).all(dim=1)).sum()); trials+=16
res['normalized_rows_sampled']=trials
res['normalized_rows_all_strictly_negative']=viol
res['sum_check_target']=float(torch.softmax(torch.randn(1,8,device=dev),dim=-1).sum())
print(json.dumps(res))
