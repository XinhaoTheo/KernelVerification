
import json, importlib.util, torch
spec=importlib.util.spec_from_file_location("k","/root/cases/case_22/kernel.py")
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
rows=[]; worst_count=0.0; worst_rel=0.0
torch.manual_seed(2)
for K in [1,2,3,7,100,255,256,257,511,512,513]:
    o=torch.ones(K,device="cuda",dtype=torch.float32)
    cnt=m.splitk_dot(o,o).item()
    a=torch.randn(K,device="cuda",dtype=torch.float32)
    b=torch.randn(K,device="cuda",dtype=torch.float32)
    v=m.splitk_dot(a,b).item()
    ref=torch.dot(a.double(),b.double()).item()
    sumabs=torch.sum((a.double()*b.double()).abs()).item()
    rel=abs(v-ref)/max(sumabs,1e-30)
    rows.append({"K":K,"ones_count":cnt,"count_minus_K":cnt-K,"randn_val":v,"fp64_ref":ref,
                 "abs_err":abs(v-ref),"rel_err_vs_sumabs":rel})
    worst_count=max(worst_count,abs(cnt-K)); worst_rel=max(worst_rel,rel)
print(json.dumps({"rows":rows,"worst_abs_count_deviation":worst_count,"worst_rel_err_vs_sumabs":worst_rel}))
